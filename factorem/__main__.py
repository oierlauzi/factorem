from typing import Optional, Sequence
import argparse
import starfile
import numpy as np
import sys
import math
import tqdm
import sklearn.decomposition
import logging
import jax

logger = logging.getLogger(__name__)

from . import geometry
from . import image
from . import analysis
from . import synchronization
from .bsr_array_builder import BsrArrayBuilder

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='factorem',
        description='Heterogeneity analysis pipeline for CryoEM',
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file with particle data'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        metavar='STAR',
        help='Output STAR file with embedding data'
    )
    parser.add_argument(
        '--prefix',
        metavar='DIR',
        help='Prefix for the MRC binary files.'
    )
    parser.add_argument(
        '--min_particles',
        type=int,
        default=100,
        help='Minimum number of particles for analysis'
    )
    parser.add_argument(
        '--padding_factor',
        type=float,
        default=2.0,
        help='Padding factor to increase spectral resolution'
    )
    parser.add_argument(
        '--embedding',
        choices=['pca', 'spectral'],
        default='pca',
        help='Embedding method to use for dimensionality reduction'
    )
    parser.add_argument(
        '--components',
        type=int,
        default=6,
        help='Number of components for dimensionality reduction'
    )
    parser.add_argument(
        '--diameter',
        type=float,
        help='Particle diameter in angstrom'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=4.0,
        help='Maximum resolution in angstrom'
    )
    parser.add_argument(
        '--aperture_index',
        type=float,
        default=1.0,
        help='Projection direction aperture of the Shannon angle'
    )
    parser.add_argument(
        '--direction_index',
        type=float,
        default=1.0,
        help='Projection direction sampling in terms of the Shannon angle. Should be greater or equal to the aperture index.'
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="gpu:0", 
        help="Device to use. Format: 'cpu' or 'gpu:X' (e.g., 'gpu:0', 'gpu:1')"
    )

    return parser.parse_args(argv)

def select_device(index: str):
    try:
        if ":" in index:
            backend_name, device_id = index.split(":")
            device_id = int(device_id)
        else:
            backend_name = index
            device_id = 0  # Default to first device if no ID provided
            
        backend_name = backend_name.lower()
    except ValueError:
        raise ValueError(f"Invalid device format: '{index}'. Use 'cpu' or 'gpu:N'.")

    try:
        available_devices = jax.devices(backend_name)
        target_device = available_devices[device_id]
        logger.info(f"Successfully selected device: {target_device}")
        
    except RuntimeError:
        available_backends = jax.backends()
        raise RuntimeError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available_backends}"
        )
    except IndexError:
        num_devices = len(jax.devices(backend_name))
        raise IndexError(
            f"Device ID {device_id} out of bounds for backend '{backend_name}'. "
            f"Found only {num_devices} device(s)."
        )
        
    return target_device

def _image_count_groups(groups: Sequence[Sequence[int]], n_images: int) -> np.ndarray:
    result = np.zeros(n_images, dtype=np.int64)

    for group in groups:
        result[group] += 1
    
    return result

def run(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    
    logger.info('Reading input')
    star = starfile.read(args.input)
    particles_md = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']
    amplitude_contrast = optics.at[0, 'rlnAmplitudeContrast']
    spherical_aberration = optics.at[0, 'rlnSphericalAberration']
    voltage = optics.at[0, 'rlnVoltage']
    box_size = optics.at[0, 'rlnImageSize']
    device = select_device(args.device)
    
    image_locations = particles_md['rlnImageName'].map(image.ImageLocation.parse)
    rotations = geometry.euler_zyz_to_matrix(
        np.deg2rad(np.asarray(particles_md['rlnAngleRot'])),
        np.deg2rad(np.asarray(particles_md['rlnAngleTilt'])),
        np.deg2rad(np.asarray(particles_md['rlnAnglePsi'])),
    )
    shifts = (1/pixel_size) * np.stack(
        (
            np.asarray(particles_md['rlnOriginXAngst']), 
            np.asarray(particles_md['rlnOriginYAngst'])
        ), 
        axis=1
    )
    defocus_u = np.asarray(particles_md['rlnDefocusU'])
    defocus_v = np.asarray(particles_md['rlnDefocusV'])
    defocus = 0.5*(defocus_u + defocus_v)
    image_count = len(image_locations)
    
    shannon_angle = args.resolution / args.diameter
    direction_spacing = args.direction_index * shannon_angle
    direction_aperture = args.aperture_index * shannon_angle
    max_freq = pixel_size / args.resolution
    
    logger.info('Shannon angle: %.2fdeg' % math.degrees(shannon_angle))
    logger.info('Direction spacing: %.2fdeg' % math.degrees(direction_spacing))
    logger.info('Direction aperture: %.2fdeg' % math.degrees(direction_aperture))
    logger.info('Maximum (digital) frequency: %.2f' % max_freq)

    logger.info('Computing projection directions')
    direction_count =  geometry.estimate_projection_direction_count(
        direction_spacing
    )
    directions = geometry.sample_projection_directions(direction_count)
    direction_matrices = geometry.euler_zyz_to_matrix(
        directions[:,0],
        directions[:,1],
        np.array(0)
    )
    groups = geometry.group_projection_directions(
        rotations[:,2,:],
        direction_matrices[:,2,:],
        direction_aperture
    )
    for group in groups:
        group.sort()
    
    logger.info('Setting up directional analysis')
    padded_box_size = round(args.padding_factor*box_size)
    loader = analysis.DataLoader(
        image_locations=image_locations,
        image_prefix=args.prefix,
        rotations=rotations,
        shifts=shifts,
        defocus=defocus
    )
    preprocessor = analysis.Preprocessor(
        padded_box_size=padded_box_size,
        pixel_size_a=pixel_size,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        max_freq=max_freq,
        grain_size=256
    )
    
    component_count = args.components
    if args.embedding == 'pca':
        processor = analysis.PCA(
            n_components=component_count, 
            particle_size=box_size
        )
    else:
        assert args.embedding == 'spectral'
        processor = analysis.SpectralEmbedding(
            n_components=component_count,
            kernel='median'
        )

    jobs = []
    significant_groups = []
    for i, group in enumerate(groups):
        if len(group) < args.min_particles:
            logger.info(f'Skipping direction group {i}')
            continue
        
        significant_groups.append(group)
        jobs.append(
            analysis.Job(
                key=i,
                indices=group,
                direction_matrix=direction_matrices[i],
            )
        )
    analyzed_direction_count = len(significant_groups)
    image_multiplicity = _image_count_groups(significant_groups, image_count)
    skipped_images = np.argwhere(image_multiplicity<1).squeeze()
    if len(skipped_images) > 0:
        logger.warning(f'The following images were not analyzed due to insufficient group population: {skipped_images.tolist()}')

    logger.info('Analyzing directional groups')
    builder = BsrArrayBuilder((analyzed_direction_count*component_count, image_count))
    runner = analysis.PipelinedRunner(
        loader=loader,
        preprocessor=preprocessor,
        processor=processor,
        device=device,
        prefetch=4
    )
    
    progress = tqdm.tqdm(total=analyzed_direction_count, unit='dir')
    for job, y in runner.run(jobs, sequential=True):
        i = job.key
        indices = groups[i]
        assert np.all(indices[:-1] < indices[1:])
        for j, index in enumerate(indices):
            builder.add_block(index, np.asarray(y[j,:,None]))
        builder.next_block_row()
        progress.update(1)
    progress.close()

    embeddings = builder.build()
    similarities = embeddings @ embeddings.T
    similarities /= abs(similarities).max()
    
    logger.info('Synchronizing')
    synchronization_transform, _ = synchronization.burer_monteiro_ortho_group_synchronization(
        similarities,
        synchronization.burer_monteiro_random_start(
            n=analyzed_direction_count,
            k=component_count,
            p=2*component_count+1
        )
    )
    
    logger.info('Correcting orientations')
    embeddings = synchronization.correct_embeddings(
        embeddings=embeddings,
        transforms=synchronization_transform
    )
    
    similarities = embeddings @ embeddings.T
    similarities /= abs(similarities).max()
    
    logger.info('Averaging')
    unified_embedding = synchronization.average_embeddings(
        embeddings=embeddings, 
        max_iter=0 # TODO: Set it 16 when fixed.
    )
    
    logger.info('Computing PCA for the output')
    pca = sklearn.decomposition.PCA(n_components=component_count)
    unified_embedding = pca.fit_transform(unified_embedding)

    particles_md['factoremEmbedding'] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in unified_embedding
    ]
    particles_md['factoremGroupCount'] = image_multiplicity
    star['particles'] = particles_md
    starfile.write(star, args.output)

def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
