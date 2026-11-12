from dataclasses import dataclass
from typing import List, Optional
import argparse
import logging
import math
import sys

import jax
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.decomposition
import starfile
import tqdm

from . import analysis
from . import geometry
from . import synchronization

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class Optics:
    pixel_size: float
    box_size: int
    voltage: float
    spherical_aberration: float
    amplitude_contrast: float

    @staticmethod
    def from_star(optics: pd.DataFrame) -> 'Optics':
        row = optics.iloc[0]
        return Optics(
            pixel_size=float(row['rlnImagePixelSize']),
            box_size=int(row['rlnImageSize']),
            voltage=float(row['rlnVoltage']),
            spherical_aberration=float(row['rlnSphericalAberration']),
            amplitude_contrast=float(row['rlnAmplitudeContrast']),
        )


def _particle_rotations(particles: pd.DataFrame) -> np.ndarray:
    return geometry.euler_zyz_to_matrix(
        np.deg2rad(particles['rlnAngleRot'].to_numpy()),
        np.deg2rad(particles['rlnAngleTilt'].to_numpy()),
        np.deg2rad(particles['rlnAnglePsi'].to_numpy()),
    )


def _particle_shifts(particles: pd.DataFrame, pixel_size: float) -> np.ndarray:
    origin = particles[['rlnOriginXAngst', 'rlnOriginYAngst']].to_numpy()
    return origin / pixel_size


def _particle_defocus(particles: pd.DataFrame) -> np.ndarray:
    return 0.5 * (
        particles['rlnDefocusU'].to_numpy() + particles['rlnDefocusV'].to_numpy()
    )


def _make_jobs(
    rotations: np.ndarray,
    spacing: float,
    aperture: float,
    min_particles: int,
) -> List[analysis.Job]:
    """Group particles by projection direction, keeping populated groups."""
    count = geometry.estimate_projection_direction_count(spacing)
    directions = geometry.sample_projection_directions(count)
    direction_matrices = geometry.euler_zyz_to_matrix(
        directions[:, 0], directions[:, 1], np.array(0)
    )
    groups = geometry.group_projection_directions(
        rotations[:, 2, :], direction_matrices[:, 2, :], aperture
    )

    jobs = []
    for i, group in enumerate(groups):
        if len(group) < min_particles:
            logger.info(f'Skipping direction group {i}')
            continue
        # Sorted indices keep reads of the same stack close together.
        jobs.append(analysis.Job(
            key=i,
            indices=np.sort(group),
            direction_matrix=direction_matrices[i],
        ))
    return jobs


def _make_processor(args: argparse.Namespace, box_size: int) -> analysis.Processor:
    if args.embedding == 'pca':
        return analysis.PCA(n_components=args.components, particle_size=box_size)
    assert args.embedding == 'spectral'
    return analysis.SpectralEmbedding(
        n_components=args.components,
        kernel='median'
    )


def stack_embeddings(
    jobs: List[analysis.Job],
    results: List[np.ndarray],
    image_count: int,
) -> scipy.sparse.bsr_array:
    """Block matrix with a block row per direction and a block column per
    particle, holding each particle's embedding in each of its directions."""
    component_count = results[0].shape[1]
    return scipy.sparse.bsr_array(
        (
            np.concatenate(results)[:, :, None],
            np.concatenate([job.indices for job in jobs]),
            np.cumsum([0] + [len(job.indices) for job in jobs]),
        ),
        shape=(len(jobs) * component_count, image_count),
    )


def _unify_embeddings(
    embeddings: scipy.sparse.bsr_array,
    direction_count: int,
    component_count: int,
) -> np.ndarray:
    """Align the per-direction embeddings with each other and average them."""
    similarities = embeddings @ embeddings.T
    similarities /= abs(similarities).max()

    logger.info('Synchronizing')
    transforms, _ = synchronization.burer_monteiro_ortho_group_synchronization(
        similarities,
        synchronization.burer_monteiro_random_start(
            n=direction_count,
            k=component_count,
            p=2*component_count + 1
        )
    )

    logger.info('Correcting orientations')
    embeddings = synchronization.correct_embeddings(
        embeddings=embeddings,
        transforms=transforms
    )

    logger.info('Averaging')
    unified = synchronization.average_embeddings(
        embeddings=embeddings,
        max_iter=0 # TODO: Set it 16 when fixed.
    )

    logger.info('Computing PCA for the output')
    pca = sklearn.decomposition.PCA(n_components=component_count)
    return pca.fit_transform(unified)


def run(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)

    logger.info('Reading input')
    star = starfile.read(args.input, always_dict=True)
    particles = star['particles']
    optics = Optics.from_star(star['optics'])
    device = select_device(args.device)

    rotations = _particle_rotations(particles)
    image_count = len(particles)

    shannon_angle = args.resolution / args.diameter
    direction_spacing = args.direction_index * shannon_angle
    direction_aperture = args.aperture_index * shannon_angle
    max_freq = optics.pixel_size / args.resolution

    logger.info('Shannon angle: %.2fdeg' % math.degrees(shannon_angle))
    logger.info('Direction spacing: %.2fdeg' % math.degrees(direction_spacing))
    logger.info('Direction aperture: %.2fdeg' % math.degrees(direction_aperture))
    logger.info('Maximum (digital) frequency: %.2f' % max_freq)

    logger.info('Computing projection directions')
    jobs = _make_jobs(
        rotations, direction_spacing, direction_aperture, args.min_particles
    )
    group_count = np.bincount(
        np.concatenate([job.indices for job in jobs]), minlength=image_count
    )
    skipped_images = np.flatnonzero(group_count == 0)
    if len(skipped_images) > 0:
        logger.warning(f'The following images were not analyzed due to insufficient group population: {skipped_images.tolist()}')

    logger.info('Setting up directional analysis')
    dataset = analysis.make_dataset(
        image_names=particles['rlnImageName'],
        rotations=rotations,
        shifts=_particle_shifts(particles, optics.pixel_size),
        defocus=_particle_defocus(particles),
        image_prefix=args.prefix,
    )
    preprocessor = analysis.Preprocessor(
        padded_box_size=round(args.padding_factor * optics.box_size),
        pixel_size_a=optics.pixel_size,
        voltage_kv=optics.voltage,
        spherical_aberration_mm=optics.spherical_aberration,
        amplitude_contrast=optics.amplitude_contrast,
        max_freq=max_freq,
    )
    processor = _make_processor(args, optics.box_size)

    logger.info('Analyzing directional groups')
    stream = analysis.analyze_directions(
        dataset, jobs, preprocessor, processor, device
    )
    results = [y for _, y in tqdm.tqdm(stream, total=len(jobs), unit='dir')]

    embeddings = stack_embeddings(jobs, results, image_count)
    unified_embedding = _unify_embeddings(embeddings, len(jobs), args.components)

    particles['factoremEmbedding'] = [
        '[' + ', '.join(f'{v:.4e}' for v in row) + ']'
        for row in unified_embedding
    ]
    particles['factoremGroupCount'] = group_count
    star['particles'] = particles
    starfile.write(star, args.output)


def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)


if __name__ == '__main__':
    sys.exit(main())
