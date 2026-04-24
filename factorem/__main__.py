from typing import Optional
import argparse
import starfile
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import jax

from . import geometry
from . import image
from . import analysis

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
    #parser.add_argument(
    #    '-o', '--output',
    #    required=True,
    #    metavar='DIR',
    #    help='Output directory'
    #)
    parser.add_argument(
        '--angular_spacing',
        metavar='DEG',
        type=float,
        default=5.0,
        help='Average spacing between projection directions. In degrees.'
    )
    parser.add_argument(
        '--group_angle',
        metavar='DEG',
        type=float,
        default=5.0,
        help='Maximum angle deviation in the projection directions. In degrees.'
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
        '--batch_size',
        type=int,
        default=256,
        help='Number of particles processed concurrently'
    )
    parser.add_argument(
        '--padding_factor',
        type=float,
        default=2.0,
        help='Padding factor to increase spectral resolution'
    )


    return parser.parse_args(argv)

def run(args: argparse.Namespace):
    star = starfile.read(args.input)
    particles_md = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']
    amplitude_contrast = optics.at[0, 'rlnAmplitudeContrast']
    spherical_aberration = optics.at[0, 'rlnSphericalAberration']
    voltage = optics.at[0, 'rlnVoltage']
    
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
    
    direction_count =  geometry.estimate_projection_direction_count(
        math.radians(args.angular_spacing)
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
        math.radians(args.group_angle)
    )
    
    loader = analysis.DataLoader(
        image_locations=image_locations,
        image_prefix=args.prefix,
        rotations=rotations,
        shifts=shifts,
        defocus=defocus,
        padded_box_size=round(args.padding_factor*288), # TODO
        pixel_size_a=pixel_size,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        amplitude_contrast=amplitude_contrast
    )
    
    processor: analysis.Processor = None
    if False:
        processor = analysis.PCA(n_components=5)
    else:
        processor = analysis.SpectralEmbedding(
            n_components=5,
            batch_size=args.batch_size,
            kernel='median'
        )
    
    #frequency_mask = analysis.butterworth_2d(padded_box_size, 0.25, 2)
    for i in range(direction_count):
        if len(groups[i]) < 100:
            print(f'Skipping direction {i}')
            continue
        
        y = processor.fit_transform(
            loader=loader,
            indices=groups[i],
            direction_matrix=direction_matrices[i]
        )
        y = jax.device_get(y)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(y[:,0], y[:,1], y[:,2])
        #plt.show()
    
        
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
