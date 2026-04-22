from typing import Optional
import argparse
import starfile
import numpy as np
import pandas as pd
import sys
import math
import logging
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

    return parser.parse_args(argv)

def run(args: argparse.Namespace):
    star = starfile.read(args.input)
    particles_md = star['particles']
    optics = star['optics']
    pixel_size = optics.at[0, 'rlnImagePixelSize']
    
    image_locations = particles_md['rlnImageName'].map(image.ImageLocation.parse)
    rotations = geometry.euler_zyz_to_matrix(
        np.deg2rad(particles_md['rlnAngleRot']),
        np.deg2rad(particles_md['rlnAngleTilt']),
        np.deg2rad(particles_md['rlnAnglePsi']),
    )
    shifts = (1/pixel_size) * np.stack(
        (particles_md['rlnOriginXAngst'], particles_md['rlnOriginYAngst']), 
        axis=1
    )
    
    direction_count =  geometry.estimate_projection_direction_count(
        math.radians(args.angular_spacing)
    )
    directions = geometry.sample_projection_directions(direction_count)
    direction_vectors = geometry.spherical_to_cartesian(
        directions[:,0],
        directions[:,1]
    )
    groups = geometry.group_projection_directions(
        rotations[:,2,:],
        direction_vectors,
        math.radians(args.group_angle)
    )
    
    batch_size = 16384 # TODO
    reader = image.BatchReader(args.prefix)
    for i in range(direction_count):
        particle_indices = groups[i]
        n = len(particle_indices)
        start = 0
        
        direction_rot = directions[i,0]
        direction_tilt = directions[i,1]
        direction_psi = 0.0
        direction_matrix = geometry.euler_zyz_to_matrix(
            np.array(direction_rot),
            np.array(direction_tilt),
            np.array(direction_psi)
        )
        
        while start < n:
            end = min(start + batch_size, n)
            indices = particle_indices[start:end]

            batch_images = jnp.asarray(reader.read_batch(image_locations[indices]))
            batch_rotations = rotations[indices]
            batch_shifts = shifts[indices]
            centre = np.array(batch_images.shape[1:]) / 2
            
            rotation2d = geometry.align_inplane(direction_matrix, batch_rotations)
            affine = geometry.make_affine(rotation2d, batch_shifts, centre)
            affine = np.linalg.inv(affine)
            
            transformed_images = analysis.apply_affine_batch(batch_images, jnp.asarray(affine))
            
            start = end
        
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
