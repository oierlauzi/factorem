from typing import Optional
import argparse
import starfile
import numpy as np
import pandas as pd
import sys
import math

from . import geometry
from . import image

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
    
    images = particles_md['rlnImageName'].map(image.ImageLocation.parse)
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
        directions[:,1],
        directions[:,0]
    )
    groups = geometry.group_projection_directions(
        rotations[:,2,:],
        direction_vectors,
        math.radians(args.group_angle)
    )
    
    batch_size = 512
    reader = image.BatchReader(args.prefix)
    for group in groups:
        n = len(group)
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            indices = group[start:end]

            batch_images = reader.read_batch(images[indices])
            batch_rotations = rotations[indices]
            batch_shifts = shifts[indices]

            start = end
        
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
