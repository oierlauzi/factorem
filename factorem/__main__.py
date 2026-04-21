from typing import Optional
import argparse
import starfile
import numpy as np
import pandas as pd
import sys
import math

from . import geometry

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
        '-s', '--pixel_size',
        required=True,
        metavar='ANGST',
        type=float,
        help='Pixel size in angstroms'
    )
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


    return parser.parse_args(argv)

def _matrices_from_md(
    md: pd.DataFrame,
    sampling_rate: float
) -> np.ndarray:
    result = np.zeros((len(md), 4, 4))
    
    geometry.euler_zyz_to_matrix(
        np.deg2rad(md['rlnAngleRot']),
        np.deg2rad(md['rlnAngleTilt']),
        np.deg2rad(md['rlnAnglePsi']),
        out=result[:,:3,:3]
    )
    
    rotations = result[:,:3,:3]
    shift = np.stack(
        (md['rlnOriginXAngst'], md['rlnOriginYAngst'], np.zeros(len(md))), 
        axis=1
    )
    shift /= sampling_rate
    
    np.matmul(rotations, shift.T, out=result[:,:3,3].T)
    
    result[:,3,3] = 1
    return result
    
def run(args: argparse.Namespace):
    particles_md = starfile.read(args.input)
    transforms = _matrices_from_md(particles_md, args.sampling_rate)
    
    direction_count =  geometry.estimate_projection_direction_count(
        math.radians(args.angular_spacing)
    )
    directions = geometry.sample_projection_directions(direction_count)
    direction_vectors = geometry.spherical_to_cartesian(
        directions[:,0],
        directions[:,1]
    )
    groups = geometry.group_projection_directions(
        transforms[:,:3,2], # TODO determine transform's Z
        direction_vectors,
        math.radians(args.group_angle)
    )
    
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
