from typing import Optional, Sequence
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
from . import ctf

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

def process_direction(
    direction: np.ndarray,
    image_locations: Sequence[image.ImageLocation],
    rotations: np.ndarray,
    shifts: np.ndarray,
    defocus: np.ndarray,
    indices: np.ndarray,
    reader: image.BatchReader,
    ctf_context: ctf.CtfContext,
    padded_box_size: int,
    batch_size: int
):
    n = len(indices)
    
    direction_rot = direction[0]
    direction_tilt = direction[1]
    direction_psi = 0.0
    direction_matrix = geometry.euler_zyz_to_matrix(
        np.array(direction_rot),
        np.array(direction_tilt),
        np.array(direction_psi)
    )
    
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]

        batch_images = jnp.asarray(reader.read_batch(image_locations[batch_indices]))
        batch_rotations = rotations[batch_indices]
        batch_shifts = shifts[batch_indices]
        batch_defocus = defocus[batch_indices]
        _, box_size_y, box_size_x = batch_images.shape
        centre = np.array((box_size_y/2, box_size_x/2))
        
        rotation2d = geometry.align_inplane(direction_matrix, batch_rotations)
        affine = geometry.make_affine(rotation2d, batch_shifts, centre)
        affine = np.linalg.inv(affine)
        
        transformed_images = analysis.apply_affine_batch(batch_images, jnp.asarray(affine))
        transformed_images = analysis.pad_images_2d(transformed_images, padded_box_size)
        
        ctf_images = ctf.compute_ctf_image_2d(jnp.asarray(batch_defocus), padded_box_size, ctf_context)
        print(ctf_images.shape)
        plt.imshow(ctf_images[0])
        plt.show()
        
        start = end

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
        np.deg2rad(particles_md['rlnAngleRot']),
        np.deg2rad(particles_md['rlnAngleTilt']),
        np.deg2rad(particles_md['rlnAnglePsi']),
    )
    shifts = (1/pixel_size) * np.stack(
        (particles_md['rlnOriginXAngst'], particles_md['rlnOriginYAngst']), 
        axis=1
    )
    defocus_u = particles_md['rlnDefocusU']
    defocus_v = particles_md['rlnDefocusV']
    defocus = 10*0.5*(defocus_u + defocus_v)
    
    ctf_context = ctf.CtfContext(
        pixel_size_a=pixel_size,
        spherical_aberration_mm=spherical_aberration,
        q0=amplitude_contrast,
        voltage_kv=voltage
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
    
    batch_size = 256
    reader = image.BatchReader(args.prefix)
    padded_box_size = 256 # TODO
    for i in range(direction_count):
        process_direction(
            direction=directions[i],
            image_locations=image_locations,
            rotations=rotations,
            shifts=shifts,
            defocus=defocus,
            indices=groups[i],
            batch_size=batch_size,
            reader=reader,
            padded_box_size=padded_box_size,
            ctf_context=ctf_context
        )
    
        
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
