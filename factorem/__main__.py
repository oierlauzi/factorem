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
import sklearn.manifold
import sklearn.decomposition

from . import geometry
from . import image
from . import analysis
from . import ctf
from .data_loader import DataLoader

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

def process_direction(
    direction_matrix: np.ndarray,
    indices: np.ndarray,
    loader: DataLoader,
    batch_size: int
):
    n = len(indices)
    
    distances2 = jnp.empty((n, n))
    start0 = 0
    while start0 < n:
        end0 = min(start0 + batch_size, n)
        batch0_indices = indices[start0:end0]
        batch0_images, batch0_ctfs = loader.load(batch0_indices, direction_matrix)
    
        start1 = 0
        while start1 < start0:
            end1 = min(start1 + batch_size, n)
            batch1_indices = indices[start1:end1]
            batch1_images, batch1_ctfs = loader.load(batch1_indices, direction_matrix)
            
            tile_distances2 = analysis.crossed_pairwise_distance2(
                batch0_images,
                batch0_ctfs,
                batch1_images,
                batch1_ctfs
            )
            distances2 = distances2.at[start0:end0,start1:end1].set(tile_distances2)
            distances2 = distances2.at[start1:end1,start0:end0].set(tile_distances2.T)
            
            start1 = end1
    
        tile_distances2 = analysis.self_pairwise_distance2(
            batch0_images,
            batch0_ctfs
        )
        distances2 = distances2.at[start0:end0,start0:end0].set(tile_distances2)

        start0 = end0
    
    #plt.hist(distances2.flatten())
    #plt.show()
    
    affinity = analysis.local_scaling_kernel(distances2)
    spectral_embedding = sklearn.manifold.SpectralEmbedding(n_components=3, affinity='precomputed')
    y = spectral_embedding.fit_transform(affinity)
    plt.hist2d(y[:,0], y[:,1])
    plt.show()

    #laplacian = analysis.compute_laplacian(affinity)
    #eig_vals, eig_vecs = jnp.linalg.eigh(laplacian)
    #plt.scatter(eig_vecs[:,-1], eig_vecs[:,-2])
    #plt.show()
    
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
    defocus = 0.5*(defocus_u + defocus_v)
    
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
    
    loader = DataLoader(
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
    
    #frequency_mask = analysis.butterworth_2d(padded_box_size, 0.25, 2)
    batch_size = args.batch_size
    for i in range(direction_count):
        if len(groups[i]) < 100:
            continue
        
        process_direction(
            direction_matrix=direction_matrices[i],
            indices=groups[i],
            loader=loader,
            batch_size=batch_size
        )
    
        
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())
