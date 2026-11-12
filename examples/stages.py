"""The stages of the analysis, one by one.

`factorem.analyze` chains them with its defaults. Calling them one by one
leaves room to replace any of them: `embed_directions` takes whatever
embeds the particles of a direction, and anything that returns a transform
for each direction can stand for the synchronizer.

    python stages.py particles.star embedded.star --diameter 180
"""

from __future__ import annotations

import argparse
import logging

import factorem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--diameter', type=float, required=True)
    parser.add_argument('--resolution', type=float, default=16.0)
    parser.add_argument('--components', type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = factorem.StarDataset.read(args.input)
    particles = dataset.particles
    design = factorem.Design()

    # The images, read once at four pixels to the resolution.
    stack = factorem.load_images(
        particles, args.resolution / design.oversampling
    )

    # Projection directions a Shannon angle apart, each with the particles
    # that view the molecule from within a cone around it.
    shannon_angle = args.resolution / args.diameter
    internal = max(args.components, design.min_components)
    groups = factorem.group_by_direction(
        particles.rotations,
        factorem.sample_directions(design.direction_spacing * shannon_angle),
        design.aperture * shannon_angle,
        min_size=design.images_per_component * internal,
        usable=stack.usable,
    )

    # An embedding of the particles of each direction, as 2D images.
    directional = factorem.embed_directions(
        stack,
        particles,
        groups,
        resolution=args.resolution,
        mask_diameter=design.mask_scale * args.diameter,
        components=internal,
        embedder=factorem.ProbabilisticPCA(),
    )

    # The transform that takes each of them to a frame common to all.
    transforms = factorem.BurerMonteiroSynchronizer()(directional)

    # Their average, along its principal axes.
    merged = factorem.merge_directions(directional, transforms, len(particles))
    coordinates, variances = factorem.principal_components(
        merged, args.components
    )
    print('Variance along each axis:', variances.round(3))
    dataset.write(args.output, coordinates, groups.multiplicity(len(particles)))


if __name__ == '__main__':
    main()
