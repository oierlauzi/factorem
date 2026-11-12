"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from importlib import metadata


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f'{text} is not positive.')
    return value


def _count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'{text} is not a count.')
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='factorem',
        description=(
            'Embed the particles of a consensus refinement according to '
            'their heterogeneity.'
        ),
    )
    parser.add_argument(
        'input',
        help='STAR file of RELION with the particles, their alignment and '
        'their CTF.',
    )
    parser.add_argument(
        'output',
        help='STAR file to write, with the coordinates of each particle in '
        'the factoremEmbedding column.',
    )
    parser.add_argument(
        '--diameter',
        type=_positive,
        required=True,
        metavar='ANGSTROMS',
        help='Diameter of the particle.',
    )
    parser.add_argument(
        '--resolution',
        type=_positive,
        default=16.0,
        metavar='ANGSTROMS',
        help='Resolution of the analysis (default: %(default)s).',
    )
    parser.add_argument(
        '--components',
        type=_count,
        default=6,
        metavar='COUNT',
        help='Coordinates to give each particle (default: %(default)s).',
    )
    parser.add_argument(
        '--root',
        metavar='DIRECTORY',
        help='Directory that the images named in the STAR file are relative '
        'to (default: the working directory).',
    )
    parser.add_argument(
        '--device',
        metavar='NAME',
        help='Device to compute on, such as cpu or gpu:0 (default: the '
        'first GPU, or the CPU without one).',
    )
    parser.add_argument(
        '--scratch',
        metavar='DIRECTORY',
        help='Directory to keep the working images in, instead of in memory.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Report the progress.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {metadata.version("factorem")}',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'
    )
    # Only the progress of the analysis, not that of the libraries under it.
    logging.getLogger(__package__).setLevel(
        logging.INFO if args.verbose else logging.WARNING
    )
    # JAX would otherwise take most of the memory of the GPU for itself,
    # whatever it ends up needing. It reads this when it is imported.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from .io import StarDataset, StarFormatError
    from .pipeline import analyze

    try:
        dataset = StarDataset.read(args.input, root=args.root)
        embedding = analyze(
            dataset.particles,
            diameter=args.diameter,
            resolution=args.resolution,
            components=args.components,
            device=args.device,
            scratch=args.scratch,
        )
    except (StarFormatError, OSError, ValueError) as error:
        print(f'factorem: error: {error}', file=sys.stderr)
        return 1
    dataset.write(args.output, embedding.coordinates, embedding.group_count)
    return 0
