"""Command line interface.

It is made of commands, ``factorem COMMAND ...``. A command is a function
that adds its parser to those of the others, with the function that runs it
as the ``run`` default, and is listed in `build_parser`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from importlib import metadata
from typing import Any

# By a probabilistic PCA of the images, or by the multidimensional scaling
# of the Mahalanobis distances between them.
EMBEDDINGS = ('ppca', 'mahalanobis-mds')

# By the common lines of the eigenimages of the directions, or by the
# particles that they share.
SYNCHRONIZATIONS = ('common-lines', 'burer-monteiro')


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


def _add_analyze(commands: Any, shared: argparse.ArgumentParser) -> None:
    parser = commands.add_parser(
        'analyze',
        parents=[shared],
        help='Embed the particles according to their heterogeneity.',
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
        '--embedding',
        choices=EMBEDDINGS,
        default=EMBEDDINGS[0],
        help='How the particles of each projection direction are embedded: '
        'by a probabilistic PCA with the CTF in its model, or by the '
        'multidimensional scaling of the Mahalanobis distances between '
        'the images (default: %(default)s).',
    )
    parser.add_argument(
        '--synchronization',
        choices=SYNCHRONIZATIONS,
        default=SYNCHRONIZATIONS[0],
        help='How the projection directions are brought to a common frame: '
        'by the common lines of their eigenimages, or by the particles '
        'they share, with the relaxation of Burer and Monteiro (default: '
        '%(default)s).',
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
    parser.set_defaults(run=_analyze)


def _analyze(args: argparse.Namespace) -> int:
    from .embedding import Embedder, MahalanobisMDS, ProbabilisticPCA
    from .io import StarDataset, StarFormatError
    from .pipeline import analyze
    from .synchronization import (
        BurerMonteiroSynchronizer,
        CommonLineSynchronizer,
        Synchronizer,
    )

    embedder: Embedder = (
        MahalanobisMDS()
        if args.embedding == 'mahalanobis-mds'
        else ProbabilisticPCA()
    )
    synchronizer: Synchronizer = (
        BurerMonteiroSynchronizer()
        if args.synchronization == 'burer-monteiro'
        else CommonLineSynchronizer()
    )
    try:
        dataset = StarDataset.read(args.input, root=args.root)
        embedding = analyze(
            dataset.particles,
            diameter=args.diameter,
            resolution=args.resolution,
            components=args.components,
            embedder=embedder,
            synchronizer=synchronizer,
            device=args.device,
            scratch=args.scratch,
        )
    except (StarFormatError, OSError, ValueError) as error:
        print(f'factorem: error: {error}', file=sys.stderr)
        return 1
    dataset.write(args.output, embedding.coordinates, embedding.group_count)
    return 0


def _add_annotate(commands: Any, shared: argparse.ArgumentParser) -> None:
    parser = commands.add_parser(
        'annotate',
        parents=[shared],
        help='Look at the embedded particles and sort them into classes.',
        description=(
            'Show the particles that factorem analyze embedded on a plane, '
            'in a window, and sort them into classes there. It needs the '
            "annotate extra: pip install 'factorem[annotate]'."
        ),
    )
    parser.add_argument(
        'input',
        help='STAR file with the coordinates of each particle in the '
        'factoremEmbedding column, as factorem analyze writes it.',
    )
    parser.add_argument(
        'output',
        help='STAR file to write, with the class of each particle: from '
        'one on, and zero for the particles without coordinates.',
    )
    parser.add_argument(
        '--label',
        default='factoremClass',
        metavar='NAME',
        help='Column to write the classes in. With rlnClassNumber, RELION '
        'takes them for its own (default: %(default)s).',
    )
    parser.set_defaults(run=_annotate)


def _annotate(args: argparse.Namespace) -> int:
    from .io import StarFormatError, read_embedding, write_classes

    try:
        from .annotation import AnnotationWindow
    except ImportError as error:
        print(
            f'factorem: error: {error}. The annotation needs matplotlib '
            "and scikit-learn: pip install 'factorem[annotate]'.",
            file=sys.stderr,
        )
        return 1

    try:
        tables, coordinates = read_embedding(args.input)
        window = AnnotationWindow(
            coordinates,
            save=lambda classes: write_classes(
                args.output, tables, classes, args.label
            ),
        )
        classes = window.run()
    except (StarFormatError, OSError, ValueError, RuntimeError) as error:
        print(f'factorem: error: {error}', file=sys.stderr)
        return 1
    if classes is None:
        print(
            'factorem: error: The window was closed without saving, so '
            'nothing was written.',
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='factorem',
        description='Heterogeneity analysis of cryo-EM single particles.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {metadata.version("factorem")}',
    )

    # What every command takes.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Report the progress.',
    )

    commands = parser.add_subparsers(
        dest='command', metavar='COMMAND', required=True
    )
    _add_analyze(commands, shared)
    _add_annotate(commands, shared)
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
    status: int = args.run(args)
    return status
