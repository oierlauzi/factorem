from typing import Optional
import argparse
import starfile
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

import factorem.image

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='test_reader'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='STAR',
        help='Input STAR file with particle data'
    )
    parser.add_argument(
        '--prefix'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256
    )
    parser.add_argument(
        '--warm',
        type=int,
        default=64
    )
    parser.add_argument(
        '--test',
        type=int,
        default=256
    )
    
    return parser.parse_args(argv)

def run(args: argparse.Namespace):
    star = starfile.read(args.input)
    particles_md = star['particles']
    
    image_locations = particles_md['rlnImageName'].map(factorem.image.ImageLocation.parse)
    reader = factorem.image.BatchReader(args.prefix)
    
    rng = np.random.default_rng(0)
    indices = np.arange(len(image_locations))
    
    for _ in range(args.warm):
        batch_indices = rng.choice(indices, size=args.batch_size, replace=False)
        batch_locations = image_locations[batch_indices]

        _ = reader.read_batch(batch_locations)
    
    durations = np.empty(args.test)
    for i in range(args.test):
        batch_indices = rng.choice(indices, size=args.batch_size, replace=False)
        batch_locations = image_locations[batch_indices]

        start = time.perf_counter()
        _ = reader.read_batch(batch_locations)
        end = time.perf_counter()
        durations[i] = (end - start) / args.batch_size
    
    plt.boxplot(durations)
    plt.show()

    
def main(argv=None) -> Optional[int]:
    args = _parse_args(argv)
    run(args)

if __name__ == '__main__':
    sys.exit(main())

