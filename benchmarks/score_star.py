"""Score the embedding that the factorem command line wrote to a STAR file.

python score_star.py <dataset> <factorem_output.star>
"""

import json
import sys

import numpy as np
import starfile

import datasets
import metrics
from factorem.io import EMBEDDING_LABEL, parse_embedding


def read_embedding(path: str) -> np.ndarray:
    tables = starfile.read(path, always_dict=True)
    (particles,) = (t for t in tables.values() if EMBEDDING_LABEL in t)
    return parse_embedding(particles[EMBEDDING_LABEL])


if __name__ == '__main__':
    name, path = sys.argv[1:3]
    z = read_embedding(path)
    zs = np.load(datasets.DATASETS[name].recovar)
    reference = {'recovar4': zs['z4'], 'recovar10': zs['z10']}
    scores = metrics.evaluate(
        name, z, datasets.ground_truth(name, len(z)), reference
    )
    print(name, path, json.dumps({k: round(v, 4) for k, v in scores.items()}))
