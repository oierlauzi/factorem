"""Score the embedding that the factorem command line wrote to a STAR file.

    python score_star.py <dataset> <factorem_output.star>
"""
import json
import sys

import numpy as np
import starfile

import datasets
import metrics


def read_embedding(path: str) -> np.ndarray:
    particles = starfile.read(path, always_dict=True)['particles']
    return np.array([json.loads(v) for v in particles['factoremEmbedding']])


if __name__ == '__main__':
    name, path = sys.argv[1:3]
    z = read_embedding(path)
    zs = np.load(datasets.DATASETS[name].recovar)
    reference = {'recovar4': zs['z4'], 'recovar10': zs['z10']}
    scores = metrics.evaluate(name, z, datasets.ground_truth(name, len(z)), reference)
    print(name, path, json.dumps({k: round(v, 4) for k, v in scores.items()}))
