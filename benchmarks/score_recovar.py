"""Score the RECOVAR embeddings with the same metrics used for factorem."""
import json
import sys

import numpy as np

import datasets
import metrics

for name in sys.argv[1:]:
    dataset = datasets.DATASETS[name]
    zs = np.load(dataset.recovar)
    count = len(zs['z4'])
    truth = datasets.ground_truth(name, count)
    reference = {'recovar4': zs['z4'], 'recovar10': zs['z10']}
    for key in zs.files:
        scores = metrics.evaluate(name, zs[key], truth, reference)
        print(name, key, json.dumps({k: round(v, 4) for k, v in scores.items()}), flush=True)
