"""Export the latent coordinates of a RECOVAR run to a plain .npz file.

Run with the Python of the RECOVAR environment:
    python export_recovar.py <recovar_output_dir> <out.npz>
"""
import sys

import numpy as np
from recovar import utils

embeddings = utils.pickle_load(sys.argv[1] + '/model/embeddings.pkl')
out = {}
for zdim, z in embeddings['zs'].items():
    z = np.real(np.asarray(z))
    out[f'z{zdim}'] = z.reshape(len(z), -1)
np.savez(sys.argv[2], **out)
print({k: v.shape for k, v in out.items()})
