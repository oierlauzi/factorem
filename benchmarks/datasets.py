"""The datasets used to benchmark factorem and what is known about each."""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

BENCH = '/data/olauzirika/factorem-bench'


@dataclass(frozen=True)
class Dataset:
    name: str
    star: str
    prefix: str
    diameter: float  # Particle diameter [A], a physical property.
    recovar: str  # Exported RECOVAR latent coordinates.
    bad: tuple = ()  # Corrupt particles (0-based).


DATASETS: Dict[str, Dataset] = {
    # CryoBench IgG-1D: 100 conformations x 1000 particles around a closed
    # circular motion; particle i is in state i // 1000.
    'igg': Dataset(
        name='igg',
        star='/home/olauzirika/Datasets/IgG-1D/snr0.01/snr0.01_rln4.star',
        prefix='/home/olauzirika/Datasets/IgG-1D/snr0.01',
        diameter=180.0,
        recovar=f'{BENCH}/recovar/igg_full.npz',
    ),
    # Ribosome LSU assembly intermediates, Fourier cropped from 320 to 128 px.
    'e10076': Dataset(
        name='e10076',
        star='/data/olauzirika/empiar10076b/particles.128.star',
        prefix='/data/olauzirika/empiar10076b',
        diameter=280.0,
        recovar=f'{BENCH}/recovar/10076_full.npz',
    ),
    # Pre-catalytic spliceosome, Fourier cropped from 320 to 128 px.
    'e10180': Dataset(
        name='e10180',
        star='/data/olauzirika/empiar10180/particles.128.star',
        prefix='/data/olauzirika/empiar10180',
        diameter=400.0,
        recovar=f'{BENCH}/recovar/10180_full.npz',
        bad=(308902,),
    ),
}


def ground_truth(name: str, count: int) -> Dict[str, np.ndarray]:
    """Per-particle references to score an embedding against."""
    if name == 'igg':
        return {'state': np.arange(count) // 1000}
    if name == 'e10076':
        labels = f'{BENCH}/labels'
        return {
            # Davis et al. 2016 classes; -1 marks particles they discarded.
            'major': np.load(f'{labels}/published_labels_major.npz')['arr_0'].astype(int),
            'minor': np.load(f'{labels}/published_labels.npz')['arr_0'].astype(int),
        }
    if name == 'e10180':
        # Particles cryoDRGN kept after filtering its latent space.
        kept = np.zeros(count, dtype=int)
        kept[np.load(f'{BENCH}/labels/cdrgn_10180_filtered.ind.npz')['arr_0']] = 1
        return {'kept': kept}
    raise KeyError(name)
