"""Scores for a latent embedding. All of them look only at the nearest
neighbours of each particle, so they are invariant to rotations of the latent
space and do not assume a shape for the conformational manifold."""

import numpy as np
import sklearn.neighbors


def neighbours(z: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k nearest neighbours of every row, excluding itself."""
    index = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    index.fit(z)
    return index.kneighbors(z, return_distance=False)[:, 1:]


def circular_error(nn: np.ndarray, states: np.ndarray, period: int) -> float:
    """Mean circular distance between the state of a particle and the states
    of its neighbours. 0 is perfect, period / 4 is chance."""
    delta = np.abs(states[nn] - states[:, None])
    return float(np.minimum(delta, period - delta).mean())


def label_accuracy(nn: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Leave-one-out majority vote of the neighbours. Reports plain and class
    balanced accuracy."""
    classes, encoded = np.unique(labels, return_inverse=True)
    votes = np.zeros((len(labels), len(classes)), dtype=np.int32)
    for column in encoded[nn].T:
        votes[np.arange(len(labels)), column] += 1
    correct = votes.argmax(axis=1) == encoded
    recall = [correct[encoded == c].mean() for c in range(len(classes))]
    return {'acc': float(correct.mean()), 'bal_acc': float(np.mean(recall))}


def explained_variance(nn: np.ndarray, target: np.ndarray) -> float:
    """Fraction of the variance of ``target`` that the mean of the neighbours
    predicts (leave-one-out k-NN regression R2, pooled over columns)."""
    target = target.reshape(len(target), -1)
    residual = target - target[nn].mean(axis=1)
    return float(
        1.0
        - np.square(residual).sum() / np.square(target - target.mean(0)).sum()
    )


def evaluate(
    name: str,
    z: np.ndarray,
    truth: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None = None,
    nuisance: dict[str, np.ndarray] | None = None,
    valid: np.ndarray | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Score embedding ``z`` of dataset ``name``.

    ``reference`` holds other embeddings (RECOVAR) and ``nuisance`` holds
    quantities an embedding should not encode (defocus, viewing direction).
    """
    if valid is None:
        valid = np.ones(len(z), dtype=bool)
    valid = valid & np.isfinite(z).all(axis=1)
    out = {'coverage': float(valid.mean())}

    def knn(mask):
        return neighbours(z[mask], k)

    nn = knn(valid)
    if name == 'igg':
        out['knn_err'] = circular_error(nn, truth['state'][valid], period=100)
    elif name == 'e10076':
        for key in ('major', 'minor'):
            # Score on the particles the authors classified.
            mask = valid & (truth[key] >= 0)
            for metric, value in label_accuracy(
                knn(mask), truth[key][mask]
            ).items():
                out[f'{key}_{metric}'] = value
    elif name == 'e10180':
        for metric, value in label_accuracy(nn, truth['kept'][valid]).items():
            out[f'kept_{metric}'] = value

    for key, ref in (reference or {}).items():
        mask = valid & np.isfinite(ref).all(axis=1)
        nn_mask = nn if mask[valid].all() else knn(mask)
        out[f'r2_{key}'] = explained_variance(nn_mask, ref[mask])
    for key, value in (nuisance or {}).items():
        out[f'nuis_{key}'] = explained_variance(nn, value[valid])
    return out
