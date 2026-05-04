from typing import Sequence
import numpy as np
import scipy.sparse

def _compute_averages(
    measurements: scipy.sparse.bsr_array, 
    gains: np.ndarray, 
    sigma2: np.ndarray,
    m: int,
    p: int
) -> np.ndarray:
    n = measurements.shape[1]
    numerator = np.zeros((n, p))
    denominator = np.zeros((n, p))
    for i in range(m):
        gain = gains[i]
        noise2 = sigma2[i]
        start = measurements.indptr[i]
        end = measurements.indptr[i+1]
        indices = measurements.indices[start:end]
        values = measurements.data[start:end,:,0]
        
        numerator[indices] += gain / noise2 * values
        denominator[indices] += np.square(gain) / noise2

    return numerator / denominator

def _compute_gains(
    measurements: scipy.sparse.bsr_array, 
    averages: np.ndarray, 
    sigma2: np.ndarray,
    m: int,
    p: int
) -> np.ndarray:
    power = np.mean(averages*averages, axis=0)

    correlations = np.empty((m, p))
    counts = np.empty((m, 1), dtype=np.int64)
    for i in range(m):
        start = measurements.indptr[i]
        end = measurements.indptr[i+1]
        indices = measurements.indices[start:end]
        y = measurements.data[start:end,:,0]
        x = averages[indices]
        
        correlations[i] = np.mean(x*y, axis=0)
        counts[i] = len(x)

    l = (np.sum(correlations, axis=0) - m*power) / np.sum(sigma2/counts, axis=0)
    return (correlations - l*sigma2/counts) / power

def _compute_sigma2(
    measurements: scipy.sparse.bsr_array, 
    averages: np.ndarray, 
    gains: np.ndarray,
    m: int,
    p: int
) -> np.ndarray:
    result = np.empty((m, p))

    for i in range(m):
        gain = gains[i]
        start = measurements.indptr[i]
        end = measurements.indptr[i+1]
        indices = measurements.indices[start:end]
        y = measurements.data[start:end,:,0]
        x = averages[indices]
        
        error = gain*x - y
        result[i] = np.mean(error*error, axis=0)

    return result

def _average_embedding_component(
    measurements = scipy.sparse.bsr_array,
    max_iter: int = 16
) -> np.ndarray:
    p = measurements.blocksize[0]
    m = measurements.shape[0] // p
    n = measurements.shape[1]
    
    gains = np.ones((m, p))
    sigma2 = np.ones((m, p))
    for _ in range(max_iter):
        averages = _compute_averages(measurements, gains, sigma2, m, p)
        gains = _compute_gains(measurements, averages, sigma2, m, p)
        sigma2 = _compute_sigma2(measurements, averages, gains, m, p)

    return  _compute_averages(measurements, gains, sigma2, m, p)

def _correct_embedding_orientations(
    embeddings: scipy.sparse.bsr_array, 
    transforms: np.ndarray
) -> scipy.sparse.bsr_array:
    n = embeddings.shape[1]
    m, _, k = transforms.shape
    data = np.empty((len(embeddings.data), k, 1))
    indices = embeddings.indices
    indptr = embeddings.indptr
    for i, transform in enumerate(transforms):
        start = indptr[i]
        end = indptr[i+1]
        
        np.matmul(
            transform.T,
            embeddings.data[start:end],
            out=data[start:end]
        )
    
    return scipy.sparse.bsr_array(
        (data, indices, indptr),
        shape=(m*k, n)
    )
    
def average_embeddings(
    embeddings: scipy.sparse.bsr_array, 
    transforms: np.ndarray
) -> np.ndarray:
    embeddings = _correct_embedding_orientations(embeddings, transforms)
    return _average_embedding_component(embeddings)
