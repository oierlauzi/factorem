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

def average_embeddings(
    embeddings: scipy.sparse.bsr_array,
    max_iter: int = 16
) -> np.ndarray:
    p = embeddings.blocksize[0]
    m = embeddings.shape[0] // p
    n = embeddings.shape[1]
    
    gains = np.ones((m, p))
    sigma2 = np.ones((m, p))
    for _ in range(max_iter):
        averages = _compute_averages(embeddings, gains, sigma2, m, p)
        gains = _compute_gains(embeddings, averages, sigma2, m, p)
        sigma2 = _compute_sigma2(embeddings, averages, gains, m, p)

    return  _compute_averages(embeddings, gains, sigma2, m, p)
