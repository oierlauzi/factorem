from typing import Sequence
import numpy as np
import scipy.sparse

def _compute_averages(
    measurements: Sequence[scipy.sparse.csc_array], 
    gains: np.ndarray, 
    sigma2: np.ndarray
) -> np.ndarray:
    p = len(measurements)
    n, m = measurements[0].shape

    numerator = np.zeros((n, p))
    denominator = np.zeros((n, p))
    for k in range(p):
        plane = measurements[k]
        for j in range(m):
            gain = gains[j,k]
            sigma2 = sigma2[j,k]
            start = plane.indptr[j]
            end = plane.indptr[j+1]
            indices = plane.indices[start:end]
            values = plane.data[start:end]
            
            numerator[indices,k] += gain / sigma2 * values
            denominator[indices,k] += np.square(gain) / sigma2

    return numerator / denominator

def _compute_gains(
    measurements: Sequence[scipy.sparse.csc_array], 
    averages: np.ndarray, 
    sigma2: np.ndarray
) -> np.ndarray:
    p = len(measurements)
    n, m = measurements[0].shape
    
    result = np.empty((m, p))
    correlations = np.empty(m)
    counts = np.empty(m, dtype=np.int64)
    for k in range(p):
        plane = measurements[k]
        power = np.dot(averages[:,k], averages[:,k]) / n
        sigma2 = sigma2[:,k]
        for j in range(m):
            start = plane.indptr[j]
            end = plane.indptr[j+1]
            indices = plane.indices[start:end]
            y = plane.data[start:end]
            x = averages[indices,k]
            
            correlations[j] = np.dot(x, y) / len(x)
            counts[j] = len(x)

        l = (np.sum(correlations) - m*power) / np.sum(sigma2/counts)
        result[:,k] = (correlations - l*sigma2/counts) / power
        
    return result

def _compute_sigma2(
    measurements: Sequence[scipy.sparse.csc_array], 
    averages: np.ndarray, 
    gains: np.ndarray
) -> np.ndarray:
    p = len(measurements)
    _, m = measurements[0].shape
        
    result = np.empty((m, p))
    for k in range(p):
        plane = measurements[k]
        for j in range(m):
            gain = gains[j,k]
            start = plane.indptr[j]
            end = plane.indptr[j+1]
            indices = plane.indices[start:end]
            y = plane.data[start:end]
            x = averages[indices,k]
            
            error = gain*x - y
            result[j,k] = np.dot(error, error) / len(error)

    return result

def _average_embedding_component(
    measurements: Sequence[scipy.sparse.csc_array], 
    max_iter: int = 16
) -> np.ndarray:
    m = len(measurements)
    p = measurements[0].shape[1]
    
    # Alternating maximization
    gains = np.ones((m, p))
    noise2 = np.ones((m, p))
    for _ in range(max_iter):
        averages = _compute_averages(measurements, gains, noise2)
        gains = _compute_gains(measurements, averages, noise2)
        noise2 = _compute_sigma2(measurements, averages, gains)

    return  _compute_averages(measurements, gains, noise2)

def average_embeddings(
    embeddings: scipy.sparse.csc_array, 
    transforms: np.ndarray
) -> np.ndarray:
    n_directions, n_analysis_components, n_total_components = transforms.shape

    transformed_embeddings = []
    for i in range(n_directions):
        start = i*n_analysis_components
        end = start + n_analysis_components
        directional_embedding = embeddings[:,start:end]

         # (transforms.T @ directional_embedding.T).T
        transformed_embeddings.append(directional_embedding @ transforms[i])
    
    print(type(transformed_embeddings[0]))
    
    x = []    
    for i in range(n_total_components):
        columns = (m.getcol(i) for m in transformed_embeddings)
        print(transformed_embeddings[0].getcol(i))
        x.append(scipy.sparse.hstack(columns, format='csc'))

    return _average_embedding_component(x)
