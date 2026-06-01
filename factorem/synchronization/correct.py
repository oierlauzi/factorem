import numpy as np
import scipy.sparse

def correct_embeddings(
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
    