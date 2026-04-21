import numpy as np

def _apply_shifts(
    rotation: np.ndarray,
    shifts: np.ndarray,
    origin: np.ndarray,
    shift_first: bool,
    out: np.ndarray
):
    if shift_first:
        pre_shift = shifts - origin
        post_shift = origin
    else:
        pre_shift = -origin
        post_shift = shifts + origin
    
    # Apply the shifts
    np.matmul(rotation, pre_shift[...,None], out=out[...,None])
    out += post_shift

def make_affine(
    rotation: np.ndarray, 
    shifts: np.ndarray,
    origin: np.ndarray,
    shift_first: bool = True,
    include_last_row: bool = True
) -> np.ndarray:
    batch_shape = np.broadcast_shapes(rotation.shape[:-2], shifts.shape[:-1])
    dtype = np.promote_types(rotation.dtype, shifts.dtype)
    n_dim = shifts.shape[-1]

    if rotation.shape[-2:] != (n_dim, n_dim):
        raise RuntimeError('Shift and rotation dimensions do not match')
    
    if include_last_row:
        matrix_shape = (n_dim+1, n_dim+1)
    else:
        matrix_shape = (n_dim, n_dim+1)

    result_shape = batch_shape + matrix_shape
    result = np.empty(shape=result_shape, dtype=dtype)

    result[...:n_dim,:n_dim] = rotation
    _apply_shifts(
        rotation=rotation, 
        shifts=shifts, 
        origin=origin, 
        shift_first=shift_first,
        out=result[...,:n_dim,n_dim]
    )

    if include_last_row:
        result[...,n_dim,:n_dim] = 0
        result[...,n_dim,n_dim] = 1
    
    return result
