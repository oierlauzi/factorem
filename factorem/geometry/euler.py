from typing import Optional
import numpy as np


def euler_zyz_to_matrix(
    rot: np.ndarray,
    tilt: np.ndarray,
    psi: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray.Tensor:
    
    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype

    if out is None:
        out = np.empty(result_shape, dtype=dtype)
    elif out.shape != result_shape or out.dtype != dtype:
        raise RuntimeError('Invalid output array was provided')
    
    ai = -rot
    aj = -tilt
    ak = -psi 

    # Obtain sin and cos of the half angles
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    
    # Obtain the combinations
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    # Build the matrix
    out[...,0,0] = cj * cc - ss
    out[...,0,1] = cj * sc + cs
    out[...,0,2] = -sj * ck
    out[...,1,0] = -cj * cs - sc
    out[...,1,1] = -cj * ss + cc
    out[...,1,2] = sj * sk
    out[...,2,0] = sj * ci
    out[...,2,1] = sj * si
    out[...,2,2] = cj
    
    return out
