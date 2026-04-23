from typing import Optional
import numpy as np


def euler_zyz_to_matrix(
    rot: np.ndarray,
    tilt: np.ndarray,
    psi: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    
    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype

    if out is None:
        out = np.empty(result_shape, dtype=dtype)
    elif out.shape != result_shape or out.dtype != dtype:
        raise RuntimeError('Invalid output array was provided')
    
    ai = rot
    aj = tilt
    ak = psi 



    

    # Obtain sin and cos of the half angles
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    
    # Obtain the combinations
    cc = cj * ci
    cs = cj * si
    sc = sj * ci
    ss = sj * si

    # Build the matrix
    out[...,0,0] = ck*cc - sk*si
    out[...,0,1] = ck*cs + sk*ci
    out[...,0,2] = -ck*sj
    out[...,1,0] = -sk*cc - ck*si
    out[...,1,1] = -sk*cs + ck*ci
    out[...,1,2] = sk*sj
    out[...,2,0] = sc
    out[...,2,1] = ss
    out[...,2,2] = cj

    return out
