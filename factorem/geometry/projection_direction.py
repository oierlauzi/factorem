from typing import List
import numpy as np
import math

def estimate_projection_direction_count(spacing: float):
    spacing2 = spacing*spacing
    return round(2*math.pi / spacing2)

def sample_projection_directions(n: int) -> np.ndarray:
    out = np.empty((n, 2))
    
    K = math.pi*(3 - math.sqrt(5))
    i = np.arange(n)
    out[:,0] = K * i
    
    z = np.linspace(0.0, 1.0, n)
    np.arccos(z, out=out[:,1])

    return out

def spherical_to_cartesian(
    theta: np.ndarray,
    phi: np.ndarray
) -> np.ndarray:
    batch_shape = np.broadcast_shapes(theta.shape, phi.shape)
    dtype = np.promote_types(theta.dtype, phi.dtype)
    out = np.empty(batch_shape + (3, ), dtype=dtype)
    
    np.cos(phi, out=out[...,0])
    np.sin(phi, out=out[...,1])
    np.cos(theta, out=out[...,2])
    out[...,:2] *= np.sin(theta[:,None])

    return out

def group_projection_directions(
    directions: np.ndarray,
    references: np.ndarray,
    max_distance_rad: float,
    consider_mirrors: bool = True,
    batch_size: int = 1024
) -> List[np.ndarray]:
    result = []
    
    cos_max_distance = math.cos(max_distance_rad)
    
    for reference in references:
        indices = []
        
        start = 0
        while start < len(directions):
            end = min(len(directions), start + batch_size)    
            direction_batch = directions[start:end]
            
            cos = direction_batch @ reference
            if consider_mirrors:
                cos = abs(cos)

            indices.append(np.argwhere(cos >= cos_max_distance)[:,0] + start)
            start = end

        result.append(np.concat(indices))
    
    return result
