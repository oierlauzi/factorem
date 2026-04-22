
from typing import Optional

import numpy as np

def align_inplane(
    direction_matrices_3d: np.ndarray,
    image_matrices_3d: np.ndarray,
    apply_streching: bool = False,
) -> np.ndarray:
    delta_matrices_3d = image_matrices_3d @ direction_matrices_3d.T
    
    if apply_streching:
        inverse_matrices_3d = np.linalg.inv(delta_matrices_3d)
        return np.linalg.inv(inverse_matrices_3d[..., :2,:2])
    else:
        u, _, vh = np.linalg.svd(delta_matrices_3d[..., :2,:2])
        return u @ vh
    