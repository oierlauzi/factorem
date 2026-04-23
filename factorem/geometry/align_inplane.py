
import numpy as np

def compute_in_plane_alignment(
    reference_matrix_3d: np.ndarray,
    rotation_matrices_3d: np.ndarray,
    apply_streching: bool = False,
) -> np.ndarray:
    delta_matrices_3d = reference_matrix_3d @ rotation_matrices_3d.swapaxes(-2, -1)
    
    if apply_streching:
        inverse_matrices_3d = np.linalg.inv(delta_matrices_3d)
        return np.linalg.inv(inverse_matrices_3d[..., :2,:2])
    else:
        u, _, vh = np.linalg.svd(delta_matrices_3d[..., :2,:2])
        return u @ vh
    