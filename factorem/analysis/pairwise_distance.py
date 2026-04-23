import jax.numpy as jnp

def crossed_pairwise_distance2(
    left_images: jnp.ndarray,
    left_ctfs: jnp.ndarray,
    right_images: jnp.ndarray,
    right_ctfs: jnp.ndarray,
) -> jnp.ndarray:
    # Expand: 
    # |L_i*b_j - R_j*a_i|^2 = 
    # |L_i|^2*b_j^2 + |R_j|^2*a_i^2 - 2*a_i*b_j*Re(L_i*conj(R_j))
    n = left_images.shape[0]
    m = right_images.shape[0]
    L = left_images.reshape(n, -1)   # complex (n, p)
    R = right_images.reshape(m, -1)  # complex (m, p)
    a = left_ctfs.reshape(n, -1)     # real    (n, p)
    b = right_ctfs.reshape(m, -1)    # real    (m, p)

    L_abs2 = jnp.square(L.real) + jnp.square(L.imag)  # (n, p)
    R_abs2 = jnp.square(R.real) + jnp.square(R.imag)  # (m, p)

    term1 = L_abs2 @ (b**2).T
    term2 = (a**2) @ R_abs2.T
    term3 = (a * L.real) @ (b * R.real).T + (a * L.imag) @ (b * R.imag).T

    return term1 + term2 - 2*term3

def self_pairwise_distance2(
    images: jnp.ndarray,
    ctfs: jnp.ndarray
) -> jnp.ndarray:
    # Expand: 
    # |A_i*c_j - A_j*c_i|^2 = 
    # |A_i|^2*c_j^2 + |A_j|^2*c_i^2 - 2*c_i*c_j*Re(A_i*conj(A_j))
    # term2 == term1.T by symmetry, so only 3 matmuls needed instead of 4.
    n = images.shape[0]
    A = images.reshape(n, -1)   # complex (n, p)
    c = ctfs.reshape(n, -1)     # real    (n, p)

    A_abs2 = jnp.square(A.real) + jnp.square(A.imag) # (n, p)

    term1 = A_abs2 @ (c**2).T # (n, n)
    term2 = term1.T
    term3 = (c * A.real) @ (c * A.real).T + (c * A.imag) @ (c * A.imag).T # (n, n)

    return term1 + term2 - 2*term3
