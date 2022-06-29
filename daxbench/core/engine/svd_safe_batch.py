"""
SVD with backwards formula for complex matrices and safe inverse,
"safe" in light of degenerate or vanishing singular values
with various convenience wrappers
"""

from functools import partial
from jax import custom_vjp, numpy as np

DEFAULT_EPS = 1e-12
DEFAULT_CUTOFF = 0.  # TODO changed this fpr custom jvp tracing.

# abbreviations for readability
T = np.transpose
Cc = np.conjugate
Hc = lambda arr: np.conj(arr.transpose((0, 2, 1)))


@partial(custom_vjp, nondiff_argnums=(1,))
def svd(A, epsilon=DEFAULT_EPS):
    """
    SVD with VJP (backwards mode auto-diff) formula for complex matrices and safe inverse
    (for stability when eigenvalues are degenerate or zero)
    Computes `U`, `S`, `Vh` such that
    1)
        ```
        A
        == (U * S) @ Vh
        == U @ np.diag(S) @ Vh
        == U @ (S[:, None] * Vh)
        ```
    2) S is real, non-negative
    3) U and Vh are isometric (`Hc(U) @ U == eye(k)` and `Vh @ Hc(Vh) == eye(k)`)
    Parameters
    ----------
    A : jax.numpy.ndarray
        The matrix to perform the SVD on. Shape (m,n)
    epsilon : float
        The control parameter for safe inverse. 1/x is replaced by x/(x**2 + epsilon)
        Should be very small.
    Returns
    -------
    U : jax.numpy.ndarray
        3D array. shape (b,m,k) where k = min(m,n) and (b,m,n) = A.shape
    S : jax.numpy.ndarray
        2D array of real, non-negative singular values. shape (k,)  where k = min(m,n) and (m,n) = A.shape
    Vh : jax.numpy.ndarray
        3D array. shape (b,k,n) where k = min(m,n) and (b,m,n) = A.shape
    """
    assert epsilon > 0
    return np.linalg.svd(A, full_matrices=False)


def _safe_inverse(x, eps):
    return x / (x ** 2 + eps)


def _svd_fwd(A, epsilon):
    assert epsilon > 0
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    res = (U, S, Vh, A)
    return (U, S, Vh), res


def _svd_bwd(epsilon, res, g):
    # FIXME double check

    assert epsilon > 0
    dU, dS, dVh = g
    U, S, Vh, A = res

    # avoid re-computation of the following in multiple steps
    Uc = Cc(U)
    Ut = U.transpose((0, 2, 1))
    Vt = Cc(Vh)
    Vt_dV = Vt @ Hc(dVh)
    S_squared = S ** 2  # 3
    S_inv = _safe_inverse(S, epsilon)  # 3

    # matrices in the AD formula
    I = np.eye(S.shape[-1])[None, :].repeat(S.shape[0], 0)
    F = _safe_inverse(S_squared[:, None, :] - S_squared[..., None], epsilon)
    F = F - I * F  # zeroes on the diagonal
    J = F * (Ut @ dU)
    K = F * Vt_dV
    L = I * Vt_dV

    # cc of projectors onto orthogonal complement of U (V)
    Pc_U_perp = I - Uc @ Ut
    Pc_V_perp = I - Vh.transpose((0, 2, 1)) @ Vt

    # AD formula
    S, dS, S_inv = S[:, None, :], dS[:, None, :], S_inv[:, None, :]

    dA = (Uc * dS) @ Vt \
         + Uc @ ((J + Hc(J)) * S) @ Vt \
         + (Uc * S) @ (K + Hc(K)) @ Vt \
         + .5 * ((Uc * S_inv) @ (L - Hc(L)) @ Vt) \
         + Pc_U_perp @ (dU * S_inv) @ Vt \
         + (Uc * S_inv) @ dVh @ Pc_V_perp

    return dA,


svd.defvjp(_svd_fwd, _svd_bwd)
