import jax.numpy as jnp
import numpy as np

from daxbench.core.engine.primitives.primitives import length
from sdf import sdf3


def _sdf_batch(size, f, p):
    r, h, t = size
    w = jnp.sqrt(r * r - h * h)

    q = jnp.concatenate((length(p[..., [0, 2]])[..., None], p[..., 1][..., None]), axis=-1)
    mask = h * q[..., 0] < w * q[..., 1]
    val1 = length(q - jnp.array([w, h])) - t
    val2 = jnp.abs(length(q) - r) - t
    return jnp.where(mask, val1, val2)


@sdf3
def cut_hollow_sphere_np(size):
    r, h, t = size
    w = np.sqrt(r * r - h * h)

    def f(p):
        # sampling independent computations (only depend on shape)

        # sampling dependant computations
        # q = vec2(length(p.xz), p.y)
        # return ((h * q.x < w * q.y) ? length(q-vec2(w, h)): abs(length(q) - r) ) - t
        p = p[:, [0, 2, 1]]

        q = np.hstack((np.linalg.norm(p[:, [0, 2]], axis=1)[:, None], p[:, 1][:, None]))
        mask = h * q[:, 0] < w * q[:, 1]
        val1 = np.linalg.norm(q - np.array([[w, h]]), axis=1) - t
        val2 = np.abs(np.linalg.norm(q, axis=1) - r) - t
        return np.where(mask, val1, val2)

    return f
