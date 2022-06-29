import jax.numpy as jnp

from daxbench.core.engine.primitives.primitives import length


def _sdf_batch(size, f, grid_pos):
    # p: vec3,b: vec3
    size = size.reshape((3,))
    q = jnp.abs(grid_pos) - size
    q = jnp.clip(q, 0., jnp.inf)
    out = length(q)

    tmp = jnp.where(q[..., 1] > q[..., 2], q[..., 1], q[..., 2])
    tmp = jnp.where(q[..., 0] > tmp, q[..., 0], tmp)
    tmp = jnp.clip(tmp, -jnp.inf, 0.)
    out += tmp
    # out += min(max(q[0], max(q[1], q[2])), 0.0)
    return out
