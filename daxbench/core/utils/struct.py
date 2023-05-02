from typing import NamedTuple

import jax.numpy as jnp


class StateStuct(NamedTuple):
    state: NamedTuple
    obs: jnp.array
    reward: jnp.array
    done: int
    info: dict
