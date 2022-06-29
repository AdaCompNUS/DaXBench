import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.utils.util import get_expert_start_end_cloth

my_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DefaultConf:
    N = 80
    cell_size = 1.0 / N
    gravity = 0.5
    stiffness = 900
    damping = 2
    dt = 2e-3
    max_v = 2.
    small_num = 1e-8
    mu = 3  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 2
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task = "unfold_cloth3"
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = False


UnfoldCloth3Config = DefaultConf


class UnfoldCloth3Env(ClothEnv):

    def __init__(self, batch_size, conf=None, aux_reward=False, seed=1):
        conf = DefaultConf() if conf is None else conf
        max_steps = 15
        super().__init__(conf, batch_size, max_steps, aux_reward)
        self.observation_size = 1544
        self.reset = self.build_reset()

    def create_cloth_mask(self, conf):
        N, size = conf.N, conf.size
        cloth_mask = jnp.zeros((N, N))
        cloth_mask = cloth_mask.at[size * 2:size * 3, size * 2:size * 4].set(1)

        return cloth_mask

    def random_fold(self, state, key, step=10):
        num_particle = state.x.shape[1]
        batch_idx = jnp.arange(state.x.shape[0])
        for i in range(step):
            st_point = np.random.randint(
                0, num_particle, size=(state.x.shape[0],))
            ed_point = np.random.randint(
                0, num_particle, size=(state.x.shape[0],))

            actions = jnp.concatenate(
                (state.x[batch_idx, st_point], state.x[batch_idx, ed_point]), axis=-1)
            _, _, _, info = self.step_diff(actions, state)
            state = info['state']
        return state

    def build_reset(self):
        init_state = self.simulator.reset_jax()

        def reset(key):
            key, _ = random.split(key)
            new_x = init_state.x + \
                random.normal(key, init_state.x.shape) * 0.0001
            state = init_state._replace(x=new_x)
            state = self.random_fold(state, key, step=3)
            return ClothEnv.get_obs(state), state

        return reset


if __name__ == "__main__":
    env = UnfoldCloth3Env(batch_size=1, seed=1)
    # env.collect_expert_demo(10)

    key = jax.random.PRNGKey(1)
    obs, state = env.reset(key)

    actions = np.zeros((env.batch_size, 6))
    env.step_diff(actions, state)  # to compile the jax module

    # interactive test
    for i in range(100):
        actions = get_expert_start_end_cloth(
            env.get_x_grid(state), env.cloth_mask)
        # obs, reward, done, info = env.step_diff(actions, state)
        obs, reward, done, info = env.step_with_render(actions, state)
        state = info['state']
        print(i, "reward", reward)
