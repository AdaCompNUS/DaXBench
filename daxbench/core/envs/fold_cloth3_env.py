import os
import time
from dataclasses import dataclass

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.engine.usdrender.mesh_usd import create_usd_cloth_scene
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
    mu = 0.5  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 2
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task = "fold_cloth3"
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = True


FoldCloth3Config = DefaultConf


class FoldCloth3Env(ClothEnv):

    def __init__(self, batch_size, conf=None, aux_reward=False, seed=1):
        conf = DefaultConf() if conf is None else conf
        max_steps = 4
        super().__init__(conf, batch_size, max_steps, aux_reward)
        self.observation_size = 1544

    def create_cloth_mask(self, conf):
        N, size = conf.N, conf.size
        cloth_mask = jnp.zeros((N, N))
        cloth_mask = cloth_mask.at[size * 2:size * 3, size * 2:size * 4].set(1)

        return cloth_mask


if __name__ == "__main__":
    env = FoldCloth3Env(batch_size=1)
    # env.collect_goal()
    # env.collect_expert_demo(10)

    obs, state = env.reset(env.simulator.key_global)
    actions = np.zeros((env.batch_size, 6))
    env.step_diff(actions, state)  # to compile the jax module

    # interactive test
    vertices = []
    print("time start")
    start_time = time.time()
    for i in range(3):
        actions = get_expert_start_end_cloth(env.get_x_grid(state), env.cloth_mask)
        # actions = env.get_random_fold_action(state)
        # obs, reward, done, info = env.step_diff(actions, state)
        obs, reward, done, info = env.step_with_render(actions, state)
        state = info['state']

        rgb, depth = env.render(state)
        cv2.imwrite(f"{env.conf.task}_{i}.jpg", rgb[:, :, ::-1])

        for j in range(info['state_list'].x.shape[0]):
            state_ = jax.tree_util.tree_map(lambda x: x[j], info['state_list'])
            vertice = env.get_x_grid(state_)[0].reshape((-1, 3))
            vertices.append(vertice)

    vertices = np.array(vertices)
    indices = np.array(env.simulator.indices)
    create_usd_cloth_scene(vertices, indices, "fold_cloth3.usda")
    print(time.time() - start_time)
