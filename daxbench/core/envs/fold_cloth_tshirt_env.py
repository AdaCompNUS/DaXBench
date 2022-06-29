import os
import time
from dataclasses import dataclass

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.engine.usdrender.mesh_usd import create_usd_cloth_scene
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.utils.util import get_expert_start_end_cloth

my_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DefaultConf:
    N = 180
    cell_size = 1.0 / N
    gravity = 0.5
    stiffness = 5000
    damping = 2
    dt = 0.5e-3
    max_v = 2.
    small_num = 1e-8
    mu = 0.9  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 2
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task = "fold_tshirt"
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = True


FoldTshirtConfig = DefaultConf


class FoldTshirtEnv(ClothEnv):

    def __init__(self, batch_size, conf=None, aux_reward=False, seed=1):
        conf = DefaultConf() if conf is None else conf
        max_steps = 5
        super().__init__(conf, batch_size, max_steps, aux_reward)
        self.observation_size = 1082

    def create_cloth_mask(self, conf):
        img = cv2.imread(f"{my_path}/others/t-shirt.jpg")

        size = conf.N // 2
        h_size = size // 2

        img = cv2.resize(img, (size, size))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("img", img)
        # cv2.waitKey(20)

        mask = (img.sum(-1) < 100).astype(np.int32)
        # cv2.imshow("mask", mask.astype(np.float32))
        # cv2.waitKey(20)

        cloth_mask = jnp.zeros((conf.N, conf.N))
        cloth_mask = cloth_mask.at[conf.N // 2 - h_size:conf.N // 2 + h_size,
                     conf.N // 2 - h_size:conf.N // 2 + h_size].set(mask)

        return cloth_mask

    @staticmethod
    @vmap
    @jax.jit
    def get_obs(state: ClothState, obs_type=ClothEnv.PARTICLE):

        if obs_type == ClothEnv.DEPTH:
            pixel_size = 0.003125
            bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
            points = state.x + jnp.array([[0, 0.01, 0]])
            width = 320
            height = 640
            iz = jnp.argsort(points[:, 1])
            points = points[iz]
            px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
            py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
            px = jnp.clip(px, 0, width - 1)
            py = jnp.clip(py, 0, height - 1)

            heightmap = jnp.zeros((320, 640), dtype=jnp.float32)
            heightmap.at[py, px].set(points[:, 1] - bounds[2, 0])
            heightmap = jnp.expand_dims(heightmap, axis=-1)
            obs = heightmap

        elif obs_type == ClothEnv.PARTICLE:

            # sample x (N,3) every 10 points
            x = state.x[::10, :]
            obs = jnp.concatenate(
                [
                    x.flatten(),
                    # v.flatten(),
                    state.primitive0,
                    state.primitive1,
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        return obs


if __name__ == "__main__":
    env = FoldTshirtEnv(batch_size=1, seed=1)
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
