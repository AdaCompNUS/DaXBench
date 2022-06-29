import glob
import os
import pickle
import time
from dataclasses import dataclass, field

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import open3d
import pyrender
import trimesh
from jax import random

from daxbench.core.engine.primitives.container import _sdf_batch as container_sdf, cut_hollow_sphere_np
from daxbench.core.engine.primitives.primitives import set_sdf
from daxbench.core.engine.pyrender.py_render import WaterPyRenderer
from daxbench.core.engine.usdrender.mix_usd import create_usd_mix_scene
from daxbench.core.envs.basic.mpm_env import MPMEnv

my_path = os.path.dirname(os.path.abspath(__file__))


def _d_gravity():
    return jnp.array([0, -9.8, 0])


@dataclass
class DefaultConf:
    seed = 1
    n_primitive = 2
    obs_type = MPMEnv.PARTICLE
    key = random.PRNGKey(0)

    """
    n_grid: resolution of the simulation.
    steps: number of internal steps.
    dt: internal step time.  higher rigidness requires lower dt.
    E: rigidness of object or resistance to deformation.
    """
    ground_friction: float = 0.1
    n_grid: int = 128
    dt: float = 4e-4
    primitive_action_steps = 1
    primitive_action_duration = 0.01  # seconds
    steps = int(primitive_action_duration / primitive_action_steps / dt)  # internal steps
    E: float = 100
    nu: float = 0.1

    # res = (n_grid // 2, n_grid // 3, n_grid // 2)
    res: ... = (n_grid, n_grid // 2, n_grid)

    dx, inv_dx = 1 / n_grid, float(n_grid)
    p_vol, p_rho = (dx * 0.5) ** 2, 1
    p_mass = p_vol * p_rho
    gravity: jnp.ndarray = field(default_factory=_d_gravity)  # gravity ...

    task = "pour_soup2"
    goal_path = f"{my_path}/goals/{task}/goal.npy"


PourSoupConfig = DefaultConf


class PourSoupEnv(MPMEnv):

    def __init__(self, batch_size, seed, max_steps=120, conf=None, aux_reward=False, **kwargs):
        conf = DefaultConf() if conf is None else conf
        self.conf = conf

        super().__init__(conf, batch_size, max_steps, seed, focus_computation=True)
        self.renderer = WaterPyRenderer()
        self.observation_size = 45861

    @staticmethod
    @jax.vmap
    def get_primitive_actions(actions, state):
        dummy_actions = jnp.zeros_like(actions)
        actions = jnp.concatenate([actions, dummy_actions])
        actions = actions[None, ...].repeat(1, 0)

        # normalize actions
        actions = actions.at[..., :3].set(actions[..., :3] / 500.0)
        actions = actions.at[..., 3:6].set(actions[..., 3:6] / 500.0)
        actions = actions + 1e-12
        actions = actions.at[..., 1].set(0)

        return actions, state

    @staticmethod
    def process_pre_step_actions(actions, shift):
        return actions

    @staticmethod
    @jax.vmap
    def auto_reset(state, state_new, key):
        init_pos = jnp.array([0.5, 0.2, 0.5])
        key, _ = random.split(key)
        init_pos = init_pos.at[..., [0, 2]].add(random.normal(key, (2,)) * 0.02)
        position = state.primitives[0].position.at[0].set(init_pos)
        state.primitives[0] = state.primitives[0]._replace(position=position)
        state = state._replace(key=key)
        return state

    def create_mesh_for_render(self, size):
        container = cut_hollow_sphere_np(size)
        container.save('container.ply')
        container = trimesh.load('container.ply')
        mesh = pyrender.Mesh.from_trimesh(container)
        return mesh

    def reset(self, key):
        # clean up env before reset, important!
        self.clean_up_b4_reset()

        # set sdf function for collision, also set type of end effector
        set_sdf(container_sdf)

        colors = []
        size_list = []

        # create water
        state = self.simulator.add_box(conf=self.conf, state=None, hardness=1,
                                       size=[0.07, 0.07, 0.07], init_pos=[0.5, 0.2, 0.5],
                                       z_rotation_angle=0, material=0, density=4)
        size_list.append(state.x.shape[0])

        # create tofu
        state = self.simulator.add_box(conf=self.conf, state=state, hardness=0.3,
                                       size=[0.03, 0.03, 0.03], init_pos=[0.47, 0.2, 0.5],
                                       z_rotation_angle=0, material=1, density=2)
        size_list.append(state.x.shape[0] - sum(size_list))

        state = self.simulator.add_box(conf=self.conf, state=state, hardness=0.3,
                                       size=[0.03, 0.03, 0.03], init_pos=[0.5, 0.2, 0.55],
                                       z_rotation_angle=0, material=1, density=2)
        size_list.append(state.x.shape[0] - sum(size_list))
        colors.extend([[255, 255, 255]] * state.x.shape[0])

        # load veg
        # veg_pc = jnp.array(np.load(f"{my_path}/../engine/pyrender/models/veg/model.npy"))
        # veg_pc = (veg_pc - veg_pc.mean(0)) / 500.0
        # veg_pc += jnp.array([0.55, 0.2, 0.5])
        # state = state._replace(x=jnp.concatenate((state.x, veg_pc), axis=0))
        #
        # size_list.append(state.x.shape[0]-size_list[-1])
        # veg = open3d.io.read_point_cloud(f"{my_path}/../engine/pyrender/models/veg/model.pcd")
        # veg = veg.voxel_down_sample(voxel_size=0.5)
        # veg_pc = np.asarray(veg.points)
        # self.veg_pc_color = np.asarray(veg.colors)
        veg = open3d.io.read_point_cloud(f"{my_path}/../engine/pyrender/models/veg/model.pcd")
        veg = veg.voxel_down_sample(voxel_size=0.5)
        veg_pc = np.asarray(veg.points)
        veg_pc_color = np.asarray(veg.colors)
        colors.extend(veg_pc_color.tolist())

        veg_pc = (veg_pc - veg_pc.mean(0)) / 400.0
        veg_pc += jnp.array([0.55, 0.2, 0.5])
        state = state._replace(x=jnp.concatenate((state.x, veg_pc), axis=0))
        size_list.append(state.x.shape[0] - sum(size_list))

        # set veg material and hardness
        material_ = jnp.full((len(veg_pc),), 1)
        h_ = jnp.full((len(veg_pc),), 0.3)
        self.simulator.material = jnp.concatenate((self.simulator.material, material_), axis=0)
        self.simulator.h = jnp.concatenate((self.simulator.h, h_), axis=0)

        # add two bowls
        box_size = np.array([[0.09, 0., 0.008], [0.08, 0., 0.008]])
        self.create_primitive(self.conf, state=state, friction=0.1, color=[0.5, 0.5, 0.5],
                              size=box_size[0], init_pos=[0.5, 0.2, 0.5])

        self.create_primitive(self.conf, state=state, friction=0.1, color=[0.5, 0.5, 0.5],
                              size=box_size[1], init_pos=[0.5, 0.06, 0.3])

        # initialize state after adding particles and primitives, important!
        self.initialize_after_adding_particle_primitives(state)

        self.state = self.auto_reset(self.init_state, self.init_state, self.init_state.key)
        self.size_list = size_list
        self.colors = colors
        return self.get_obs(self.state), self.state

    def get_exp_action(self):
        # create a dummy image with size (100,100)
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.imshow('control pad', img)
        # get key event from cv2 image
        k = cv2.waitKey(0) & 0xFF
        # print out the key pressed

        # print(k)

        unit = 1.0
        if k == 119:  # w
            return np.array([-unit, 0, 0, 0, 0, 0])
        elif k == 115:  # s
            return np.array([unit, 0, 0, 0, 0, 0])
        elif k == 97:  # a
            return np.array([0, 0, -unit, 0, 0, 0])
        elif k == 100:  # d
            return np.array([0, 0, unit, 0, 0, 0])
        elif k == 225:  # shift
            return np.array([0, 0, 0, unit * 30, 0, 0])
        elif k == 9:  # shift
            return np.array([0, 0, 0, -unit * 30, 0, 0])
        elif k == 13:  # enter
            return None

        return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    def collect_expert_demo(self, num_demo=10):
        assert self.batch_size == 1

        # get number of existing demo files
        num_existing_demo = len(glob.glob(f"{my_path}/expert_demo/{self.conf.task}/*.pkl"))
        i = num_existing_demo
        self.simulator.key_global, _ = random.split(self.simulator.key_global)
        obs, state = self.reset(env.simulator.key_global)

        while i < num_demo:
            demo = {"obs": [], "action": [], "state": []}
            valid_episode = True
            while True:
                self.render(state)
                actions = env.get_exp_action()
                if actions is None:
                    valid_episode = False
                    break

                actions = actions[None, ...]
                demo['state'].append(state)
                demo['action'].append(actions)
                demo['obs'].append(obs)

                if len(demo['obs']) == self.max_steps - 1 and i == 0:
                    # save goal
                    os.makedirs(f"{my_path}/goals/{self.conf.task}", exist_ok=True)
                    np.save(f"{my_path}/goals/{self.conf.task}/goal.npy", state.x[0])

                obs, reward, done, info = self.step_diff(actions, state)
                # obs, reward, _, info = self.step_with_render(actions, state)
                state = info['state']

                if done[0] == 1:
                    break

                print(state.cur_step, "reward", reward)

            if valid_episode:
                os.makedirs(f"{my_path}/expert_demo/{self.conf.task}", exist_ok=True)
                with open(f"{my_path}/expert_demo/{self.conf.task}/demo_{i}.pkl", "wb") as f:
                    pickle.dump(demo, f)

                i += 1

        exit(0)


def grad_test():
    env = PourSoupEnv(batch_size=1, seed=1)
    # env.reset = jax.vmap(env.reset)
    obs, state = env.reset(env.simulator.key_global)
    actions_up = jnp.array([0, 0, 0.0, 0, 0, 0])[None, ...]

    def loss_fn(actions, state):
        obs, reward, done, info = env.step_diff(actions, state)
        state = info["state"]

        return state.x.sum()

    loss_fn = jax.jit(jax.grad(loss_fn))
    grad_raw = loss_fn(actions_up, state)
    print(grad_raw)


if __name__ == "__main__":
    env = PourSoupEnv(batch_size=1, seed=1)
    # env.collect_expert_demo(num_demo=11)
    actions = jnp.zeros((env.batch_size, 6))
    obs, state = env.reset(env.simulator.key)

    print("time start")
    start_time = time.time()
    state_list = []
    for it in range(160):
        state = env.auto_reset(env.init_state, state, state.key)
        # obs, state = env.reset(env.simulator.key)
        for i in range(env.max_steps - 1):
            actions = env.get_exp_action().reshape((1, 6))
            # obs, reward, done, info = env.step_diff(actions, state)
            obs, reward, done, info = env.step_with_render(actions, state, True)
            state = info["state"]
            state_list.append(state)
            print("it", it, "step", i, reward, time.time() - start_time)
            if i == 100:
                break

        print(time.time() - start_time)
        colors = np.array(env.colors)
        size_list = np.array(env.size_list)
        create_usd_mix_scene(state_list, colors, size_list, "soup.usda")
        exit(0)
