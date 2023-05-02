import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from daxbench.core.engine.mpm_simulator import MPMState
from daxbench.core.engine.primitives.box import _sdf_batch as box_sdf
from daxbench.core.engine.primitives.primitives import set_sdf
from daxbench.core.engine.pyrender.py_render import ParticlePyRenderer
from daxbench.core.engine.usdrender.water_usd import create_usd_liquid_scene
from daxbench.core.envs.basic.mpm_env import MPMEnv
from daxbench.core.envs.others.metric import calc_IOU
from daxbench.core.utils.util import get_expert_start_end_mpm

my_path = os.path.dirname(os.path.abspath(__file__))


class DefaultConf:
    seed = 1
    n_primitive = 1
    obs_type = MPMEnv.PARTICLE
    key = random.PRNGKey(0)

    """
    n_grid: resolution of the simulation.
    steps: number of internal steps.
    dt: internal step time.  higher rigidness requires lower dt.
    E: rigidness of object or resistance to deformation.
    h: the speed to recover its original shape after deformation. (used when create the object)
    """
    ground_friction = 2
    n_grid, steps, dt = 96, 16, 2e-4
    E, nu = 2, 0.2
    res = (n_grid // 2, n_grid//3, n_grid // 2)

    dx, inv_dx = 1 / n_grid, float(n_grid)
    p_vol, p_rho = (dx * 0.5) ** 2, 1
    p_mass = p_vol * p_rho
    gravity = jnp.array([0, -9.8, 0])  # gravity ...
    sdf_func = box_sdf

    task = "shape_elasto_plastic"
    goal_path = f"{my_path}/goals/{task}/goal.npy"

    ## Rope property
    rope_width = [0.2, 0.06, 0.12]
    rope_init_pos = [0.5, 0.07, 0.5]
    rope_z_rotation_angle = 0
    rope_hardness = 1.0  # 0.1 - 5. How fast the shape recovers.


class ShapeRopeEnv(MPMEnv):

    def __init__(self, batch_size, seed, max_steps=6, conf=None, aux_reward=False):
        conf = DefaultConf() if conf is None else conf
        self.conf = conf
        self.focus_computation = True

        super().__init__(conf, batch_size, max_steps, seed, self.focus_computation)
        self.aux_reward = aux_reward
        self.renderer = ParticlePyRenderer()
        self.observation_size = 3540

    @staticmethod
    @jax.vmap
    def auto_reset(state, state_new, key):
        # TODO complete auto reset
        return state_new

    @staticmethod
    def process_pre_step_actions(actions, shift):
        actions = actions.at[..., 0:3].add(shift)
        actions = actions.at[..., 3:].add(shift)
        return actions

    @staticmethod
    @jax.vmap
    def get_primitive_actions(actions, state: MPMState):
        start, end = actions[:3], actions[3:]
        start = start.at[1].set(0.01)
        end = end.at[1].set(0.01)
        norm = jnp.linalg.norm(end - start) + 1e-8

        # set max move length
        vec = (end - start) / norm
        scale = norm.clip(0.0, 0.1)
        end = start + vec * scale

        p_state = state.primitives[0]
        position = p_state.position.at[0].set(start)
        state.primitives[0] = state.primitives[0]._replace(position=position)

        act_push = end - start
        act_push = act_push[None, ...].repeat(20, axis=0) / 20
        act_push = act_push.at[..., 1].set(0)
        sub_actions = act_push

        sub_actions_rotation = jnp.zeros_like(sub_actions)
        sub_actions = jnp.concatenate([sub_actions, sub_actions_rotation], axis=-1)
        return sub_actions, state

    def random_push(self, step=10):
        for i in range(step):
            actions = self.random_policy(self.batch_size)
            actions[:, 1] = 0
            # state, reward, done, info = env.step_with_render(actions, self.state)
            state, reward, done, info = self.step_diff(actions, self.state)
            self.state = info["state"]
        # self.state = jax.lax.stop_gradient(self.state)

    def random_policy(self, n_actions, radius=0.05):
        pc = np.array(self.state.x[0])
        n_particles = pc.shape[0]
        p_ids = np.random.randint(0, n_particles, n_actions)
        end_list = pc[p_ids]
        angles = np.random.random((n_actions,)) * np.pi * 2
        end_list[:, 0] += np.cos(angles) * radius
        end_list[:, 2] += np.sin(angles) * radius
        start_list = pc[p_ids]
        start_list[:, 0] -= np.cos(angles) * radius
        start_list[:, 2] -= np.sin(angles) * radius

        act_list = []
        for i in range(n_actions):
            start_pos = start_list[i]
            end_pos = end_list[i]
            act_list.append([*start_pos, *end_pos])

        act_list = np.array(act_list)
        return act_list

    def reset(self, key):
        # clean up env before reset, important!
        self.clean_up_b4_reset()

        # set sdf function for collision, also set type of end effector
        set_sdf(box_sdf)

        # create rope
        state = self.simulator.add_box(conf=self.conf, state=None, hardness=self.conf.rope_hardness,
                                       size=self.conf.rope_width, init_pos=self.conf.rope_init_pos,
                                       z_rotation_angle=self.conf.rope_z_rotation_angle, material=2, density=3)

        # add primitive end effector to the state
        state = self.create_primitive(conf=self.conf, state=state, friction=0.1, color=[0.5, 0.5, 0.5],
                                      size=[0.015, 0.06, 0.015], init_pos=[0.5, 0.01, 0.45])

        # initialize state after adding particles and primitives, important!
        self.initialize_after_adding_particle_primitives(state)

        # self.random_push(step=2)
        return self.get_obs(self.state), self.state

    def collect_goal(self):
        assert self.batch_size == 1
        while True:
            self.simulator.key_global, _ = random.split(self.simulator.key_global)
            obs, state = self.reset(self.simulator.key_global)
            valid_episode = True
            while True:
                self.render(state)
                actions = get_expert_start_end_mpm(state.x, size=512)

                # click on the same place to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) < 1e-3:
                    break

                # click on two far away points to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) > 0.8:
                    valid_episode = False
                    break

                # obs, reward, _, info = env.step_diff(actions, state)
                obs, reward, _, info = self.step_with_render(actions, state)
                state = info['state']
                print("reward", reward)

            if valid_episode:
                os.makedirs(f"{my_path}/goals/{self.conf.task}", exist_ok=True)
                np.save(f"{my_path}/goals/{self.conf.task}/goal.npy", state.x[0])
                exit(0)

    def collect_expert_demo(self, num_demo=10):
        assert self.batch_size == 1

        # visualize goal
        size = 512
        x = np.load(f"{my_path}/goals/{self.conf.task}/goal.npy")
        x = (x[:, [0, 2]] * size).astype(np.int32)
        x = np.array(x)
        idx_x, idx_y = x[:, 0], x[:, 1]

        goal_map = np.zeros((size, size), dtype=np.float32)
        goal_map[idx_y, idx_x] = 1.

        # get number of existing demo files
        num_existing_demo = len(glob.glob(f"{my_path}/expert_demo/{self.conf.task}/*.pkl"))
        i = num_existing_demo
        while i < num_demo:
            self.simulator.key_global, _ = random.split(self.simulator.key_global)
            obs, state = self.reset(self.simulator.key_global)
            demo = {"obs": [], "action": [], "state": []}
            valid_episode = True
            while True:
                self.render(state)
                actions = get_expert_start_end_mpm(state.x, size=512, goal_map=goal_map)

                # click on the same place to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) < 1e-3:
                    break

                # click on two far away points to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) > 0.8:
                    valid_episode = False
                    break

                demo['state'].append(state)
                demo['action'].append(actions)
                demo['obs'].append(obs)

                obs, reward, _, info = self.step_diff(actions, state)
                # obs, reward, _, info = self.step_with_render(actions, state)
                state = info['state']
                print(state.cur_step, "reward", reward)
                print("iou", calc_IOU(state.x[0], env.conf.goal_path))

            if valid_episode:
                os.makedirs(f"{my_path}/expert_demo/{self.conf.task}", exist_ok=True)
                with open(f"{my_path}/expert_demo/{self.conf.task}/demo_{i}.pkl", "wb") as f:
                    pickle.dump(demo, f)
                i += 1

        exit(0)


if __name__ == "__main__":
    print(jax.devices())
    env = ShapeRopeEnv(batch_size=1, seed=1)
    # env.collect_goal()
    # env.collect_expert_demo(10)
    # actions = jnp.zeros((env.batch_size, 6))
    # obs, state = env.reset(env.simulator.key)
    print("time start")
    start_time = time.time()
    for it in range(100):
        # state = env.auto_reset(env.init_state, state, state.key)
        obs, state = env.reset(env.simulator.key)
        for i in range(20):
            actions = get_expert_start_end_mpm(state.x, size=512)
            # obs, reward, done, info = env.step_diff(actions, state)
            obs, reward, done, info = env.step_with_render(actions, state)
            state = info["state"]
            print("it", it, "step", i, time.time() - start_time)
            # print("iou", calc_IOU(state.x[0], env.conf.goal_path))
            # print("reward", reward)
            if i == 5:
                create_usd_liquid_scene(np.array(state.x[0]), "elasto.usda")
                exit(0)
        print(time.time() - start_time)
