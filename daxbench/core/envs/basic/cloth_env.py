import glob
import math
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Box
from jax import random, vmap

from daxbench.core.engine.cloth_simulator import ClothSimulator, ClothState
from daxbench.core.engine.pyrender.py_render import MeshPyRenderer
from daxbench.core.utils.util import calc_chamfer, get_expert_start_end_cloth, get_projection

my_path = os.path.dirname(os.path.abspath(__file__))


class ClothEnv:
    PARTICLE = "PARTICLE"
    DEPTH = "DEPTH"
    RGB = "RGB"

    def __init__(self, conf, batch_size, max_steps, aux_reward=False):

        assert conf
        cloth_mask = self.create_cloth_mask(conf)
        collision_func = self.get_collision_func()
        simulator = ClothSimulator(conf, batch_size, collision_func, cloth_mask)

        self.conf = conf
        self.aux_reward = aux_reward
        self.simulator = simulator
        self.cloth_mask = simulator.cloth_mask
        self.max_steps = max_steps
        self.batch_size = simulator.batch_size
        self.cur_step = 0
        self.action_size = 6
        self.seed(conf.seed)

        assert conf.goal_path
        self.goal_path = conf.goal_path

        num_p = int(self.cloth_mask.astype(jnp.int32).sum())
        self.observation_size = num_p * 6 + 8
        self.cloth_state_shape = (num_p, 6)
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(num_p * 6 + 8,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.spec = None

        self.idx_i, self.idx_j = jnp.nonzero(self.cloth_mask)
        self.renderer = MeshPyRenderer()
        self.step_diff = self.build_step_diff()
        self.step_diff = jax.jit(self.step_diff)
        self.reset = self.build_reset()

        if not os.path.exists(conf.goal_path):
            print("**************** Warning: goal file does not exist!")
            self.goal = jnp.zeros((1, 3))
        else:
            goal_map = np.load(conf.goal_path)
            self.goal = jnp.array(goal_map)

    def seed(self, seed):
        self.simulator.key_global = random.PRNGKey(seed)
        np.random.seed(seed)

    def state_to_depth(self, state):
        pixel_size = 0.003125
        z_offset = 0.01
        width = 320
        height = 320
        bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
        points = state.x + jnp.array([[0, z_offset, 0]])
        points = points[0]

        iz = jnp.argsort(points[..., 1])
        points = points[iz]
        px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
        py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
        px = jnp.clip(px, 0, width - 1)
        py = jnp.clip(py, 0, height - 1)

        heightmap = jnp.zeros((width, height), dtype=jnp.float32)
        heightmap = heightmap.at[py, px].set(points[:, 1])
        heightmap = jnp.expand_dims(heightmap, axis=-1)
        heightmap = np.array(heightmap)

        return np.array(heightmap)

    @staticmethod
    @vmap
    @jax.jit
    def get_obs(state: ClothState, obs_type=PARTICLE):

        if obs_type == ClothEnv.DEPTH:
            pixel_size = 0.003125
            z_offset = 0.01
            bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
            points = state.x + jnp.array([[0, z_offset, 0]])
            points = points[0]
            width = 320
            height = 320
            iz = jnp.argsort(points[..., 1])
            points = points[iz]
            px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
            py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
            px = jnp.clip(px, 0, width - 1)
            py = jnp.clip(py, 0, height - 1)

            heightmap = jnp.zeros((width, height), dtype=jnp.float32)
            heightmap = heightmap.at[py, px].set(points[:, 1])
            heightmap = jnp.expand_dims(heightmap, axis=-1)
            obs = heightmap

        elif obs_type == ClothEnv.PARTICLE:
            obs = jnp.concatenate(
                [
                    state.x.flatten(),
                    # state.v.flatten(),
                    state.primitive0,
                    state.primitive1,
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        return obs

    @staticmethod
    @partial(vmap, in_axes=(0, 0), out_axes=1)
    def get_pnp_actions(actions, state: ClothState):
        pick, place = actions[:3], actions[3:]

        # particle_num = state.x.shape[-2]
        # pickup_place = pick[None, :].repeat(particle_num, -2)
        # nearest_idx = jnp.sqrt(jnp.sum((pickup_place - state.x) ** 2, -1)).argmin(-1)
        # shift = state.x[nearest_idx] - pick
        # pick += shift

        pick = pick.at[1].set(0)
        place = place.at[1].set(0)

        act_down = pick - state.primitive0[:3]
        act_down = jnp.ones(4).at[:3].set(act_down)
        act_down = act_down[None, ...].repeat(3, axis=0)
        act_down = act_down.at[..., :3].set(act_down[..., :3] / 3)
        sub_actions = act_down

        act_up = jnp.array([0, 0.06, 0, 0])
        act_up = act_up[None, ...].repeat(10, axis=0)
        act_up = act_up.at[..., :3].set(act_up[..., :3] / 10)
        sub_actions = jnp.concatenate([sub_actions, act_up], axis=0)

        act_move = place - pick
        act_move = act_move.at[1].set(0)
        act_move = jnp.zeros(4).at[:3].set(act_move)
        act_move = act_move[None, ...].repeat(20, axis=0)
        act_move = act_move.at[..., :3].set(act_move[..., :3] / 20)
        sub_actions = jnp.concatenate([sub_actions, act_move], axis=0)

        act_release = jnp.array([0, 0, 0, 1])
        act_release = act_release[None, ...].repeat(7, axis=0)
        sub_actions = jnp.concatenate([sub_actions, act_release], axis=0)

        dummy_actions = jnp.zeros_like(sub_actions)
        sub_actions = jnp.concatenate([sub_actions, dummy_actions], axis=1)

        return sub_actions

    def get_x_grid(self, state):
        return self.simulator.get_x_grid(state.x)

    def build_reset(self):
        init_state = self.simulator.reset_jax()

        def reset(key):
            key, _ = random.split(key)
            new_x = init_state.x.at[..., [0, 2]].add(random.normal(key, (2,)) * 0.05)
            state = init_state._replace(x=new_x)
            return self.get_obs(state), state

        return reset

    def step_with_render(self, actions, state: ClothState, visualize=True):
        obs, reward, done, info = self.step_diff(actions, state)
        actions = self.get_pnp_actions(actions, state)
        img_list = []
        for action in actions:
            state, _ = self.simulator.step_jax(state, action)
            rgb, depth = self.render(state, visualize)
            img_list.append(rgb)

        info["img_list"] = img_list
        return obs, reward, done, info

    def build_step_diff(self):
        get_obs_list = jax.vmap(self.get_obs)

        def step_diff(actions, state: ClothState):
            old_chamfer_distance = calc_chamfer(state.x, self.goal)
            pickup_place = actions[..., :3]
            particle_num = state.x.shape[-2]
            pickup_place = pickup_place[..., None, :].repeat(particle_num, -2)
            contact_distance = jnp.sqrt(jnp.sum((pickup_place - state.x) ** 2, -1)).min(-1)
            actions = self.get_pnp_actions(actions, state)
            state, state_list = jax.lax.scan(self.simulator.step_jax, state, actions, length=actions.shape[0])

            state = state._replace(cur_step=state.cur_step + 1)
            obs = self.get_obs(state)

            if self.conf.use_substep_obs:
                obs_list = get_obs_list(state_list)
            else:
                obs_list = obs
            reward, done, info = 0, state.cur_step >= self.max_steps, {"state": state, "obs_list": obs_list,
                                                                       "state_list": state_list}
            chamfer_distance = calc_chamfer(state.x, self.goal)
            reward = math.e ** (-chamfer_distance * 10)
            if self.aux_reward:
                reward += math.e ** (-contact_distance)
            real_reward = old_chamfer_distance - chamfer_distance + 0.1 * contact_distance
            info['real_reward'] = real_reward
            reward *= 0.99 ** state.cur_step
            return obs, reward, done, info

        return step_diff

    def render(self, state: ClothState, visualize=True):
        return self.renderer.render(self.get_x_grid(state)[0], self.simulator.indices, state.primitive0[0], visualize)

    def create_cloth_mask(self, conf):
        raise NotImplementedError

    def get_collision_func(self):
        def collision_func(x, v, idx_i, idx_j):
            return v

        return collision_func

    def collect_goal(self):
        assert self.batch_size == 1
        while True:
            self.simulator.key_global, _ = random.split(self.simulator.key_global)
            obs, state = self.reset(self.simulator.key_global)
            valid_episode = True
            while True:
                self.render(state)
                actions = get_expert_start_end_cloth(self.get_x_grid(state), self.cloth_mask)

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
        goal_state = np.load(self.conf.goal_path)
        goal_state = goal_state[None, ...].repeat(self.batch_size, axis=0)
        goal_grid = self.simulator.get_x_grid(goal_state)
        goal_map = get_projection(goal_grid, self.cloth_mask, size=512)
        # cv2.imshow("goal_map", goal_map[0])
        # cv2.waitKey(10)

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
                actions = get_expert_start_end_cloth(self.get_x_grid(state), self.cloth_mask, goal_map)

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

            if valid_episode:
                os.makedirs(f"{my_path}/expert_demo/{self.conf.task}", exist_ok=True)
                with open(f"{my_path}/expert_demo/{self.conf.task}/demo_{i}.pkl", "wb") as f:
                    pickle.dump(demo, f)
                    i += 1

        exit(0)

    @staticmethod
    def get_random_fold_action(state: ClothState):
        num_particle = state.x.shape[1]
        batch_size = state.x.shape[0]
        batch_idx = jnp.arange(batch_size)

        st_point = np.random.randint(0, num_particle, size=(batch_size,))
        ed_point = np.random.randint(0, num_particle, size=(batch_size,))

        actions = jnp.concatenate((state.x[batch_idx, st_point], state.x[batch_idx, ed_point]), axis=-1)
        return actions
