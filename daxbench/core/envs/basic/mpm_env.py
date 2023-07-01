import copy
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pyrender
import trimesh
from jax import jit, random
from scipy.spatial.transform import Rotation as R

from daxbench.core.engine.mpm_simulator import SimpleMPMSimulator, MPMState
from daxbench.core.engine.primitives.primitives import create_primitive, PrimitiveState
from daxbench.core.utils.util import calc_l2


class MPMEnv:
    PARTICLE = "PARTICLE"
    DEPTH = "DEPTH"
    RGB = "RGB"

    def __init__(self, conf, batch_size, max_steps, seed, focus_computation=False, use_position_control=False):

        self.simulator = SimpleMPMSimulator(conf, batch_size, use_position_control)
        self.aux_reward = False
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.cur_step = 0
        self.action_size = 6
        self.seed(seed)
        self.goal_path = conf.goal_path
        self.conf = conf
        self.focus_computation = focus_computation

        self.observation_size = 0
        self.spec = None
        self.state = None
        self.init_state = None

        self.step_diff = self.build_step_diff()
        self.step_diff = jit(self.step_diff)
        self.renderer = None
        self.effector_nodes = []

        if not os.path.exists(conf.goal_path):
            print("**************** Warning: goal file does not exist!")
            self.goal = jnp.zeros((1, 3))
        else:
            goal_map = np.load(conf.goal_path)
            self.goal = jnp.array(goal_map)

    def seed(self, seed):
        self.simulator.key_global = random.PRNGKey(seed)
        np.random.seed(seed)

    @staticmethod
    @jax.vmap
    @jax.jit
    def get_obs(state: MPMState, obs_type=PARTICLE):

        if obs_type == MPMEnv.DEPTH:
            raise NotImplementedError
        elif obs_type == MPMEnv.PARTICLE:
            obs = jnp.concatenate(
                [
                    state.x.flatten(),
                    state.v.flatten(),
                    state.primitives[0].position.flatten()
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        return obs

    @staticmethod
    def get_primitive_actions(actions, state):
        raise NotImplementedError

    @staticmethod
    def process_pre_step_actions(actions, shift):
        raise NotImplementedError

    @staticmethod
    def auto_reset(state, state_new, key):
        raise NotImplementedError

    @staticmethod
    def reward_func(state, goal):
        l2_distance = calc_l2(state.x, goal)
        reward = math.e ** (-l2_distance * 10)
        return reward

    def build_step_diff(self):
        get_obs_list = jax.vmap(MPMEnv.get_obs)

        def pre_step(actions, state: MPMState):
            # focus point pre-computation
            state_center = state.x.mean(1)
            target_center = jnp.array(self.conf.res) * 0.5 / self.conf.n_grid
            shift = target_center - state_center
            shift = shift.at[:, 1].set(0)

            # TODO: this only work on push task
            actions = self.process_pre_step_actions(actions, shift)

            shift = shift[:, None, :]
            for i in range(self.conf.n_primitive):
                state.primitives[i] = state.primitives[i]._replace(position=state.primitives[i].position + shift)
            state = state._replace(x=state.x + shift)

            return actions, state, shift

        def post_step(state, state_list, shift):
            # focus point post computation
            state = state._replace(x=state.x - shift)
            state_list = state_list._replace(x=state_list.x - shift[None, ...])

            for i in range(self.conf.n_primitive):
                state.primitives[i] = state.primitives[i]._replace(position=state.primitives[i].position - shift)
                state_list.primitives[i] = state_list.primitives[i]._replace(
                    position=state_list.primitives[i].position - shift[None, ...])
            return state, state_list

        def right_broadcasting(arr, target):
            return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))

        def step_diff(actions, state: MPMState):
            pickup_place = actions[..., :3]
            particle_num = state.x.shape[-2]
            pickup_place = pickup_place[..., None, :].repeat(particle_num, -2)

            contact_distance = jnp.sqrt(jnp.sum((pickup_place - state.x) ** 2, -1)).min(-1)
            if self.focus_computation:
                actions, state, shift = pre_step(actions, state)
            actions, state = self.get_primitive_actions(actions, state)
            actions = actions.swapaxes(0, 1)

            state, state_list = jax.lax.scan(self.simulator.step_jax, state, actions, length=actions.shape[0])
            state = state._replace(cur_step=state.cur_step + 1)

            if self.focus_computation:
                state, state_list = post_step(state, state_list, shift)

            reward, done = 0, state.cur_step >= self.max_steps

            # Important, calculate the reward before auto reset, otherwise the reward will be the same
            state = state._replace(x=jnp.nan_to_num(state.x))
            state = state._replace(v=jnp.nan_to_num(state.v))
            state = state._replace(C=jnp.nan_to_num(state.C))
            state = state._replace(F=jnp.nan_to_num(state.F))
            state = state._replace(J=jnp.nan_to_num(state.J))

            reward = self.reward_func(state, self.goal)
            if self.aux_reward:
                reward += math.e ** (-contact_distance)
            new_state = self.auto_reset(self.init_state, state, state.key)
            new_state = jax.lax.stop_gradient(new_state)
            state = jax.tree_util.tree_map(lambda x, y: jnp.where(right_broadcasting(done, x), y, x), state, new_state)

            obs = self.get_obs(state)
            obs_list = get_obs_list(state_list)

            info = {"state": state, "state_list": state_list, "obs_list": obs_list}
            return obs, reward, done, info

        return step_diff

    def step_with_render(self, actions, state: MPMState, visualize=True):
        obs, reward, done, info = self.step_diff(actions, state)

        actions, _ = self.get_primitive_actions(actions, state)
        actions = actions.swapaxes(0, 1)
        img_list = []
        for i in range(len(actions)):
            state = jax.tree_util.tree_map(lambda x: x[i], info['state_list'])
            rgb, depth = self.render(state, visualize)
            img_list.append(rgb)

        info["img_list"] = img_list
        return obs, reward, done, info

    def clean_up_b4_reset(self):
        print("reset in progress")
        if self.state is not None:
            self.state = self.state._replace(primitives=[])
        self.state = None

        for node in self.effector_nodes:
            self.renderer.scene.remove_node(node)
        self.effector_nodes = []

    def initialize_after_adding_particle_primitives(self, state):
        # initialize state, important!
        self.state = self.simulator.reset_jax(state)
        self.init_state = copy.deepcopy(self.state)

        num_particles = self.state.x.shape[-2]
        if self.goal is not None and num_particles != self.goal.shape[-2]:
            # resample goal to match the number of particles
            idx_list_from_goal = np.random.choice(self.goal.shape[-2], num_particles, replace=True)
            self.goal = self.goal[idx_list_from_goal]

    def create_mesh_for_render(self, size):
        boxf_trimesh = trimesh.creation.box(extents=size[[0, 2, 1]] * 2)
        boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
        boxf_trimesh.visual.face_colors = boxf_face_colors
        boxf_mesh = pyrender.Mesh.from_trimesh(boxf_trimesh, smooth=False)
        return boxf_mesh

    def create_primitive(self, conf, state, friction, color, size, init_pos, softness=666):
        size = np.array(size)
        init_pos = np.array(init_pos)
        p_state = create_primitive(conf, friction=friction, softness=softness, color=color,
                                   size=size, init_pos=init_pos)
        state.primitives.append(p_state)

        # add to renderer
        primitive_mesh = self.create_mesh_for_render(size)
        self.effector_nodes.append(self.renderer.scene.add(primitive_mesh))
        return state

    def reset(self, key):
        raise NotImplementedError

    def render(self, state, visualize=False):
        def process_orientation(p_state: PrimitiveState):
            pos = np.array(p_state.position[0])[0]
            pos = np.array(pos)[[0, 2, 1]]
            rot = np.array(p_state.rotation[0])[0]
            rot = R.from_quat(rot).as_euler('xyz')
            rot[2] *= -1
            rot = R.from_euler('zyx', rot).as_quat()
            return pos, rot

        for i, effector_node in enumerate(self.effector_nodes):
            pos, rot = process_orientation(state.primitives[i])
            effector_node.translation = pos
            effector_node.rotation = rot
        return self.renderer.render(state, visualize)
