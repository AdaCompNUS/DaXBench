import glob
import os
import pickle
import time
from dataclasses import dataclass, field
from functools import partial

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, custom_vjp

from daxbench.core.engine.mpm_simulator import MPMState
from daxbench.core.engine.primitives.box import _sdf_batch as box_sdf
from daxbench.core.engine.primitives.primitives import set_sdf
from daxbench.core.engine.pyrender.py_render import ParticlePyRenderer
from daxbench.core.envs.basic.mpm_env import MPMEnv

my_path = os.path.dirname(os.path.abspath(__file__))


def _d_gravity():
    return jnp.array([0, -9.8, 0])


@dataclass
class DefaultConf:
    seed = 1
    n_primitive = 1
    focus_computation = True
    use_position_control = True
    obs_type = MPMEnv.PARTICLE
    key = random.PRNGKey(0)

    """
    n_grid: resolution of the simulation.
    steps: number of internal steps.
        - larger steps result in lower gripper speed.
        - if rope slides too far, can (1) increase ground_friction (2) increase steps.
        - larger steps will slow down the simulation
    dt: internal step time.  higher rigidness requires lower dt.
    E: rigidness of object or resistance to deformation.
    h: the speed to recover its original shape after deformation. (used when create the object)
    """
    ground_friction: float = 0.1
    n_grid: int = 64
    dt: float = 1e-4
    primitive_action_steps = 1
    primitive_action_duration = 0.007  # seconds
    steps = int(primitive_action_duration / primitive_action_steps / dt)  # internal steps
    E: float = 100
    nu: float = 0.1

    # ground_friction = 0.1
    # n_grid, steps, dt = 64, 50, 1e-4
    # # E, nu = 500, 0.2
    # E, nu = 100, 0.2
    res: ... = (n_grid // 2, n_grid // 2, n_grid // 2)

    dx, inv_dx = 1 / n_grid, float(n_grid)
    p_vol, p_rho = (dx * 0.5) ** 2, 1
    p_mass = p_vol * p_rho
    gravity: jnp.ndarray = field(default_factory=_d_gravity)  # gravity ...

    task = "whip_rope"
    goal_path = f"{my_path}/goals/{task}/goal.npy"

    ## Rope property
    rope_width = [0.38, 0.006, 0.006]
    rope_init_pos = [0.5, 0.01, 0.5]
    rope_z_rotation_angle = np.pi / 2
    rope_hardness = 1.0  # 0.1 - 5. How fast the shape recovers.


WhipRopeConfig = DefaultConf


class WhipRopeEnv(MPMEnv):

    def __init__(self, batch_size, seed, max_steps=70, conf=None, aux_reward=False, **kwargs):
        conf = DefaultConf() if conf is None else conf
        self.conf = conf
        self.focus_computation = True

        super().__init__(conf, batch_size, max_steps, seed, conf.focus_computation, conf.use_position_control)
        self.observation_size = 612
        self.renderer = ParticlePyRenderer()

    @staticmethod
    def process_pre_step_actions(actions, shift):
        return actions

    @staticmethod
    @jax.vmap
    def auto_reset(state, state_new, key):
        init_pos = state.primitives[0].position[0]
        key, _ = random.split(key)
        shift = random.normal(key, (2,)) * 0.02
        init_pos = init_pos.at[..., [0, 2]].add(shift)
        position = state.primitives[0].position.at[0].set(init_pos)
        state.primitives[0] = state.primitives[0]._replace(position=position)

        state = state._replace(x=state.x.at[:, [0, 2]].add(shift[None, ...]))
        state = state._replace(key=key)
        return state

    @staticmethod
    @jax.vmap
    def get_primitive_actions(actions, state: MPMState):

        actions = actions + 1e-12  # hack to avoid nan
        actions /= 50.0
        actions = actions.at[..., 3:].set(0)
        return actions[None, ...], state

    def reset(self, key):
        # clean up env before reset, important!
        self.clean_up_b4_reset()

        # set sdf function for collision, also set type of end effector
        set_sdf(box_sdf)

        # create rope
        state = self.simulator.add_box(conf=self.conf, state=None, hardness=self.conf.rope_hardness,
                                       size=self.conf.rope_width, init_pos=self.conf.rope_init_pos,
                                       z_rotation_angle=self.conf.rope_z_rotation_angle, material=1, density=2.75)

        # add primitive end effector to the state
        state = self.create_primitive(self.conf, state=state, friction=0.1, color=[0.5, 0.5, 0.5],
                                      size=[0.02, 0.02, 0.02], init_pos=[0.5, 0.01, 0.3])

        # initialize state after adding particles and primitives, important!
        self.initialize_after_adding_particle_primitives(state)

        self.state = self.auto_reset(self.init_state, self.init_state, self.init_state.key)
        return self.get_obs(self.state), self.state

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

                if i == 0:
                    # save goal
                    os.makedirs(f"{my_path}/goals/{self.conf.task}", exist_ok=True)
                    np.save(f"{my_path}/goals/{self.conf.task}/goal.npy", state.x[0])

                i += 1

        exit(0)

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
            return np.array([0, -unit, 0, 0, 0, 0])
        elif k == 9:  # shift
            return np.array([0, unit, 0, 0, 0, 0])

        return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)


def grad_test():
    env = WhipRopeEnv(batch_size=1, seed=1)
    obs, first_state = env.reset(env.simulator.key)
    actions = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)[None, ...]

    def loss_fn(actions, state):
        def step_(i, carry):
            (actions, state) = carry
            state = norm_grad(state)
            actions = norm_grad(actions)
            obs, reward, done, info = env.step_diff(actions, state)
            state = info["state"]

            return (actions, state)

        @partial(custom_vjp)
        def norm_grad(x):
            return x

        def norm_grad_fwd(x):
            return norm_grad(x), ()

        def norm_grad_bwd(x, g):
            g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t + 0.0), g)
            return (g,)

        norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)
        actions, state = jax.lax.fori_loop(0, 5, step_, (actions, state))
        return state.x.sum(), state

    loss_fn = jax.jit(jax.grad(loss_fn, has_aux=True))
    state = first_state
    for i in range(100):
        actions = env.get_exp_action()[None, ...].repeat(env.batch_size, axis=0)
        env.simulator.key_global, _ = jax.random.split(env.simulator.key_global)
        # state = env.auto_reset(first_state, first_state, env.simulator.key_global[None,...])

        grad_raw, state = loss_fn(actions, state)
        env.render(state, True)
        print(i, grad_raw, state.x[0].mean(0))


if __name__ == "__main__":
    env = WhipRopeEnv(batch_size=1, seed=1)
    obs, state = env.reset(env.simulator.key)

    print("time start")
    start_time = time.time()
    for i in range(200):
        actions = env.get_exp_action()[None, ...].repeat(env.batch_size, axis=0)
        # obs, reward, done, info = env.step_diff(actions, state)
        obs, reward, done, info = env.step_with_render(actions, state)
        state = info["state"]
        print("step", i, state.primitives[0].position[0], state.primitives[0].v[0])
        print(state.cur_step, "reward", reward)
    print(time.time() - start_time)
