from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import custom_vjp, random, vmap
from jax._src.lax.control_flow import fori_loop
from jax.random import KeyArray


class ClothState(NamedTuple):
    x: jnp.ndarray
    v: jnp.ndarray
    primitive0: jnp.ndarray
    primitive1: jnp.ndarray
    action0: jnp.ndarray
    action1: jnp.ndarray
    key: KeyArray
    cur_step: jnp.ndarray
    stiffness: jnp.ndarray
    mu: jnp.ndarray


class ClothSimulator:
    def __init__(self, conf, batch_size, collision_func, cloth_mask):
        assert batch_size >= 1

        self.conf = conf
        self.batch_size = batch_size
        self.collision_func = collision_func
        self.cloth_mask = cloth_mask
        self.x_grid = None

        self.N = conf.N
        self.cell_size = 1.0 / self.N
        self.gravity = conf.gravity
        self.stiffness = conf.stiffness
        self.damping = conf.damping
        self.dt = conf.dt
        self.max_v = conf.max_v
        self.small_num = conf.small_num
        self.mu = conf.mu
        self.seed = conf.seed
        self.key_global = jax.random.PRNGKey(self.seed)

        links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
        self.links = jnp.array(links)
        self.num_triangles = (self.N - 1) * (self.N - 1) * 2

        self.idx_i, self.idx_j = jnp.nonzero(self.cloth_mask)
        self.grid_idx = jnp.concatenate([self.idx_i[:, None], self.idx_j[:, None]], axis=-1)
        self.set_indices()

        j_ = self.grid_idx.reshape((-1, 1, 2)).repeat(len(links), -2)
        j_ = j_ + self.links[None, ...]
        j_ = jnp.clip(j_, 0, self.N - 1)

        i_ = self.grid_idx.reshape((-1, 1, 2)).repeat(len(links), -2)
        original_length = self.cell_size * jnp.linalg.norm(j_ - i_, axis=-1)[..., None]
        self.ori_len_is_not_0 = (original_length != 0).astype(jnp.int32)
        self.original_length = jnp.clip(original_length, 1e-12, jnp.inf)

        self.j_x, self.j_y = j_.reshape((-1, 2))[:, 0], j_.reshape((-1, 2))[:, 1]
        self.i_x, self.i_y = i_.reshape((-1, 2))[:, 0], i_.reshape((-1, 2))[:, 1]

        self.step_jax, self.reset_jax, self.get_x_grid = self.build_core_funcs()
        self.step_jax = jax.jit(self.step_jax)
        self.step_jax = jax.vmap(self.step_jax)

    def set_indices(self):
        indices = np.zeros((self.num_triangles * 3,))
        N, cloth_mask = self.N, np.array(self.cloth_mask)

        for i, j in np.ndindex((N, N)):

            if i < N - 1 and j < N - 1:
                flag = 1
                flag *= cloth_mask[i - 1, j - 1] * cloth_mask[i - 1, j] * cloth_mask[i - 1, j + 1]
                flag *= cloth_mask[i, j - 1] * cloth_mask[i, j] * cloth_mask[i, j + 1]
                flag *= cloth_mask[i + 1, j - 1] * cloth_mask[i + 1, j] * cloth_mask[i + 1, j + 1]

                square_id = (i * (N - 1)) + j
                # 1st triangle of the square
                indices[square_id * 6 + 0] = i * N + j
                indices[square_id * 6 + 1] = (i + 1) * N + j
                indices[square_id * 6 + 2] = i * N + (j + 1)
                # 2nd triangle of the square
                indices[square_id * 6 + 3] = (i + 1) * N + j + 1
                indices[square_id * 6 + 4] = i * N + (j + 1)
                indices[square_id * 6 + 5] = (i + 1) * N + j

                indices[square_id * 6 + 0] *= flag
                indices[square_id * 6 + 1] *= flag
                indices[square_id * 6 + 2] *= flag
                # 2nd triangle of the square
                indices[square_id * 6 + 3] *= flag
                indices[square_id * 6 + 4] *= flag
                indices[square_id * 6 + 5] *= flag

        indices = jnp.array(indices.reshape((-1, 3)))
        self.indices = indices[indices.sum(1) != 0]

    def build_core_funcs(self):

        @partial(custom_vjp)
        def robot_step_wrapper(state: ClothState, action):
            return robot_step(state, action)

        def robot_step_fwd(state: ClothState, action):
            return robot_step(state, action), (state, action)

        def robot_step_bwd_loss(res, g_raw):
            g = g_raw[0]
            g_state_list = g_raw[1]
            state, action = res
            nstate, state_list = robot_step(state, action)

            loss = (nstate.x * g.x).sum()
            loss += (nstate.v * g.v).sum()
            loss += (nstate.primitive0 * g.primitive0).sum()
            loss += (nstate.primitive1 * g.primitive1).sum()
            loss += (nstate.action0 * g.action0).sum()
            loss += (nstate.action1 * g.action1).sum()
            loss += (nstate.stiffness * g.stiffness).sum()
            loss += (nstate.mu * g.mu).sum()

            loss += (state_list.x * g_state_list.x).sum()
            loss += (state_list.v * g_state_list.v).sum()
            loss += (state_list.primitive0 * g_state_list.primitive0).sum()
            loss += (state_list.primitive1 * g_state_list.primitive1).sum()
            loss += (state_list.action0 * g_state_list.action0).sum()
            loss += (state_list.action1 * g_state_list.action1).sum()
            loss += (state_list.stiffness * g_state_list.stiffness).sum()
            loss += (state_list.mu * g_state_list.mu).sum()

            return loss.sum()

        def robot_step_bwd(res, g):
            g_new = robot_step_bwd_loss(res, g)
            return g_new

        robot_step_bwd_loss = jax.grad(robot_step_bwd_loss, allow_int=True)
        robot_step_wrapper.defvjp(robot_step_fwd, robot_step_bwd)

        @partial(custom_vjp)
        def norm_grad(x):
            return x

        def norm_grad_fwd(x):
            return x, ()

        def norm_grad_bwd(x, g):
            g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t + 0.0), g)
            g_norm = optax.global_norm(g)
            trigger = g_norm < 1.0
            g = jax.tree_util.tree_map(lambda t: jnp.where(trigger, t, t / g_norm), g)
            return (g,)

        norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)

        def robot_step(state: ClothState, action):
            # state = norm_grad(state)
            # action = norm_grad(action)

            # normalize speed, 50 sub steps, 20 is a scale factor
            action0 = action.at[:3].set(action[:3].clip(-2, 2) / 50.)[:4]
            action1 = action.at[4:7].set(action[4:7].clip(-2, 2) / 50.)[4:8]

            # add uncertainty
            key, _ = random.split(state.key)
            state = state._replace(action0=action0, action1=action1, key=key)

            if self.conf.mem_saving_level > 0:
                state = fori_loop(0, 50, step_wrapper, state)
            else:
                state = fori_loop(0, 50, step, state)

            return state, (state)

        @partial(custom_vjp)
        def norm_grad(x):
            return x

        def norm_grad_fwd(x):
            return x, ()

        def norm_grad_bwd(x, g):
            g /= jnp.linalg.norm(g)
            g = jnp.nan_to_num(g)
            g /= self.cloth_mask.sum()

            return g,

        norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)

        def primitive_collision_func(x, v, action, ps):
            # collision with primitive ball
            pos, radius = ps[:3], ps[3]
            d_v = action[:3].reshape(1, 3)
            suction = action[-1]

            # find points on the surface
            x_ = x - jnp.array(pos).reshape(1, 3)
            dist = jnp.linalg.norm(x_, axis=-1)
            mask = dist <= radius
            mask = mask[..., None].repeat(3, -1)
            v_ = jnp.where(mask, suction * v, v)
            x_ = jnp.where(mask, x + d_v * (1 - suction), x)

            # weight = jnp.exp(-1 * (dist*20 - 1))[..., None]
            # v = v - weight * suction * v
            # x = x + d_v * weight

            # v_mask = jnp.abs(v).max() > max_v
            # v = jnp.where(v_mask, v, v_)
            # x = jnp.where(v_mask, x, x_)

            v = v_
            x = x_

            x = norm_grad(x)
            v = norm_grad(v)

            return x, v

        @partial(custom_vjp)
        def step_wrapper(f, state: ClothState):
            return step(f, state)

        def step_fwd(f, state: ClothState):
            return step(f, state), (f, state)

        def step_bwd_loss(res, g):
            f, state = res
            nstate = step(f, state)

            loss = (nstate.x * g.x).sum()
            loss += (nstate.v * g.v).sum()
            loss += (nstate.primitive0 * g.primitive0).sum()
            loss += (nstate.primitive1 * g.primitive1).sum()
            loss += (nstate.action0 * g.action0).sum()
            loss += (nstate.action1 * g.action1).sum()
            loss += (nstate.stiffness * g.stiffness).sum()
            loss += (nstate.mu * g.mu).sum()

            return loss.sum()

        def step_bwd(res, g):
            g_new = step_bwd_loss(res, g)
            return g_new

        step_bwd_loss = jax.grad(step_bwd_loss, allow_int=True)
        step_wrapper.defvjp(step_fwd, step_bwd)

        def step(f, state: ClothState):
            x, v = state.x, state.v
            v -= jnp.array([0, self.gravity * self.dt, 0])

            x_grid = jnp.zeros((self.N, self.N, 3)).at[self.idx_i, self.idx_j].set(x)
            relative_pos = x_grid[self.j_x, self.j_y] - x_grid[self.i_x, self.i_y]
            # current_length = jnp.linalg.norm(relative_pos, axis=-1)
            current_length = jnp.clip((relative_pos ** 2).sum(-1), 1e-12, jnp.inf) ** 0.5
            current_length = current_length.reshape((-1, len(self.links), 1))

            force = state.stiffness * relative_pos.reshape((-1, 8, 3)) / current_length * (
                    current_length - self.original_length) / self.original_length

            force *= self.ori_len_is_not_0

            # mask out force from invalid area
            force *= self.cloth_mask[self.j_x, self.j_y].reshape((-1, 8, 1))
            force = force.sum(1)
            force = force.at[:, 1].add(-self.gravity)

            # add friction
            friction_mask = x[:, 1] <= self.small_num
            muF = state.mu * force[:, 1].clip(-jnp.inf, 0) * -1
            xV = v[:, 0]
            yV = v[:, 2]
            sV = jnp.sqrt(xV ** 2 + yV ** 2 + self.small_num)  # to avoid divide by zero

            # dynamic friction
            dynamic_friction_mask = friction_mask * (sV > self.small_num)
            force = force.at[:, 0].set(force[:, 0] - dynamic_friction_mask.astype(jnp.float32) * muF * xV / sV)
            force = force.at[:, 2].set(force[:, 2] - dynamic_friction_mask.astype(jnp.float32) * muF * yV / sV)

            # static friction
            static_friction_mask = friction_mask * (sV <= self.small_num)
            xF = force[:, 0]
            yF = force[:, 2]
            sF = jnp.sqrt(xF ** 2 + yF ** 2 + self.small_num)  # to avoid divide by zero

            zero_force_mask = static_friction_mask * (muF > sF)
            force = force.at[:, 0].set(0 + (1.0 - zero_force_mask.astype(jnp.float32)) * force[:, 0])
            force = force.at[:, 2].set(0 + (1.0 - zero_force_mask.astype(jnp.float32)) * force[:, 2])

            non_zero_force_mask = static_friction_mask * (muF <= sF)
            non_zero_force_mask = non_zero_force_mask.astype(jnp.float32)
            R = 1.0 - muF / sF
            force = force.at[:, 0].set((R * xF) * non_zero_force_mask + force[:, 0] * (1.0 - non_zero_force_mask))
            force = force.at[:, 2].set((R * yF) * non_zero_force_mask + force[:, 2] * (1.0 - non_zero_force_mask))

            v += force * self.dt
            v *= jnp.exp(-self.damping * self.dt)

            # collision
            v = self.collision_func(x, v, self.idx_i, self.idx_j)
            x, v = primitive_collision_func(x, v, state.action0, state.primitive0)
            x, v = primitive_collision_func(x, v, state.action1, state.primitive1)

            # v_mask = jnp.abs(v).max() > max_v
            # ps0 = ps0.at[:3].add(action[:3]).clip(0, 1)
            # ps1 = ps1.at[:3].add(action[4:7]).clip(0, 1)
            # ps0 = jnp.where(v_mask, ps0, ps0_)
            # ps1 = jnp.where(v_mask, ps1, ps1_)

            ps0 = state.primitive0.at[:3].add(state.action0[:3]).clip(0, 1)
            ps1 = state.primitive1.at[:3].add(state.action1[:3]).clip(0, 1)

            # collision with the ground
            x = x.clip(0, 1)
            v = v.clip(-self.max_v, self.max_v)

            x += self.dt * v

            x = norm_grad(x)
            v = norm_grad(v)
            ps0 = norm_grad(ps0)
            ps1 = norm_grad(ps1)

            state = state._replace(x=x, v=v, primitive0=ps0, primitive1=ps1)
            return state

        def reset():
            # create x, v
            x = np.zeros((self.N, self.N, 3))
            for i, j in np.ndindex((self.N, self.N)):
                x[i, j] = np.array([
                    i * self.cell_size, 0,
                    (self.N - j) * self.cell_size
                ])
            self.x_grid = jnp.array(x)
            v = jnp.zeros((self.N, self.N, 3))
            ps0 = jnp.array([0.5, 0.5, 0.5, 0.01])
            ps1 = jnp.array([1, 1, 1, 0.01])

            # mask x and v
            x = self.x_grid[self.idx_i, self.idx_j]
            v = v[self.idx_i, self.idx_j]

            key, _ = jax.random.split(self.key_global)
            action = jnp.zeros((4,))
            cur_step = jnp.array(0)
            state = ClothState(x=x, v=v, primitive0=ps0, primitive1=ps1,
                               action0=action, action1=action, key=key, cur_step=cur_step,
                               stiffness=jnp.array(self.conf.stiffness), mu=jnp.array(self.conf.mu))
            state = jax.tree_util.tree_map(lambda x: x[None, ...].repeat(self.batch_size, 0), state)

            return state

        @vmap
        def get_x_grid(x):
            return self.x_grid.at[self.idx_i, self.idx_j].set(x)

        if self.conf.mem_saving_level <= 1:
            robot_step_wrapper = robot_step

        return robot_step_wrapper, reset, get_x_grid
