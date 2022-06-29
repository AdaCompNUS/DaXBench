import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import NamedTuple, List
from jax import random, custom_vjp
from daxbench.core.engine.svd_safe_batch import svd
from daxbench.core.engine.primitives.primitives import PrimitiveState, set_action, forward_kinematics, collide_batch, \
    position_control_batch
from jax._src.lax.control_flow import fori_loop


class MPMState(NamedTuple):
    x: jnp.ndarray = None
    v: jnp.ndarray = None
    C: jnp.ndarray = None
    F: jnp.ndarray = None
    J: jnp.ndarray = None
    cur_step: jnp.ndarray = None
    primitives: List[PrimitiveState] = []
    key: jnp.ndarray = None
    friction: jnp.ndarray = jnp.array(1.0)
    mu: jnp.ndarray = jnp.array(1.0)  # will be assign value later
    lamda: jnp.ndarray = jnp.array(1.0)


class SimpleMPMSimulator:
    def __init__(self, conf, batch_size, use_position_control=False):
        """
        define hyper-para
        """
        self.conf = conf
        self.key_global = None
        self.batch_size = batch_size
        self.ground_friction = conf.ground_friction
        self.res = conf.res
        self.dt = conf.dt
        self.dx = conf.dx
        self.inv_dx = conf.inv_dx
        self.p_mass = conf.p_mass
        self.p_vol = conf.p_vol
        self.gravity = conf.gravity
        self.key = conf.key
        self.n_particles = 0
        self.n_grid = conf.n_grid
        self.use_position_control = use_position_control
        # self.n_primitive = conf.n_primitive
        E, nu = conf.E, conf.nu  # Young's modulus and Poisson's ratio

        a, b, c = jnp.indices((3, 3, 3))
        self.idx = jnp.concatenate((a[..., None], b[..., None], c[..., None]), axis=3).reshape((-1, 3))

        a, b, c = jnp.indices(self.res)
        self.grid_idx = jnp.concatenate((a[..., None], b[..., None], c[..., None]), axis=3).reshape((-1, 3))
        self.grid_idx_3d = self.grid_idx.reshape(self.res + (3,))

        self.material = None
        self.h = None
        # self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.step_jax, self.reset_jax = self.build_core_funcs()
        self.step_jax = jax.jit(self.step_jax)
        self.step_jax = jax.vmap(self.step_jax)

    def add_box(self, conf, state, size, init_pos, hardness=1, z_rotation_angle=0, material=0, density=1):

        """
        add a box to the scene
        :param conf:
        :param state:
        :param size:
        :param init_pos:
        :param hardness: 0.1 - 5. How fast the shape recovers.
        :param z_rotation_angle:
        :param material: 0: water, 1: jelly & rigid, 2: soft
        :param density: 1 - 5
        :return:
        """

        assert density >= 1

        size, init_pos = jnp.array(size), jnp.array(init_pos)
        rotation_matrix = jnp.array([[jnp.cos(z_rotation_angle), -jnp.sin(z_rotation_angle)],
                                     [jnp.sin(z_rotation_angle), jnp.cos(z_rotation_angle)]])

        # add particles in random positions
        if material == 0:
            n_points = int(size.prod() * conf.n_grid ** 3 * density)
            x_ = (random.uniform(conf.key, (n_points, 3)) * 2 - 1) * (0.5 * size)
            x_ = x_.at[:, [0, 2]].set(x_[:, [0, 2]] @ rotation_matrix.T)
            x_ = x_ + init_pos

        else:
            # generate particles in grid
            n_grid = int(conf.n_grid * density)
            center = jnp.array([0.5, 0.01, 0.5])
            lower = (jnp.zeros(3) * 2 - 1) * (0.5 * size) + center
            upper = (jnp.ones(3) * 2 - 1) * (0.5 * size) + center
            a, b, c = jnp.indices((n_grid, n_grid, n_grid))
            grid_idx = jnp.concatenate((a[..., None], b[..., None], c[..., None]), axis=3)
            grid_idx = grid_idx * 1.0 / n_grid
            mask = grid_idx <= upper
            mask *= grid_idx >= lower
            mask = mask.astype(jnp.int32).prod(axis=-1).astype(jnp.bool_)
            x_ = grid_idx[mask]
            x_ = x_ - center
            x_ = x_.at[:, [0, 2]].set(x_[:, [0, 2]] @ rotation_matrix.T)
            x_ = x_ + init_pos
            n_points = x_.shape[0]

        # set material and hardness
        material_ = jnp.full((n_points,), material)
        h_ = jnp.full((n_points,), hardness)

        if state is None:
            # start of the reset function
            self.material = material_
            self.h = h_
        else:
            x_ = jnp.concatenate((state.x, x_), axis=0)
            self.material = jnp.concatenate((self.material, material_), axis=0)
            self.h = jnp.concatenate((self.h, h_), axis=0)

        state_ = MPMState(x=x_)
        return state_

    def add_box_from_points(self, conf, state, points, hardness=1, material=0):
        x_ = points
        n_points = x_.shape[0]

        # set material and hardness
        material_ = jnp.full((n_points,), material)
        h_ = jnp.full((n_points,), hardness)

        if state is None:
            # start of the reset function
            self.material = material_
            self.h = h_
        else:
            x_ = jnp.concatenate((state.x, x_), axis=0)
            self.material = jnp.concatenate((self.material, material_), axis=0)
            self.h = jnp.concatenate((self.h, h_), axis=0)

        state_ = MPMState(x=x_)
        return state_

    def build_core_funcs(self):
        """
        Reset Functions
        """

        def reset(state: MPMState):
            # self.n_primitive = len(state.primitives)
            self.n_particles = state.x.shape[0]
            v = jnp.zeros((self.n_particles, 3), dtype=jnp.float32)  # velocity
            C = jnp.zeros((self.n_particles, 3, 3), dtype=jnp.float32)  # affine velocity field
            F = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3).repeat(self.n_particles, axis=0)
            J = jnp.ones((self.n_particles,), dtype=jnp.float32)  # velocity
            cur_step = jnp.array(0)

            E, nu = self.conf.E, self.conf.nu
            mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            state = state._replace(v=v, C=C, F=F, J=J, cur_step=cur_step, key=self.key_global,
                                   friction=jnp.array([self.conf.ground_friction]),
                                   mu=jnp.array([mu_0]),
                                   lamda=jnp.array([lambda_0]))
            state = jax.tree_util.tree_map(lambda x: x[None, ...].repeat(self.batch_size, 0), state)

            key = jax.random.split(self.key_global, self.batch_size)
            state = state._replace(key=key)

            return state

        """
        Step Functions
        """

        def p2g_micro(v, grid_v, grid_m, fx, w, base, affine, idx):
            i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
            offset = idx[:, None, :].repeat(self.n_particles, axis=1)

            dpos = (offset.astype(jnp.float32) - fx) * self.dx
            weight = w[i][:, :, 0] * w[j][:, :, 1] * w[k][:, :, 2]
            pos_in_grid = base + offset
            grid_v_vals = weight.reshape((27, self.n_particles, 1)).repeat(3, axis=2) * (
                    self.p_mass * v + (affine @ dpos[..., None]).squeeze())

            pos_in_grid = pos_in_grid.reshape(-1, 3)
            grid_m = grid_m.at[pos_in_grid[:, 0], pos_in_grid[:, 1], pos_in_grid[:, 2]] \
                .add(weight.flatten() * self.p_mass)
            grid_v = grid_v.at[pos_in_grid[:, 0], pos_in_grid[:, 1], pos_in_grid[:, 2]] \
                .add(grid_v_vals.reshape((-1, 3)))

            return grid_v, grid_m

        def g2p_micro(grid_v, fx, w, base, new_v, new_C, idx):
            i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
            offset = idx[:, None, :].repeat(self.n_particles, axis=1)

            # dpos = (offset.astype(jnp.float32) - fx) * self.dx
            dpos = (offset.astype(jnp.float32) - fx)
            dpos = dpos.reshape(-1, 3)

            weight = w[i][:, :, 0] * w[j][:, :, 1] * w[k][:, :, 2]
            weight = weight.flatten()
            pos_in_grid = base + offset
            pos_in_grid = pos_in_grid.reshape(-1, 3)

            # TODO: this step is slow
            g_v = grid_v[pos_in_grid[:, 0], pos_in_grid[:, 1], pos_in_grid[:, 2]]

            new_v = new_v.reshape((-1, 3))
            new_v += jnp.expand_dims(weight, 1).repeat(3, axis=1) * g_v
            new_v = new_v.reshape((27, -1, 3)).sum(0)

            outer_res = jnp.einsum('ij,ik->ijk', g_v, dpos)
            new_C = new_C.reshape((-1, 3, 3))
            new_C += 4 * weight[:, None, None].repeat(3, 1).repeat(3, 2) * outer_res * self.inv_dx
            new_C = new_C.reshape((27, -1, 3, 3)).sum(0)

            return new_v, new_C

        def substep(f, state: MPMState):
            grid_v = jnp.zeros(self.res + (3,))
            grid_m = jnp.zeros(self.res)

            liquid_mask = self.material == 0
            plastic_mask = self.material == 2

            """
            p2g
            """
            base = (state.x * self.inv_dx - 0.5).astype(jnp.int32)
            fx = state.x * self.inv_dx - base.astype(jnp.float32)
            w = jnp.array([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2])

            # deformation gradient update
            F_ = (jnp.eye(3)[None, ...] + self.dt * state.C) @ state.F
            state = state._replace(F=F_)

            h = self.h.clip(0.1, 5)
            mu, la = state.mu * h, state.lamda * h
            mu = jnp.where(liquid_mask, 0.0, mu)  # comment it
            la = jnp.where(liquid_mask, 1.0, la)

            U, sig, V = svd(state.F)
            J = jnp.ones((self.n_particles,), dtype=jnp.float32)

            # plastic mask
            sig_ = sig.clip(1 - 2.5e-2 * 10, 1 + 4.5e-3 * 100)
            sig = jnp.where(plastic_mask[..., None], sig_, sig)

            J *= sig.prod(-1)
            J = J[..., None, None]

            sig_ = jnp.eye(3)[None, ...] * sig[..., None]
            F_ = jnp.where(plastic_mask[..., None, None], U @ sig_ @ V, state.F)
            state = state._replace(F=F_)

            # Reset deformation gradient to avoid numerical instability for liquid
            # F_ = jnp.where(liquid_mask[..., None, None], jnp.eye(3)[None, ...] * jnp.sqrt(J), state.F)
            # state = state._replace(F=F_)

            # soft object
            stress = 2 * mu[..., None, None] * (state.F - U @ V) @ state.F.transpose((0, 2, 1)) \
                     + jnp.eye(3).reshape((1, 3, 3)) * la[..., None, None] * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx ** 2
            affine = stress + self.p_mass * state.C

            v_ = state.v[None, ...].repeat(27, 0)
            fx_ = fx[None, ...].repeat(27, 0)
            base_ = base[None, ...].repeat(27, 0)
            affine_ = affine[None, ...].repeat(27, 0)
            grid_v, grid_m = p2g_micro(v_, grid_v, grid_m, fx_, w, base_, affine_, self.idx)

            # primitive dynamics
            for i in range(self.conf.n_primitive):
                state.primitives[i] = forward_kinematics(f, state.primitives[i])

            """
            grid operation
            """
            grid_v_ = grid_v / grid_m[..., None]
            grid_v = jnp.where(grid_m[..., None] > 0, grid_v_, grid_v)
            grid_v += self.dt * self.gravity

            # collide with primitives

            if self.use_position_control:
                for i in range(self.conf.n_primitive):
                    grid_v = position_control_batch(f, self.grid_idx_3d * self.dx, grid_v, self.dt, state.primitives[i])
            else:
                for i in range(self.conf.n_primitive):
                    grid_v = collide_batch(f, self.grid_idx_3d * self.dx, grid_v, self.dt, state.primitives[i])

            # deal with friction
            normal = jnp.array([0, 1, 0])
            lin = jnp.dot(grid_v, normal) + 1e-30
            vit = grid_v - lin[..., None] * normal.reshape((1, 1, 1, 3)) - self.grid_idx_3d * 1e-30
            lit = jnp.linalg.norm(vit + 1e-12, axis=3)

            grid_v_ = jnp.clip(1. + state.friction * lin[..., None] / lit[..., None], 0., jnp.inf) * (
                    vit + self.grid_idx_3d * 1e-30)
            grid_v_ = grid_v_.at[:, :, :, 1].set(0)
            friction_mask = jnp.zeros_like(grid_v).at[:, :3, :, :].set(1)
            fric_speed_mask = grid_v[..., 1] <= 0
            grid_v = jnp.where(friction_mask * fric_speed_mask[..., None], grid_v_, grid_v)

            # boundary condition
            a, b, c = self.grid_idx[:, 0], self.grid_idx[:, 1], self.grid_idx[:, 2]
            cond = (self.grid_idx < 3) & (grid_v[a, b, c] < 0) | \
                   (self.grid_idx > self.n_grid - 3) & (grid_v[a, b, c] > 0)
            grid_v = jnp.where(cond.reshape(self.res + (3,)), 0.0, grid_v)

            """
            g2p
            """
            new_v = jnp.zeros((self.n_particles, 3))
            new_C = jnp.zeros((self.n_particles, 3, 3))
            new_v_ = jnp.expand_dims(new_v, 0).repeat(27, 0)
            new_C_ = jnp.expand_dims(new_C, 0).repeat(27, 0)
            fx_ = jnp.expand_dims(fx, 0).repeat(27, 0)
            base_ = jnp.expand_dims(base, 0).repeat(27, 0)
            v_, C_ = g2p_micro(grid_v, fx_, w, base_, new_v_, new_C_, self.idx)

            x_ = state.x + self.dt * v_
            J_ = state.J * (1 + self.dt * C_.trace().sum(-1))
            state = state._replace(x=x_, v=v_, C=C_, J=J_)

            return state

        @partial(custom_vjp)
        def substep_wrapper(f, state: MPMState):
            return substep(f, state)

        def substep_fwd(f, state: MPMState):
            return substep(f, state), (f, state)

        def substep_bwd_loss(res, g):
            f, state = res
            nstate = substep(f, state)

            loss = (nstate.x * g.x).sum()
            loss += (nstate.v * g.v).sum()
            loss += (nstate.C * g.C).sum()
            loss += (nstate.F * g.F).sum()

            loss += (nstate.friction * g.friction).sum()
            loss += (nstate.mu * g.mu).sum()
            loss += (nstate.lamda * g.lamda).sum()

            for i in range(self.conf.n_primitive):
                sum_val = jax.tree_util.tree_map(lambda x, y: (x * y).sum(), nstate.primitives[i], g.primitives[i])
                loss += jax.tree_util.tree_reduce(lambda x, y: x + y, sum_val)

            return loss.sum()

        def substep_bwd(res, g):
            g_new = substep_bwd_loss(res, g)
            return g_new

        substep_bwd_loss = jax.grad(substep_bwd_loss, allow_int=True)
        substep_wrapper.defvjp(substep_fwd, substep_bwd)

        def copy_frame(source, target, state: MPMState):

            for i in range(self.conf.n_primitive):
                p_state = state.primitives[i]
                position = p_state.position.at[target].set(p_state.position[source])
                rotation = p_state.rotation.at[target].set(p_state.rotation[source])
                state.primitives[i] = state.primitives[i]._replace(position=position, rotation=rotation)

            return state

        @partial(custom_vjp)
        def norm_grad_state(x):
            x = x._replace(x=jnp.nan_to_num(x.x))
            x = x._replace(v=jnp.nan_to_num(x.v))
            x = x._replace(C=jnp.nan_to_num(x.C))
            x = x._replace(F=jnp.nan_to_num(x.F))
            x = x._replace(J=jnp.nan_to_num(x.J))

            return x
            # jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t + 0.0), x)

        def norm_grad_state_fwd(x):
            return norm_grad_state(x), ()

        def norm_grad_state_bwd(x, g):
            g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t + 0.0), g)
            g_norm = optax.global_norm(g)
            trigger = g_norm < 1.0
            g = jax.tree_util.tree_map(lambda t: jnp.where(trigger, t, t / g_norm), g)
            return (g,)

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

        norm_grad_state.defvjp(norm_grad_state_fwd, norm_grad_state_bwd)
        norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)

        def step(state: MPMState, action):

            state = norm_grad_state(state)
            action = norm_grad(action)

            # primitives actions
            action = action.clip(-1, 1)

            # set action for the first primitive, other primitives not moving
            for i in range(self.conf.n_primitive):
                state.primitives[i] = set_action(self.conf.steps, action[i * 6:(i + 1) * 6], state.primitives[i])

            # replace for with fori_loop save a lot compile time!
            state = fori_loop(0, self.conf.steps, substep_wrapper, state)
            state = copy_frame(self.conf.steps, 0, state)

            return state, (state)

        return step, reset
