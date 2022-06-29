from typing import NamedTuple

import jax.numpy as jnp

# sdf func to be set dynamically later
_sdf_batch = None


class PrimitiveState(NamedTuple):
    size: jnp.ndarray
    dim: jnp.ndarray
    friction: jnp.ndarray
    softness: jnp.ndarray
    color: jnp.ndarray
    position: jnp.ndarray
    rotation: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    xyz_limit: jnp.ndarray
    action_buffer: jnp.ndarray
    action_scale: jnp.ndarray
    min_dist: jnp.ndarray
    dist_norm: jnp.ndarray


def set_sdf(sdf_func):
    global _sdf_batch
    _sdf_batch = sdf_func


def create_primitive(conf, friction, softness, color, size, init_pos):
    dim = jnp.array([3])
    action_dim = 6
    max_steps = conf.steps

    size = jnp.array(size)
    friction = jnp.array(friction)
    softness = jnp.array(softness)
    color = jnp.array(color)

    position = jnp.zeros((max_steps, 3))
    position = position.at[0].set(jnp.array(init_pos))
    position = position
    rotation = jnp.array([[1., 0., 0., 0.]]).repeat(max_steps, axis=0)  # quaternion
    v = jnp.zeros((max_steps, 3))  # velocity
    w = jnp.zeros((max_steps, 3))  # angular velocity
    xyz_limit = jnp.array([[0., 1.],
                           [0., 1.],
                           [0., 1.]])

    # action_buffer = jnp.zeros((max_steps, action_dim))
    action_buffer = jnp.zeros((action_dim,))
    action_scale = jnp.ones((action_dim,))
    min_dist = jnp.array(0)  # min distance to the point cloud..
    dist_norm = jnp.array(0)

    primitive_state = PrimitiveState(size, dim, friction, softness, color, position, rotation,
                                     v, w, xyz_limit, action_buffer, action_scale, min_dist, dist_norm)

    return primitive_state


"""
_sdf and _normal are customized for box
"""


def length(x):
    return jnp.sqrt(jnp.matmul(x[:, :, :, None, :], x[..., None]) + 1e-12).squeeze()
    # return jnp.sqrt(x.dot(x) + 1e-14)


def qmul(q, r):
    # terms = r.outer_product(q)
    terms = jnp.outer(r, q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = jnp.array([w, x, y, z])
    return out / jnp.clip(jnp.sqrt(out.dot(out)), 1e-12, jnp.inf)  # normalize it to prevent some unknown NaN problems.


def w2quat(axis_angle):
    # w = axis_angle.norm()
    w = jnp.linalg.norm(axis_angle) + 1e-12
    out = jnp.array([1., 0., 0., 0.])
    # if w > 1e-9:
    v = (axis_angle / w) * jnp.sin(w / 2)
    out = out.at[0].set(jnp.cos(w / 2))
    out = out.at[1:4].set(v[:3])
    return out


def qrot_batch(rot, v):
    # rot: vec4, p vec3
    qvec = jnp.array([rot[1], rot[2], rot[3]])
    uv = jnp.cross(qvec, v)
    uuv = jnp.cross(qvec, uv)
    # uv = qvec.cross(v)
    # uuv = qvec.cross(uv)
    return v + 2 * (rot[0] * uv + uuv)


def inv_trans_batch(pos, position, rotation):
    # assert jnp.linalg.norm(rotation) > 0.9
    inv_quat = jnp.array([rotation[0], -rotation[1], -rotation[2], -rotation[3]])
    inv_quat = inv_quat / (jnp.linalg.norm(inv_quat) + 1e-12)
    return qrot_batch(inv_quat, pos - position)


def sdf_batch(f, grid_pos, state: PrimitiveState):
    grid_pos = inv_trans_batch(grid_pos, state.position[f], state.rotation[f])
    return _sdf_batch(state.size, f, grid_pos)


def _normal_batch(f, grid_pos, state: PrimitiveState):
    # TODO: replace it with analytical normal later..
    d = 1.e-6
    n = jnp.zeros_like(grid_pos)

    inc = grid_pos.at[..., 0].add(d)
    dec = grid_pos.at[..., 0].add(-d)
    n = n.at[..., 0].set((0.5 / d) * (_sdf_batch(state.size, f, inc) - _sdf_batch(state.size, f, dec)))

    inc = grid_pos.at[..., 1].add(d)
    dec = grid_pos.at[..., 1].add(-d)
    n = n.at[..., 1].set((0.5 / d) * (_sdf_batch(state.size, f, inc) - _sdf_batch(state.size, f, dec)))

    inc = grid_pos.at[..., 2].add(d)
    dec = grid_pos.at[..., 2].add(-d)
    n = n.at[..., 2].set((0.5 / d) * (_sdf_batch(state.size, f, inc) - _sdf_batch(state.size, f, dec)))

    return n / length(n)[..., None]


def normal_batch(f, grid_pos, state: PrimitiveState):
    # n2 = normal2(f, grid_pos)
    # xx = grid_pos
    grid_pos = inv_trans_batch(grid_pos, state.position[f], state.rotation[f])
    return qrot_batch(state.rotation[f], _normal_batch(f, grid_pos, state))


def collider_v_batch(f, grid_pos, dt, state: PrimitiveState):
    inv_quat = jnp.array([state.rotation[f][0], -state.rotation[f][1], -state.rotation[f][2], -state.rotation[f][3]])
    inv_quat = inv_quat / (jnp.linalg.norm(inv_quat) + 1e-12)

    relative_pos = qrot_batch(inv_quat, grid_pos - state.position[f])
    new_pos = qrot_batch(state.rotation[f + 1], relative_pos) + state.position[f + 1]
    collider_v = (new_pos - grid_pos) / dt  # TODO: revise
    return collider_v


def collide_batch(f, grid_pos, v_out, dt, state: PrimitiveState):
    dist = sdf_batch(f, grid_pos, state)
    # influence = min(jnp.exp(-dist * softness), 1)
    influence = jnp.clip(jnp.exp(-dist * state.softness), -jnp.inf, 1)[..., None]

    # if (softness > 0 and influence > 0.1) or dist <= 0:
    D = normal_batch(f, grid_pos, state)
    collider_v_at_grid = collider_v_batch(f, grid_pos, dt, state)

    input_v = v_out - collider_v_at_grid

    normal_component = jnp.matmul(input_v[:, :, :, None, :], D[..., None]).squeeze()[..., None]
    # normal_component = input_v.dot(D)

    grid_v_t = input_v - jnp.clip(normal_component, -jnp.inf, 0.) * D

    grid_v_t_norm = length(grid_v_t)[..., None]
    grid_v_t_friction = grid_v_t / grid_v_t_norm * jnp.clip(grid_v_t_norm + normal_component * state.friction, 1e-12,
                                                            jnp.inf)
    # grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0, grid_v_t_norm + normal_component * friction)

    grid_v_t_dot = jnp.matmul(grid_v_t[:, :, :, None, :], grid_v_t[..., None]).reshape(normal_component.shape)

    # TODO check gradient here
    flag = (normal_component < 0).astype(jnp.int32) * (jnp.sqrt(grid_v_t_dot) > 1e-12).astype(jnp.int32) * 1.0
    # flag = int(normal_component < 0 and jnp.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30) * 1.0
    grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
    v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence
    return v_out


def forward_kinematics(f, state: PrimitiveState):
    position = state.position.at[f + 1].set(state.position[f] + state.v[f])
    position = jnp.clip(position, -2, 2)

    # rotate in world coordinates about it
    # rotation[f + 1] = qmul(w2quat(w[f]), rotation[f])
    rotation = state.rotation.at[f + 1].set(qmul(w2quat(state.w[f]), state.rotation[f]))

    state = state._replace(position=position, rotation=rotation)
    return state


def get_state_kernel(self, f, controller):
    for j in range(3):
        controller[j] = self.position[f][j]
    for j in range(4):
        controller[j + self.dim] = self.rotation[f][j]


def set_state_kernel(f, controller, state: PrimitiveState):
    for j in range(3):
        state.position = state.position.at[f, j].set(controller[j])
    for j in range(4):
        state.rotation = state.rotation.at[f, j].set(controller[j + state.dim])
    return state


def set_velocity(n_substeps, state: PrimitiveState):
    # rewrite set velocity for different
    v, w = state.v, state.w
    # TODO check if can optimize for loop
    for j in range(0, n_substeps):
        v = v.at[j].set(state.action_buffer[:3] * state.action_scale[:3] / n_substeps)
        w = w.at[j].set(state.action_buffer[3:] * state.action_scale[3:] / n_substeps)

    state = state._replace(v=v, w=w)
    return state


def set_action(n_substeps, action, state: PrimitiveState):
    # set actions for n_substeps ...

    state = state._replace(action_buffer=action)
    state = set_velocity(n_substeps, state)
    return state


def position_control_batch(f, grid_pos, v_out, dt, state: PrimitiveState):
    dist = sdf_batch(f, grid_pos, state)

    control_mask = dist < state.size[0] * 1.5
    v_out = jnp.where(control_mask[..., None], state.v[f].reshape((1, 1, 1, 3)) / dt, v_out)

    # TODO Consider controlling position x?
    return v_out
