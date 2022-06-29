import os.path

import cv2
import jax.numpy as jnp
import numpy as np


def get_projection(x_grid, cloth_mask, size=64):
    x = x_grid

    x = jnp.clip(x, 0, 1)
    x = x[:, cloth_mask.astype(jnp.bool_)]
    x = (x[:, :, [0, 2]] * size).astype(jnp.int32)
    x = np.array(x)
    idx_x, idx_y = x[:, :, 0], x[:, :, 1]

    colormap = np.zeros((x_grid.shape[0], size, size), dtype=np.float32)
    colormap[:, idx_y, idx_x] = 1.
    return colormap


def get_goal_image(x, cloth_mask):
    goal_image = get_projection(x, cloth_mask, size=jnp.array(512))
    goal_image = goal_image[0]

    canvas = np.zeros(goal_image.shape)
    goal_image = np.array([goal_image, canvas, canvas])
    goal_image = goal_image.transpose(1, 2, 0)
    return goal_image


def show_goal(x, cloth_mask):
    colormap = get_goal_image(x, cloth_mask)

    cv2.namedWindow("goal")
    cv2.imshow("goal", colormap)
    cv2.waitKey(10)


def get_expert_start_end_cloth(x_grid, cloth_mask, goal_map=None):
    colormap = get_projection(x_grid, cloth_mask, size=jnp.array(512))
    if goal_map is not None:
        colormap += goal_map * 0.3

    return get_expert_start_end(colormap)


def get_expert_start_end_mpm(x, size=64, goal_map=None):
    x = (x[:, :, [0, 2]] * size).astype(jnp.int32)
    x = np.array(x)
    idx_x, idx_y = x[:, :, 0], x[:, :, 1]

    colormap = np.zeros((x.shape[0], size, size), dtype=np.float32)
    colormap[:, idx_y, idx_x] = 1.

    if goal_map is not None:
        colormap += goal_map * 0.5
    return get_expert_start_end(colormap)


def get_expert_start_end(colormap):
    batch_size = colormap.shape[0]
    colormap = colormap[0]
    # collect actions
    mouse_clicks = []
    bounds = np.array([[0, 1], [0, 1], [-0.2, 0.3]])

    def draw_circle(event, x, y, flags, colormap):

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(colormap, (x, y), 3, (255, 0, 0), 2)
            mouse_clicks.append((x * 1.0, y * 1.0))

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle, colormap)
    cv2.imshow("image", colormap)
    cv2.waitKey(10)
    while True:
        # both windows are displaying the same img
        cv2.imshow("image", colormap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(mouse_clicks) > 0:
            cv2.imshow("image", colormap)
            cv2.waitKey(10)
            break

    mouseX, mouseY = mouse_clicks[0]
    x_start = mouseX / colormap.shape[1] * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    y_start = mouseY / colormap.shape[0] * (bounds[1, 1] - bounds[1, 0]) + bounds[1, 0]

    while True:
        # both windows are displaying the same img
        cv2.imshow("image", colormap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if len(mouse_clicks) > 1:
            cv2.imshow("image", colormap)
            cv2.waitKey(10)
            break

    mouseX, mouseY = mouse_clicks[1]
    x_end = mouseX / colormap.shape[1] * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    y_end = mouseY / colormap.shape[0] * (bounds[1, 1] - bounds[1, 0]) + bounds[1, 0]

    print("demo act", x_start, y_start, x_end, y_end)

    actions = jnp.array([x_start, 0, y_start, x_end, 0, y_end])
    actions = actions[None, ...].repeat(batch_size, axis=0)
    return actions


def get_pnp_actions_mpm(actions, state, repeat=20):
    action = actions[-1]
    shift, rotation = action[:3], action[3:]
    # TODO: account for rotation
    # primitive_pos = state[12]

    act_move = shift
    act_move = act_move[None, ...].repeat(repeat, axis=0)
    sub_actions = act_move

    dummy_actions = jnp.zeros_like(sub_actions)
    sub_actions = jnp.concatenate([sub_actions, dummy_actions], axis=1)
    return sub_actions


def calc_iou(x, cloth_mask, goal_path):
    color_map = get_projection(x, cloth_mask)
    goal_map = np.load(goal_path)
    goal_map = jnp.array(goal_map)

    # compute IoU
    iou = jnp.sum(color_map * goal_map) / (jnp.sum(color_map) + jnp.sum(goal_map) - jnp.sum(color_map * goal_map))
    return iou


def calc_chamfer(x, y, metric='l2', direction='bi'):
    y = y[None, ...].repeat(x.shape[0], axis=0)
    pred = x[..., None, :].repeat(y.shape[1], -2)
    pred_demo = y[..., None, :, :].repeat(x.shape[1], -3)
    pred_dis = jnp.sqrt(((pred - pred_demo) ** 2).mean(-1))
    x2y_min = pred_dis.min(-1).mean(1)

    # for every state in demo_traj_ find closest state in rollout_traj
    demo = y[..., None, :].repeat(x.shape[1], -2)
    demo_pred = x[..., None, :, :].repeat(y.shape[1], -3)
    demo_dis = jnp.sqrt(((demo - demo_pred) ** 2).mean(-1))
    y2x_min = demo_dis.min(-1).mean(1)

    chamfer_dist = y2x_min + x2y_min

    return chamfer_dist


def calc_l2(x, y):
    y = y[None, ...].repeat(x.shape[0], axis=0)
    l2 = jnp.sqrt(((x - y) ** 2).mean(-1)).mean(-1)
    return l2
