import sys

import cv2
import pexpect
import numpy as np

"""
Communication
"""


def cmd_interact(cmd, exp, ps):
    child = pexpect.spawn(cmd)
    child.logfile = sys.stdout
    child.expect(exp)
    child.sendline(ps)
    child.expect(pexpect.EOF)
    child.close()


"""
Observation
"""


def transform_xyz_to_base_link(xyz_array, tf_mat):
    # transform into base_link
    pcd = xyz_array.reshape(-1, 3)
    pcd = pcd.T
    calc_projection_const = np.ones((1, pcd.shape[1]))
    pcd = np.concatenate((pcd, calc_projection_const), axis=0)
    pcd = np.matmul(tf_mat, pcd).T[:, :3]
    return pcd


def add_goal_to_image(colormap, bounds, goal_pc_file):
    if goal_pc_file is None:
        return colormap
    goal_pc = np.load(goal_pc_file)
    goal_pc -= shift_vec[[0, 2, 1]][None, :]
    pos_x = (goal_pc[:, 0] - bounds[0, 0]) * colormap.shape[1] / (bounds[0, 1] - bounds[0, 0])
    pos_y = (goal_pc[:, 2] - bounds[1, 0]) * colormap.shape[0] / (bounds[1, 1] - bounds[1, 0])

    pos_x, pos_y = pos_x.astype(np.int32), pos_y.astype(np.int32)
    colormap[pos_y, pos_x] = np.array([0, 150, 0], dtype=np.uint8)

    return colormap


def get_heightmap(rgb_image, bounds, xyz_array, tf_mat, grid_size=0.003125, mask=None):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      rgb_image: HxWx3 RGB image.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      xyz_array: Pointcloud.
      tf_mat: Transformation between point cloud's frame and the world/base frame
      grid_size: size between two points
      mask: HxW binary image. 1 corresponds to the object.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
      pcmap: HxWx3 float array of points.
    """
    pcd = transform_xyz_to_base_link(xyz_array, tf_mat)
    points = np.reshape(pcd, rgb_image.shape)
    if mask is not None:
        points[mask==0] = [np.nan, np.nan, np.nan]

    colors = rgb_image

    # Width/Height here do not correspond to the width/heigh of the images
    # They correspond to coordinates in the base frame
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / grid_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / grid_size))

    # Init the arrays
    heightmap = np.zeros((height, width), dtype=np.float32)
    pcmap = np.zeros((height, width, 3), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[..., 0] >= bounds[0, 0]) & (points[..., 0] < bounds[0, 1])
    iy = (points[..., 1] >= bounds[1, 0]) & (points[..., 1] < bounds[1, 1])
    iz = (points[..., 2] >= bounds[2, 0]) & (points[..., 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / grid_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / grid_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    pcmap[py, px] = points
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap, pcmap


"""
Other functions
"""


def fill_pc(pc):
    # fill the body
    pc_filled = None
    step_size = 0.006
    for i in range(3):
        pc_cp = pc.copy()
        pc_cp[:, 2] -= step_size * i
        pc_filled = pc_cp if pc_filled is None else np.vstack((pc_filled, pc_cp))

    pc_filled = pc_filled[:, [0, 2, 1]]
    return pc_filled

def rm_nan(pc):
    # filter invalid point
    pc = pc[np.nan_to_num(pc.sum(-1)) != 0]

    return pc


def get_executable_actions(act, args):
    start_pos, end_pos = np.array(act['params']['pose0'][0]), np.array(act['params']['pose1'][0])

    print(start_pos, end_pos)

    # generate corresponding actions
    push_vec = end_pos - start_pos
    actions = [[push_vec[0] / args.horizon,
                0,
                push_vec[2] / args.horizon]] * args.horizon

    return start_pos, actions


"""
bean task methods
"""
# shift_vec = np.array([0.5, 0, 0.225])
shift_vec = np.array([0., 0, 0.])


def shift_pc_bean(pc):
    pc += shift_vec[None, :]

    return pc


def unshift_action(action):
    if type(action) is not dict:
        start, actions = action
        start = np.array(start)
        start[0] -= shift_vec[0]
        start[1] -= shift_vec[1]
        return start, actions

    start, end = np.array(action['params']['pose0'][0]), np.array(action['params']['pose1'][0])
    start[0] -= shift_vec[0]
    start[1] -= shift_vec[1]
    end[0] -= shift_vec[0]
    end[1] -= shift_vec[1]
    rot = [0, 0, 0, 1]

    action = {
        "camera_config": "",
        "params": {
            "pose0": (start, rot),
            "pose1": (end, rot)
        }
    }

    return action


def seg_img_bean(obs_color):
    seg = (obs_color.astype(np.float) / 255.0)
    seg = seg[..., 2] / seg[..., 1] > 1.5
    seg = seg.astype(np.float)

    bean_masks = np.ones_like(seg)
    # bean_masks[:20, :] = 0  # top
    bean_masks[460:, :] = 0  # bottom blocked by gripper
    # bean_masks[:, :80] = 0  # left
    # bean_masks[:240, :320] = 0  # top left
    # bean_masks[460:, :320] = 0  # bottom left
    bean_masks_img = bean_masks.astype(np.uint8)[..., None].repeat(3, -1) * 255
    bean_masks_img *= obs_color
    cv2.imwrite("tmp_files/bean_masks_img.png", bean_masks_img)

    seg *= bean_masks
    seg = seg.astype(np.bool)

    return seg


def measure_bow_size(pc):
    pc = rm_nan(pc)

    pc = pc[pc[:, 0] < 0.36]
    pc = pc[pc[:, 1] < 0.6]
    pc = pc[pc[:, 2] > 0.1]

    radius = pc[:, 1].max() - pc[:, 1].min()
    center = pc.mean(0)
    center[0] += 0.04

    return radius / 2, center, pc


"""
rope open task
"""


def seg_rope_open_barrier(obs_color):
    # obs_color bgr
    lower = np.array([22, 93, 0])
    upper = np.array([45, 255, 255])

    seg = (obs_color > lower).astype(np.int) * (obs_color < upper).astype(np.int)
    seg = seg.sum(-1) < 1

    obj_seg = seg_rope_open_object(obs_color)
    seg = seg.astype(np.int) * (1 - obj_seg.astype(np.int))

    return seg.astype(np.bool)


def seg_rope_open_object(obs_color):
    # obs_color bgr
    obs_color = obs_color.astype(np.float) / 255.0
    con1 = obs_color.sum(-1) > 2.7
    con2 = obs_color[:, :, 0] > (200. / 255)
    seg = con1.astype(np.int) * con2.astype(np.int)

    return seg.astype(np.bool)
