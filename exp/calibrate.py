"""Script to calibrate the shape rope environment."""
import argparse
import glob
import os
import pickle
import pathlib
import logging

import cv2
import numpy as np
import jax.numpy as jnp

import daxbench.core.engine.primitives.primitives as primitives
from daxbench.core.envs.shape_rope_env import ShapeRopeEnv, DefaultConf


def cmdline_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="./calibration_data",
        help="Directory to save the calibration data.",
    )
    p.add_argument(
        "-t",
        "--tmp_dir",
        type=str,
        default="./tmp",
        help="Directory to dump temporary files.",
    )
    return p.parse_args()


def get_color_map_from_particle_state(x, size=512):
    """x: Nx3 numpy array particle positions"""
    x = (x[:, [0, 2]] * size).astype(np.int32)
    x = np.array(x)
    idx_x, idx_y = x[:, 0], x[:, 1]

    colormap = np.zeros((size, size), dtype=np.float32)
    try:
        colormap[idx_y, idx_x] = 150
    except Exception as e:
        logging.warning("Particles seems to be out of scene. %s" % e)
    return colormap


def get_two_ends(x):
    x = np.array(x)
    x_lower_idx = np.argmin(x[:, 0])
    x_upper_idx = np.argmax(x[:, 0])
    p1 = x[x_lower_idx]
    p2 = x[x_upper_idx]
    center = (p1 + p2) / 2
    dp = p1 - p2
    length = np.linalg.norm(dp)
    angle = np.arctan2(dp[2], dp[0])
    return center, length, angle


def compare_file(file_path, tmp_dir: pathlib.Path, should_render=False):
    def draw_arrow(array, start_coord, end_coord):

        x_len, y_len = array.shape

        def _to_pixel(x, y):
            return int(x_len * x), int(y_len * y)

        ret = cv2.arrowedLine(
            array,
            _to_pixel(x_start, y_start),
            _to_pixel(x_end, y_end),
            color=255,
            thickness=2,
        )
        return ret

    primitives._sdf_batch = None
    print("=" * (45 + len(file_path)))
    print(f"======= compare sim and real on file {file_path} =======")
    print("=" * (45 + len(file_path)))
    file_name = file_path.split("/")[-1].split(".")[0]

    tmp_dir.mkdir(parents=True, exist_ok=True)
    sim_s_check = (tmp_dir / (file_name + "-s-check-sim.jpg")).as_posix()
    sim_s = (tmp_dir / (file_name + "-s0-sim.jpg")).as_posix()
    exp_s = (tmp_dir / (file_name + "-s0-exp.jpg")).as_posix()
    sim_sp = (tmp_dir / (file_name + "-s1-sim.jpg")).as_posix()
    exp_sp = (tmp_dir / (file_name + "-s1-exp.jpg")).as_posix()
    compare_all = (tmp_dir / ("compare-" + file_name + ".jpg")).as_posix()

    with open(file_path, "rb") as _file:
        calib_data = pickle.load(_file, encoding="latin-1")
    calib_s = calib_data[0]
    calib_s[:, [2, 1]] = calib_s[:, [1, 2]]
    calib_action = calib_data[1]
    (x_start, x_end) = (
        calib_action["params"]["pose0"][0][0],
        calib_action["params"]["pose1"][0][0],
    )
    (y_start, y_end) = (
        calib_action["params"]["pose0"][0][1],
        calib_action["params"]["pose1"][0][1],
    )
    calib_sp = calib_data[2]
    calib_sp[:, [2, 1]] = calib_sp[:, [1, 2]]

    center, length, angle = get_two_ends(calib_s)
    print("center:", center)
    print("angle: ", angle)

    conf = DefaultConf()
    conf.rope_init_pos[0] = center[0]
    conf.rope_init_pos[2] = center[2]
    conf.rope_z_rotation_angle = angle
    env = ShapeRopeEnv(10, 1, conf=conf)
    obs, state_reset = env.reset(env.simulator.key)
    cv2.imwrite(sim_s_check, get_color_map_from_particle_state(state_reset.x[0]))

    action_null = jnp.array([0.08, 0.001, 0.88, 0.12, 0.001, 0.92])
    actions_null = action_null[None, ...].repeat(1, axis=0)
    obs, _, _, info = env.step_diff(actions_null, state_reset)
    state0 = info["state"]
    sim_s_array = get_color_map_from_particle_state(state0.x[0])
    sim_s_array = draw_arrow(sim_s_array, (x_start, y_start), (x_end, y_end))
    cv2.imwrite(sim_s, sim_s_array)
    calib_s_array = get_color_map_from_particle_state(calib_s)
    calib_s_array = draw_arrow(calib_s_array, (x_start, y_start), (x_end, y_end))
    cv2.imwrite(exp_s, calib_s_array)
    calib_sp_array = get_color_map_from_particle_state(calib_sp)
    cv2.imwrite(exp_sp, calib_sp_array)
    print("action start: ", (x_start, y_start))
    print("action end: ", (x_end, y_end))

    action = jnp.array([x_start, 0, y_start, x_end, 0, y_end])
    actions = action[None, ...].repeat(1, axis=0)
    if should_render:
        obs, _, _, info = env.step_with_render(actions, state0)
        # obs, _, _, info = env.step_with_render(actions_null, info["state"])
    else:
        obs, _, _, info = env.step_diff(actions, state0)
        # obs, _, _, info = env.step_diff(actions_null, info["state"])
    sim_sp_array = get_color_map_from_particle_state(info["state"].x[0])
    cv2.imwrite(sim_sp, sim_sp_array)
    cv2.imwrite(
        compare_all,
        np.block([[sim_s_array, calib_s_array], [sim_sp_array, calib_sp_array]]),
    )


if __name__ == "__main__":
    args = cmdline_args()
    tmp_dir = pathlib.Path(args.tmp_dir)
    files = glob.glob(os.path.join(args.data_dir, "*.pkl"))
    print(files)
    # compare_file(files[6], tmp_dir, should_render=True)
    # compare_file(files[0], tmp_dir)
    for f in files:
        try:
            compare_file(f, tmp_dir)
        except Exception as e:
            print(e)
