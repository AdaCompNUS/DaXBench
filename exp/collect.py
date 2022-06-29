"""Script for pax push rope environment calibration.

Before running this script:
- The Kinova ROS driver must be up.

Save (s, a, s') tuple into pickled file.

state: robot_interface.CameraSensorData
"""
import os
import argparse
import pickle
import random
import string

from robot_interface import RobotNodeRope


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def cmdline_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-d",
        "--save_dir",
        type=str,
        default="./calibration_data",
        help="Directory to save the calibration data.",
    )
    p.add_argument("-f", "--file_name", type=str, default=None, help="Name of file.")
    return p.parse_args()


if __name__ == "__main__":
    args = cmdline_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    robot = RobotNodeRope()

    file_name = get_random_string(10) if args.file_name is None else args.file_name
    file_path = os.path.join(args.save_dir, file_name) + ".pkl"

    s = robot.get_camera_sensor_data().seg_pcd
    should_quit = raw_input("q to quit. Any other key to continue") == "q"
    if should_quit:
        exit(0)

    # push action test
    # height will be overwritten by robot
    act = {
        "camera_config": "",
        "primitive": "pick_place",
        "params": {
            "pose0": ([0.4, 0.3, 0.024], [1, 0, 0, 0]),
            "pose1": ([0.5, 0.3, 0.024], [1, 0, 0, 0]),
        },
    }
    robot.step_straight_push(act)

    s_prime = robot.get_camera_sensor_data().seg_pcd
    should_quit = raw_input("q to quit. Any other key to continue") == "q"
    if should_quit:
        exit(0)

    with open(file_path, "w") as _file:
        pickle.dump((s, act, s_prime), _file, protocol=2)
