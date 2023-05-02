"""Robot client code for Pax hardware experiment.

Before running this script:
- The Pax hardware experiment server must be up
- The Kinova ROS driver must be up. (roslaunch launch_kortex_drivers kortex_driver.launch pc:=true)

This script repeats until goal reached or max_iter exceeded:
- Get sensor data form the robot; preprocess the sensor data into observation.
- Send observation to the server, and wait for inferred action.
- Exceute the inferred action.
"""
import argparse

import utils
from robot_interface import RobotNodeRope


def cmdline_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p = utils.argparse_common(p)
    p.add_argument(
        "-i", "--max_iter", type=int, default=10, help="Max number of iterations."
    )
    p.add_argument(
        "-v", "--verify", action="store_true", default=False, help="Verify the setup."
    )
    p.add_argument("-g", "--goal", type=str, default=None, help="Path to the goal file")

    return p.parse_args()


def is_obs_close_to_goal(obs, goal):
    return False


def verify_setup(robot):
    """Verify the perception and execution setup.\

    Perception: if the robot is in positioned to give good images.

    Execution: push height.
    """
    _ = robot.get_camera_sensor_data()
    if raw_input("is segmentation ok?") != "y":
        print("calibration failed.")
        return

    # push action test
    # height will be overwritten by robot
    act = {
        "camera_config": "",
        "primitive": "pick_place",
        "params": {
            "pose0": ([0.4, 0.29375, 0.024], [1, 0, 0, 0]),
            "pose1": ([0.40625, 0.19062500000000002, 0.024], [1, 0, 0, 0]),
        },
    }
    robot.step_straight_push(act)


def do_experiment(robot, mqtt_interface):
    """Perform the experiment."""
    it = 0
    while it < args.max_iter:
        camera_data = robot.get_camera_sensor_data()
        # if is_obs_close_to_goal(obs, goal):
        #     break
        msg = {"topic": args.obs_topic, "payload": camera_data.seg_heightmap}
        mqtt_interface.send_msg(msg)
        robot.move_to_prepare_pose(camera_data.seg_pcd)
        action = mqtt_interface.await_msg(args.action_topic)["payload"]
        robot.step_straight_push(action)
        user = raw_input("q to quit. Any other key to continue.")
        if user == "q":
            break


if __name__ == "__main__":
    args = cmdline_args()
    print(args)
    robot = RobotNodeRope()
    if args.verify:
        verify_setup(robot)
        exit(0)
    mqtt_interface = utils.create_mqtt_client(args.mqtt, [args.action_topic])
    do_experiment(robot, mqtt_interface)
