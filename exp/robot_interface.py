import argparse
import copy
import cv2
import logging
import numpy as np
import os
import time
from subprocess import check_output
from collections import namedtuple

import ros_numpy
import rospy
import tf.transformations as T
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from kortex_robot.kortex_robot import KortexRobot
import kortex_api
from sensor_msgs.msg import Image, PointCloud2

from robot_utils import (
    get_heightmap,
    transform_xyz_to_base_link,
    rm_nan,
    add_goal_to_image,
)


def take_rgb_img(image):
    global rgb_image
    rgb_image = CvBridge().imgmsg_to_cv2(image, desired_encoding="bgr8")


def take_depth_img(image):
    global depth_image
    depth_image = CvBridge().imgmsg_to_cv2(image, desired_encoding="32FC1")


def take_point_cloud(ros_point_cloud):
    global xyz_array
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
        ros_point_cloud, remove_nans=False
    )


def get_pid(name):
    return check_output(["pidof", name])


CameraSensorData = namedtuple(
    "CameraSensorData",
    "colormap obs_with_goal heightmap seg_pcd seg_heightmap seg_colormap",
)


class RobotNode:
    def __init__(self, goal_file_path=None, dump_dir="./tmp", velocity=0.5):

        print("pid", get_pid("python"))
        self._goal_file_path = goal_file_path
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        self._dump_dir = dump_dir

        rospy.init_node("KortexRobot")
        rospy.Subscriber("/kortex/camera/color/image_rect_color", Image, take_rgb_img)
        rospy.Subscriber("/kortex/camera/color/image_rect_depth", Image, take_depth_img)
        rospy.Subscriber(
            "/kortex/camera/depth_registered/points",
            PointCloud2,
            take_point_cloud,
            queue_size=1,
            buff_size=52428800,
        )
        self.robot = KortexRobot(use_ros=False)

        self.push_height = 0.025
        self.ob_pos = [
            1.6609188965056685,
            0.06235299423737792,
            -2.4447499227596596,
            1.0054323942433756,
            0.1138276578568115,
            1.870451084750923,
            -2.3186958504634845,
        ]
        self.tf_mat = np.array(
            [
                [0.00332604, -0.96407175, 0.26562114, 0.28],
                [-0.98515835, 0.04242628, 0.16632206, 0.167],
                [-0.17161572, -0.26223208, -0.94962223, 0.744],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.bounds = np.array([[0, 1.0], [0, 1.0], [-0.2, 0.3]])
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        logging.debug("Starting robot node")

        self.init_robot(velocity)

    def init_robot(self, velocity_factor):
        self.robot._arm.arm_group.set_max_velocity_scaling_factor(velocity_factor)
        self.robot._arm.velocity_scale_factor = velocity_factor
        self.robot._arm.arm_group.set_planner_id("RRTstar")
        self.robot.arm_move_to_joint(self.ob_pos)
        self.robot.gripper_position(self.robot.GRIPPER_OPENED_POS)

    def move_to_prepare_pose(self, seg_pcd):
        self.robot.gripper_position(self.robot.GRIPPER_CLOSED_POS)
        position = seg_pcd.mean(axis=0)
        position[2] += 0.08

        arm_target_pose = PoseStamped()
        arm_target_pose.header.frame_id = "base"
        arm_target_pose.pose.position.x = position[0]
        arm_target_pose.pose.position.y = position[1]
        arm_target_pose.pose.position.z = position[2]
        quat = T.quaternion_from_euler(0, np.radians(180), np.radians(270))
        arm_target_pose.pose.orientation.x = quat[0]
        arm_target_pose.pose.orientation.y = quat[1]
        arm_target_pose.pose.orientation.z = quat[2]
        arm_target_pose.pose.orientation.w = quat[3]

        self.robot.arm_move_to_pose(
            arm_target_pose, arm="left", allowed_planning_time=2
        )

    def step_straight_push(self, act):
        self.robot.gripper_position(self.robot.GRIPPER_CLOSED_POS)

        start_pos, end_pos = np.array(act["params"]["pose0"][0]), np.array(
            act["params"]["pose1"][0]
        )
        start_pos[2] = self.push_height
        end_pos[2] = self.push_height

        print("step_straight_push start_pos", start_pos)

        # move to start pos
        arm_target_pose = PoseStamped()
        arm_target_pose.header.frame_id = "base_link"
        arm_target_pose.pose.position.x = start_pos[0]
        arm_target_pose.pose.position.y = start_pos[1]
        arm_target_pose.pose.position.z = start_pos[2] + 0.03
        quat = T.quaternion_from_euler(0, np.radians(180), np.radians(270))
        arm_target_pose.pose.orientation.x = quat[0]
        arm_target_pose.pose.orientation.y = quat[1]
        arm_target_pose.pose.orientation.z = quat[2]
        arm_target_pose.pose.orientation.w = quat[3]

        arm_move_to_pose_res = self.robot.arm_move_to_pose(
            arm_target_pose, arm="left", allowed_planning_time=2
        )
        if arm_move_to_pose_res:
            self.robot.arm_move_in_cartesian(dx=0, dy=0, dz=-0.03, arm="left")
            action = end_pos - start_pos
            self.robot.arm_move_in_cartesian(
                dx=action[0], dy=action[1], dz=0, arm="left"
            )
            self.robot.arm_move_in_cartesian(dx=0, dy=0, dz=0.03, arm="left")

        self.robot.arm_move_to_joint(self.ob_pos)
        self.robot.gripper_position(self.robot.GRIPPER_OPENED_POS)
        return arm_move_to_pose_res

    def seg_object(self, obs_color):
        raise NotImplementedError

    def get_camera_sensor_data(self):
        obs_color = rgb_image.astype(np.uint8)
        object_seg_mask = self.seg_object(obs_color)

        heightmap, colormap, pcmap = get_heightmap(
            rgb_image, self.bounds, xyz_array, self.tf_mat
        )
        seg_heightmap, seg_colormap, _ = get_heightmap(
            rgb_image, self.bounds, xyz_array, self.tf_mat, mask=object_seg_mask
        )
        obs_with_goal = add_goal_to_image(
            copy.copy(colormap), self.bounds, self._goal_file_path
        )

        # save color img
        colormap = colormap.astype(np.uint8)
        seg_colormap = seg_colormap.astype(np.uint8)

        # visualize
        cv2.imwrite(os.path.join(self._dump_dir, "colormap.jpg"), colormap)
        cv2.imwrite(os.path.join(self._dump_dir, "heightmap.jpg"), heightmap)
        cv2.imwrite(os.path.join(self._dump_dir, "seg_colormap.jpg"), seg_colormap)
        cv2.imwrite(os.path.join(self._dump_dir, "seg_heightmap.jpg"), seg_heightmap)
        cv2.imwrite(os.path.join(self._dump_dir, "obs_with_goal.jpg"), obs_with_goal)
        cv2.imwrite(
            os.path.join(self._dump_dir, "obj_seg.jpg"),
            (object_seg_mask * 255).astype(np.uint8),
        )

        # use the seg_image to crop pc
        pcd = transform_xyz_to_base_link(xyz_array, self.tf_mat)
        pcd = pcd.reshape(obs_color.shape)
        seg_pcd = pcd[object_seg_mask]
        seg_pcd = rm_nan(seg_pcd)
        seg_pcd = seg_pcd[seg_pcd[..., 1] > 0]
        seg_pcd = seg_pcd[seg_pcd[..., 1] < 0.8]
        seg_pcd = seg_pcd[seg_pcd[..., 0] > 0]
        seg_pcd = seg_pcd[seg_pcd[..., 0] < 0.8]
        seg_pcd = rm_nan(seg_pcd)

        # visualize depth
        return CameraSensorData(
            colormap, obs_with_goal, heightmap, seg_pcd, seg_heightmap, seg_colormap
        )


class RobotNodeRope(RobotNode):
    def seg_object(self, obs_color):
        """Return a 2D binary mask of the object. 1 corresponds to the object."""
        # obs_color bgr
        obs_color = obs_color.astype(np.float) / 255.0
        con1 = obs_color.sum(-1) > 2.8
        con2 = obs_color[:, :, 0] > (230.0 / 255)
        seg = con1.astype(np.int) * con2.astype(np.int)
        seg[:, -30:] = 0

        return seg.astype(np.bool)
