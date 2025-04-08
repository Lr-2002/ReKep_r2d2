import time
import cv2 
import torch
import pysnooper
import gymnasium as gym
import numpy as np
import os
import datetime
import rekep.transform_utils as T

# import trimesh
# import open3d as o3d
import imageio
import json
from pathlib import Path
import mani_skill
# OG related
# import omnigibson as og
# from omnigibson.macros import gm
# from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_mesh_to_trimesh_mesh, mesh_prim_shape_to_trimesh_mesh
# from omnigibson.robots.fetch import Fetch
# from omnigibson.controllers import IsGraspingState


# from og_utils import OGCamera
# from omnigibson.robots.manipulation_robot import ManipulationRobot
# from omnigibson.controllers.controller_base import ControlType, BaseController


from rekep.utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_config,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)

# some customization to the OG functions
# def custom_clip_control(self, control):
#     """
#     Clips the inputted @control signal based on @control_limits.

#     Args:
#         control (Array[float]): control signal to clip

#     Returns:
#         Array[float]: Clipped control signal
#     """
#     clipped_control = control.clip(
#         self._control_limits[self.control_type][0][self.dof_idx],
#         self._control_limits[self.control_type][1][self.dof_idx],
#     )
#     idx = (
#         self._dof_has_limits[self.dof_idx]
#         if self.control_type == ControlType.POSITION
#         else [True] * self.control_dim
#     )
#     if len(control) > 1:
#         control[idx] = clipped_control[idx]
#     return control

# Fetch._initialize = ManipulationRobot._initialize
# BaseController.clip_control = custom_clip_control


class RobotController:
    def __init__(self):
        self.joint_limits = {
            "position": {
                "upper": [
                    2.8973,
                    1.7628,
                    2.8973,
                    -0.0698,
                    2.8973,
                    3.7525,
                    2.8973,
                ],  # Franka真实限制
                "lower": [
                    -2.8973,
                    -1.7628,
                    -2.8973,
                    -3.0718,
                    -2.8973,
                    -0.0175,
                    -2.8973,
                ],
            },
            "velocity": {
                "upper": [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                "lower": [
                    -2.1750,
                    -2.1750,
                    -2.1750,
                    -2.1750,
                    -2.6100,
                    -2.6100,
                    -2.6100,
                ],
            },
        }

        self.current_joint_angles = np.zeros(7)
        self.current_ee_pose = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.current_eef_position = np.array([0.5, 0.0, 0.5])
        self.world2robot_homo = np.eye(4)
        self.gripper_state = 0.0
        self.robot_state_path = Path("./robot_state.json")

    def get_relative_eef_position(self):
        """
        Mock version of get_relative_eef_position
        Returns:
            np.ndarray: [3,] array representing the end effector position in robot frame
        """
        return self.current_eef_position

    def get_relative_eef_orientation(self):
        """
        Mock version of get_relative_eef_orientation
        Returns:
            np.ndarray: [4,] array representing the end effector orientation as quaternion [w,x,y,z]
        """
        return np.array([1.0, 0.0, 0.0, 0.0])  # 默认朝向

    def clip_control(self, control, control_type="position"):
        return control
        # if control_type not in self.joint_limits:
        #     print(f"Warning: Unknown control type {control_type}")
        #     return control
        # upper = self.joint_limits[control_type]['upper']
        # lower = self.joint_limits[control_type]['lower']

        # n_joints = min(len(control), len(upper))
        # clipped = np.clip(control[:n_joints], lower[:n_joints], upper[:n_joints])
        # print(f"Clipping {control_type} control from {control} to {clipped}")
        # return clipped

    def send_command(self, command, control_type="position"):
        """
        Send a safe command to the robot

        Args:
            command: Control command
            control_type: Type of control
        """
        safe_command = self.clip_control(command, control_type)
        print(f"Sending {control_type} command: {safe_command}")
        # Here you would add your actual robot control code
        return safe_command


class ManiEnv:
    def __init__(self, config=None, verbose=False):
        self.video_cache = []
        # self.env = gym.make()
        self.env = gym.make(
            "Tabletop-Pick-Apple-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode="rgb+depth+segmentation",  # there is also "state_dict", "rgbd", ...
            control_mode="pd_joint_pos",  # there is also "", ...
            render_mode="rgb_array",
        )
        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])
        self.interpolate_pos_step_size = self.config["interpolate_pos_step_size"]
        self.interpolate_rot_step_size = self.config["interpolate_rot_step_size"]
        self.robot_state_path = Path("./robot_state.json")

        self.robot = RobotController()
        self.reset_joint_pos = np.array(
            [0.5, 0, 0.5, 1, 0, 0, 0]
        )  # Default home position
        self.gripper_state = 1.0  # 1.0 is open, 0.0 is closed
        self.world2robot_homo = np.eye(4)
        print("Robot interface initialized")
        # Initialize robot state
        obs, info = self.env.reset()
        obs = self.recursive_process_obs(obs)
        self.dt= 1/ 60 
        self.obs = obs
        self.update_robot_state(obs)
    def update_cameras_view(self):
        obs = self.env.unwrapped.render_all()
        obs = np.squeeze(obs.numpy())
        # print(f"\033[91m end-effector position is {self.get_ee_pos()}  \033[0m")
        bgr_obs_show = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("obs", bgr_obs_show)
        cv2.waitKey(int(self.dt * 1000))

    def recursive_process_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.cpu().numpy()[0]

        if isinstance(obs, dict):
            for key, value in obs.items():
                # print(key, value)
                obs[key] = self.recursive_process_obs(value)
        # else: 
        #     return self.recursive_process_obs(obs)
                # if isinstance(value, dict):
            #     obs[key] = self.recursive_process_obs(value)
            # else:
            #     continue
        # print(obs)
        # input()
        return obs

    @pysnooper.snoop()
    def update_robot_state(self, obs):
        """Read and update robot state from json file"""
        self.update_cameras_view()

        self.current_joint_angles = obs["agent"]["qpos"]
        # kk = obs.keys()
        self.ee_pose = obs["extra"]["tcp_pose"]
        self.ee_position = self.ee_pose[:3]
        self.ee_orientation = self.ee_pose[3:]
        self.gripper_state = self.current_joint_angles[-1]
        self.collision_status = False
        self.safety_status = {
            "collision_status": "false",
            "safety_status": "normal",
            "errors": [],
        }
        self.world2robot_homo = np.array(
            [
                [
                    0.08499406514372043,
                    -0.3632242726576058,
                    0.9278168658968742,
                    0.1800359259945526,
                ],
                [
                    -0.9956023482689502,
                    -0.06777719314130937,
                    0.06467005652724911,
                    -0.31374200777302785,
                ],
                [
                    0.039395088674820444,
                    -0.9292332214477907,
                    -0.36738759797530474,
                    0.4800809773815081,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return True

    def get_ee_pose(self, from_robot=False):
        """Get end-effector pose"""
        return self.ee_pose

    def get_ee_pos(self):
        """Get end-effector position"""
        return self.ee_pose[:3]

    def get_ee_quat(self):
        """Get end-effector orientation"""
        return self.ee_pose[3:]

    def compute_target_delta_ee(self, target_pose):
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)
        return pos_diff, rot_diff

    def get_arm_joint_positions(self):
        """
        Mock version of get_arm_joint_positions that returns simulated joint positions

        Returns:
            np.ndarray: Array of 7 joint positions for a 7-DOF arm
            [torso_lift, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex]
        """
        return self.current_joint_angles

    # TODO: future work, not used yet
    def execute_action(
        self,
        action,
        precise=True,
    ):
        """
        Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

        Args:
            action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
            precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
        Returns:
            tuple: A tuple containing the position and rotation errors after reaching the target pose.
        """
        if precise:
            pos_threshold = 0.03
            rot_threshold = 3.0
        else:
            pos_threshold = 0.10
            rot_threshold = 5.0
        action = np.array(action).copy()
        assert action.shape == (8,)
        target_pose = action[:7]
        gripper_action = action[7]

        # ======================================
        # = status and safety check
        # ======================================
        if np.any(target_pose[:3] < self.bounds_min) or np.any(
            target_pose[:3] > self.bounds_max
        ):
            print(
                f"{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}"
            )
            target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

        # ======================================
        # = interpolation
        # ======================================
        current_pose = self.get_ee_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
        rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
        pos_is_close = pos_diff < self.interpolate_pos_step_size
        rot_is_close = rot_diff < self.interpolate_rot_step_size
        if pos_is_close and rot_is_close:
            self.verbose and print(
                f"{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}"
            )
            pose_seq = np.array([target_pose])
        else:
            num_steps = get_linear_interpolation_steps(
                current_pose,
                target_pose,
                self.interpolate_pos_step_size,
                self.interpolate_rot_step_size,
            )
            pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
            self.verbose and print(
                f"{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}"
            )

        # ======================================
        # = move to target pose
        # ======================================
        # move faster for intermediate poses
        intermediate_pos_threshold = 0.10
        intermediate_rot_threshold = 5.0
        for pose in pose_seq[:-1]:
            self._move_to_waypoint(
                pose, intermediate_pos_threshold, intermediate_rot_threshold
            )
        # move to the final pose with required precision
        pose = pose_seq[-1]
        self._move_to_waypoint(
            pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40
        )
        # compute error
        pos_error, rot_error = self.compute_target_delta_ee(target_pose)
        self.verbose and print(
            f"\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n"
        )

        # ======================================
        # = apply gripper action
        # ======================================
        if gripper_action == self.get_gripper_open_action():
            self.open_gripper()
        elif gripper_action == self.get_gripper_close_action():
            self.close_gripper()
        elif gripper_action == self.get_gripper_null_action():
            pass
        else:
            raise ValueError(f"Invalid gripper action: {gripper_action}")

        return pos_error, rot_error

    def close_gripper(self):
        """Close gripper"""
        print("Closing gripper")
        self.gripper_state = 1.0  # transfer to AnyGrasp

    def open_gripper(self):
        """Open gripper"""
        print("Opening gripper")
        self.gripper_state = 0.0

    def get_gripper_open_action(self):
        return -1.0

    def get_gripper_close_action(self):
        return 1.0

    def get_gripper_null_action(self):
        return 0.0

    def is_grasping(self, candidate_obj=None):
        """Check if gripper is grasping"""
        # Could be enhanced with force sensor readings
        # TODO, how to modify this
        # print("Yes it is grasping")
        return self.env.unwrapped.is_grasping(candidate_obj).cpu().numpy()[0]

    def get_collision_points(self, noise=True):
        """Get collision points of gripper"""
        # Return mock collision points
        return np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4]])

    def get_keypoint_positions(self):
        return self.keypoints

    def get_object_by_keypoint(self, keypoint_idx):
        # TODO: asscociate keypoints with closest object (mask?)
        return None

    def register_keypoints(self, keypoints):
        """Register keypoints to track"""
        print(f"Registering keypoints number: {len(keypoints)}")
        self.keypoints = keypoints

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """Get signed distance field"""
        print("Getting SDF voxels (mock data)")
        # Return mock SDF grid
        return np.zeros((10, 10, 10))

    def get_cam_obs(self):
        """Get camera observations"""
        print("Getting camera observations")
        # Return mock RGB-D data
        return {"rgb": np.zeros((480, 640, 3)), "depth": np.zeros((480, 640))}

    def reset(self):
        """Reset robot to home position"""
        print("Resetting robot to home position")
        self.ee_pose = [0.5, 0, 0.5, 1, 0, 0, 0]
        self.open_gripper()

    def sleep(self, seconds):
        """Wait for specified duration"""
        print(f"Sleeping for {seconds} seconds")
        time.sleep(seconds)

    def save_video(self, save_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), "videos")
        os.makedirs(save_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(
                save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4'
            )
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(rgb)
        video_writer.close()
        return save_path

    # ======================================
    # = internal functions
    # ======================================
    def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
        """
        this is supposed to be for true ee pose (franka hand) in robot frame
        """  # TODO transform to robot frame
        current_pos = self.get_ee_pose()[:3]
        current_xyzw = self.get_ee_pose()[3:7]
        current_rotmat = T.quat2mat(current_xyzw)
        target_rotmat = T.quat2mat(target_xyzw)
        # calculate position delta
        pos_diff = (target_pos - current_pos).flatten()
        pos_error = np.linalg.norm(pos_diff)
        # calculate rotation delta
        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
        # print status
        self.verbose and print(
            f"{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}"
        )
        self.verbose and print(
            f"{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}"
        )
        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
            self.verbose and print(
                f"{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}"
            )
            return True, pos_error, rot_error
        return False, pos_error, rot_error

    def _move_to_waypoint(
        self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10
    ):
        pos_errors = []
        rot_errors = []
        count = 0
        while count < max_steps:
            reached, pos_error, rot_error = self._check_reached_ee(
                target_pose_world[:3],
                target_pose_world[3:7],
                pos_threshold,
                rot_threshold,
            )
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                break
            # convert world pose to robot pose
            target_pose_robot = np.dot(
                self.world2robot_homo, T.convert_pose_quat2mat(target_pose_world)
            )

            # Franka的action空间: [7个关节角度 + 1个夹爪值]
            action = np.zeros(8)
            # 前7维是关节角度 - 这里需要通过逆运动学(IK)计算
            joint_angles = self.compute_ik(target_pose_robot)  # 需要实现IK
            action[:7] = joint_angles
            # 最后1维是夹爪
            action[7] = self.gripper_state

            # 执行动作
            self._step(action=action)
            count += 1

        if count == max_steps:
            print(
                f"Pose not reached after {max_steps} steps (pos_error: {pos_errors[-1]:.4f}, rot_error: {np.rad2deg(rot_errors[-1]):.4f})"
            )

    def compute_ik(self, target_pose):
        """Mock IK solver - 在实际应用中需要替换为真实的IK"""
        # 这里返回一个合理范围内的关节角度
        return np.zeros(7)  # 7个关节的角度

    def _step(self, action):
        """执行一步动作"""
        # 模拟执行动作，更新机器人状态
        # joint_angles = action[:7]
        # gripper_action = action[7]

        # 更新状态
        obs, done, trunc, reward, info = self.env.step(action)
        self.current_joint_angles = joint_angles
        self.gripper_state = gripper_action
        # 通过正运动学更新末端执行器位置（在实际应用中需要从真实机器人读取）
        # self.current_eef_position = self.compute_fk(joint_angles)  # 需要实现FK

if __name__ == "__main__":
    env = R2D2Env(get_config("../configs/config.yaml")["env"])
    env.is_grasping("apple")


