from open3d import data

import open3d as o3d
import torch
import numpy as np
import json
import os
import pdb

import argparse

# from rekep.environment import R2D2Env
from rekep.maniskill_environment import ManiEnv
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.ik_solver import FrankaIKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
import rekep.transform_utils as T

from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

"""
metadata.json
{
    "init_keypoint_positions": [
        [-0.1457058783982955, -0.47766187961876, 0.98],
        [-0.0144477656708159, 0.012521396914707113, 0.745],
        [0.14099338570298237, 0.5722672713826932, 1.283],
        [0.2693722882157947, -0.3018593983523729, 1.047],
        [0.43524427390119413, -0.04595746991503292, 0.6970000000000001]
    ],
    "num_keypoints": 5,
    "num_stages": 4,
    "grasp_keypoints": [1, -1, 2, -1],
    "release_keypoints": [-1, 1, -1, 2]
}
"""
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")

import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute"
        )
        return result

    return wrapper


class MSCamera:
    """
    Defines the camera class
    """

    def __init__(self, obs, name) -> None:
        self.name = name
        # self.name = 'left_camera'
        # self.cam = insert_camera(name=config['name'], og_env=og_env, width=config['resolution'], height=config['resolution'])
        # self.cam.set_position_orientation(config['position'], config['orientation'])
        # self.data = obs['sensor_data'][self.name]
        self.obs = obs
        self.param = obs["sensor_param"][self.name]
        # breakpoint()

        self.intrinsics = self.param["intrinsic_cv"]
        self.extrinsics = self.param["cam2world_gl"]

    def pixel_to_3d_points(self, depth_image, intrinsics, extrinsics):
        # breakpoint()
        new_pcd, new_pcd_colors, new_pcd_mask = self.get_3d_point_cloud()
        # new_pcd = new_pcd[(new_pcd[:, 2] >= -0.4 )& (new_pcd[:, 2] <= 1.4) & (np.abs(new_pcd[:,1])<1)] # TODO importent
        # self.draw_pc(new_pcd)
        return new_pcd

    def get_params(self):
        """
        Get the intrinsic and extrinsic parameters of the camera
        """
        return {"intrinsics": self.intrinsics, "extrinsics": self.extrinsics}

    def get_obs(self, obs):
        """
        Gets the image observation from the camera.
        Assumes have rendered befor calling this function.
        No semantic handling here for now.
        # obs = self.cam.get_obs()
        """
        ret = {}
        ret["rgb"] = obs[0]["rgb"][:, :, :3]  # H, W, 3
        ret["depth"] = obs[0]["depth_linear"]  # H, W
        ret["points"] = self.pixel_to_3d_points(
            ret["depth"], self.intrinsics, self.extrinsics
        )  # H, W, 3
        ret["seg"] = obs[0]["seg_semantic"]  # H, W
        ret["intrinsic"] = self.intrinsics
        ret["extrinsic"] = self.extrinsics
        return ret

    def get_3d_point_cloud(self):
        obs = self.obs
        depth = np.squeeze(obs["sensor_data"][self.name]["depth"]) / 1000.0
        rgb = np.squeeze(obs["sensor_data"][self.name]["rgb"])
        param = obs["sensor_param"][self.name]
        mask = np.squeeze(obs["sensor_data"][self.name]["segmentation"])

        intrinsic_matrix = np.squeeze(param["intrinsic_cv"])

        # points_mask_colors = self.depth_to_pc(intrinsic_matrix, depth, rgb, mask)
        points_colors = self.get_color_points(rgb, depth, param, mask)

        points = points_colors[:, :3]
        points_mask = mask.reshape((-1, 1))
        points_colors = points_colors[:, 3:]

        return points, points_colors, points_mask

    """
    Description:
        Use camera intrisic matrix and depth map to project 2D image into 3D point cloud
    Args:
        rgb: 2D RGB image. If rgb is not None, return colored point cloud.
        mask: 2D mask. If mask is not None, return masked point cloud.
    Returns:
        points: 1-3 cols are 3D point cloud, 4 col is mask, 5-7 cols are rgb.
    """

    def depth_to_pc(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth

        points = np.vstack((x.flatten(), y.flatten(), z.flatten(), mask.flatten())).T

        flatten_rgb = rgb.reshape((-1, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    def get_color_points(self, rgb, depth, cam_param, mask=None):
        intrinsic_matrix = np.squeeze(cam_param["intrinsic_cv"])

        color_points = self.depth_to_point_cloud(intrinsic_matrix, depth, rgb, mask)
        # if self.frame == "base":
        #     return color_points
        cam2world_gl = np.squeeze(cam_param["cam2world_gl"])
        color_points[:, :3] = self.transform_camera_to_world(
            color_points[:, :3], cam2world_gl
        )
        return color_points

    def depth_to_point_cloud(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        flatten_rgb = rgb.reshape((H * W, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    """
    Description: 
        transform point cloud from camera coordinicate to world coordinate
    """

    def transform_camera_to_world(self, points, extri_mat):
        R = extri_mat[:3, :3]
        t = extri_mat[:3, 3]
        pcd_world = (R @ points.T).T - t
        rotation_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        pcd_world = (rotation_y_180 @ pcd_world.T).T
        return pcd_world


@timer_decorator
class MainR2D2:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config["main"]
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed(self.config["seed"])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config["keypoint_proposer"])
        self.constraint_generator = ConstraintGenerator(
            global_config["constraint_generator"]
        )

        self.env = ManiEnv(global_config["env"])
        # self.env =
        ik_solver = FrankaIKSolver(
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(
            global_config["subgoal_solver"], ik_solver, self.env.reset_joint_pos
        )
        self.path_solver = PathSolver(
            global_config["path_solver"], ik_solver, self.env.reset_joint_pos
        )

    def load_rgbd_data(self):
        obs = self.env.obs
        # breakpoint()
        self.came_name = "left_camera"
        self.cam = MSCamera(obs, self.came_name)
        data = obs["sensor_data"][self.came_name]
        rgb = data["rgb"]
        depth = data["depth"] / 1000
        points = self.cam.pixel_to_3d_points(
            depth.squeeze(-1), self.cam.intrinsics, self.cam.extrinsics
        )
        mask = data["segmentation"]
        self.mask = mask
        print(rgb)
        # print(data.keys())
        # if folder is None:
        #     folder = 'data/rgbd/cloth-hack-1'
        # if frame is None:
        #     frame = 10
        # rgb = np.load(f'{folder}/color_{frame:06d}.npy')
        # points = np.load(f'{folder}/depth_{frame:06d}.npy')
        # mask = np.load(f'{folder}/mask_{frame:06d}.npy')
        # print(f"Debug: Loaded RGB image with shape: {rgb.shape}")
        # print(f"Debug: Loaded point cloud with shape: {points.shape}")
        # print(f"Debug: Loaded mask with shape: {mask.shape}")
        breakpoint()
        return rgb, points, mask

    @timer_decorator
    def perform_task(self, instruction, rekep_program_dir=None):
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            rgb, points, mask = self.load_rgbd_data()
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(
                rgb, points, mask
            )
            print(
                f"{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}"
            )
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {
                "init_keypoint_positions": keypoints,
                "num_keypoints": len(keypoints),
            }
            rekep_program_dir = self.constraint_generator.generate(
                projected_img, instruction, metadata
            )
            print(f"{bcolors.HEADER}Constraints generated{bcolors.ENDC}")
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir)

    def _execute(self, rekep_program_dir):
        # load metadata
        # pdb.set_trace()
        with open(os.path.join(rekep_program_dir, "metadata.json"), "r") as f:
            self.program_info = json.load(f)
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info["init_keypoint_positions"])
        # TODO : no update? world frame reset?
        # load constraints
        self.constraint_fns = dict()
        for stage in range(
            1, self.program_info["num_stages"] + 1
        ):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ["subgoal", "path"]:
                load_path = os.path.join(
                    rekep_program_dir, f"stage{stage}_{constraint_type}_constraints.txt"
                )
                get_grasping_cost_fn = get_callable_grasping_cost_fn(
                    self.env
                )  # special grasping function for VLM to call
                stage_dict[constraint_type] = (
                    load_functions_from_txt(load_path, get_grasping_cost_fn)
                    if os.path.exists(load_path)
                    else []
                )
            self.constraint_fns[stage] = stage_dict

        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(
            self.program_info["num_keypoints"] + 1, dtype=bool
        )
        self.keypoint_movable_mask[0] = (
            True  # first keypoint is always the ee, so it's movable
        )

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            # pdb.set_trace()

            scene_keypoints = self.env.get_keypoint_positions()

            self.keypoints = np.concatenate(
                [[self.env.get_ee_pos()], scene_keypoints], axis=0
            )  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_positions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config["sdf_voxel_size"])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]["path"]
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config["constraint_tolerance"]:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]["path"]
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config["constraint_tolerance"]:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:
                        break
                print(
                    f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}"
                )
                self._update_stage(new_stage)
            else:
                # ====================================
                # = get optimized plan
                # ====================================
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(
                    next_subgoal, from_scratch=self.first_iter
                )
                self.first_iter = False
                self.action_queue = next_path.tolist()
                # self.last_sim_step_counter = self.env.step_counter
                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while (
                    len(self.action_queue) > 0
                    and count < self.config["action_steps_per_iter"]
                ):
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info["num_stages"]:
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(
                            f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}"
                        )
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        # pdb.set_trace()
        subgoal_constraints = self.constraint_fns[self.stage]["subgoal"]
        path_constraints = self.constraint_fns[self.stage]["path"]
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array(
                [-self.config["grasp_depth"] / 2.0, 0, 0]
            )
        debug_dict["stage"] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        # pdb.set_trace()
        path_constraints = self.constraint_fns[self.stage]["path"]
        path, debug_dict = self.path_solver.solve(
            self.curr_ee_pose,
            next_subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=from_scratch,
        )
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    # TODO: check action sequence
    @timer_decorator
    def _process_path(self, path):
        # pdb.set_trace()
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate(
            [
                self.curr_ee_pose.reshape(1, -1),
                path,
            ],
            axis=0,
        )
        num_steps = get_linear_interpolation_steps(
            full_control_points[0],
            full_control_points[-1],
            self.config["interpolate_pos_step_size"],
            self.config["interpolate_rot_step_size"],
        )
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # pdb.set_trace()
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info["grasp_keypoints"][self.stage - 1] != -1
        self.is_release_stage = (
            self.program_info["release_keypoints"][self.stage - 1] != -1
        )
        # can only be grasp stage or release stage or none
        assert (
            self.is_grasp_stage + self.is_release_stage <= 1
        ), "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(
            1, len(self.keypoint_movable_mask)
        ):  # first keypoint is ee so always movable
            print(i)
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            print(keypoint_object)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        # pdb.set_trace()
        print("Grasp action")

    def _execute_release_action(self):
        print("Release action")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruction", type=str, required=True, help="Instruction for the task"
    )
    parser.add_argument(
        "--rekep_program_dir",
        type=str,
        required=False,
        help="keypoint constrain proposed folder",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)',
    )
    args = parser.parse_args()

    main = MainR2D2(visualize=args.visualize)
    main.perform_task(
        instruction=args.instruction, rekep_program_dir=args.rekep_program_dir
    )
