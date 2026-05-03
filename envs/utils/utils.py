import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
import random
from typing import Union, TYPE_CHECKING
from isaaclab_tasks.direct.InteractionTracking.utils.math import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward_utils import *
from isaaclab_tasks.direct.InteractionTracking.utils.func import *
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.InteractionTracking.motions.motion_loader import MotionLoader



def write_ref_state(robot: "Articulation", motion_loader: "MotionLoader", frame_indexes: torch.Tensor, env_origins: torch.Tensor, env_ids: torch.Tensor | None=None):
    if env_ids is None: env_ids = robot._ALL_INDICES

    root_pos_quat = motion_loader.root_states[frame_indexes, :7]
    root_pos_quat[:, :3] += env_origins

    robot.write_root_link_pose_to_sim(root_pos_quat, env_ids)
    robot.write_root_com_velocity_to_sim(motion_loader.root_states[frame_indexes, 7:], env_ids)
    
    # what is the difference between the two lines below?
    # env.robot.write_root_pose_to_sim(env.ref_state_buffer['root_state'][:, env.ref_state_buffer_index, :7], env_ids)
    # env.robot.write_root_state_to_sim(env.ref_state_buffer['root_state'][:, env.ref_state_buffer_index], env_ids)
    
    robot.write_joint_state_to_sim(motion_loader.dof_positions[frame_indexes],
                                    motion_loader.dof_velocities[frame_indexes],
                                    None, env_ids)

# to delete:
def get_robot_state(robot: "Articulation", env_ids: torch.Tensor | None=None) -> dict:
    if env_ids is None: env_ids = robot._ALL_INDICES
    return {
        "body_pos": robot.data.body_pos_w[env_ids],
        "body_rot": robot.data.body_quat_w[env_ids],
        "body_lin_vel": robot.data.body_lin_vel_w[env_ids],
        "body_ang_vel": robot.data.body_ang_vel_w[env_ids],
        "dof_pos": robot.data.joint_pos[env_ids],
        "dof_vel": robot.data.joint_vel[env_ids],
        "root_state": robot.data.root_state_w[env_ids],
        "root_height": robot.data.root_state_w[env_ids, 2:3],
    }

# to delete:
def sample_motion_times(env_ids: torch.Tensor, motion_loader: "MotionLoader", reset_strategy: str, motion_ids: torch.Tensor, bias: int=1) -> torch.Tensor:
    num_samples = env_ids.shape[0]
    sampled_local_times = motion_loader.sample_times(num_samples, motion_ids, bias)

    if "start" in reset_strategy:
        sampled_local_times = torch.zeros(num_samples).double().to(env_ids.device)  # start from time zero
    
    return sampled_local_times

# to delete:
def compute_N_step_times(sampled_times: torch.Tensor, dt: float, num_amp_observations: int) -> torch.Tensor:
    times = (sampled_times.unsqueeze(-1) - dt * torch.arange(0, num_amp_observations, device=sampled_times.device, dtype=sampled_times.dtype)).flatten()
    return times

# to delete:
def zero_nan_reward(rewards: torch.Tensor, num: float = 0.0):
    nan_envs = find_nan(rewards)
    if torch.any(nan_envs):
        nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
        print(f"NaN detected in rewards {nan_env_ids.tolist()}.")
        rewards = torch.nan_to_num(rewards, nan=num, posinf=num, neginf=num)
    return rewards

def forward_keyframe_indexes(robot: "Articulation", keypose, alignment: float = 0.1):
    error = torch.mean(torch.norm(robot.data.body_pos_w - keypose, dim=-1), dim=1)
    return (error < alignment)

from isaaclab.utils.math import quat_mul
def root_state_face2face(root_state: torch.Tensor, distance: float = 1.2) -> torch.Tensor:
    root_state[:, 0] += distance
    q_rot180_single = torch.tensor([0, 0, 0, 1], device=root_state.device).unsqueeze(0)
    q_rot180 = q_rot180_single.repeat(root_state.shape[0], 1)
    root_state[:, 3:7] = quat_mul(q_rot180, root_state[:, 3:7])
    return root_state

def get_dof_body_index(source_body_names: list[str], target_body_names: list[str]) -> list[int]:
    indexes = []
    for name in source_body_names:
        assert name in target_body_names, f"The specified body name ({name}) doesn't exist: {target_body_names}"
        indexes.append(target_body_names.index(name))
    return indexes


def spawn_random_objects(random_object: RigidObject, env_mask: torch.Tensor, robot: Articulation, device: torch.device):
    # get environment indices where we need to spawn objects
    env_ids = env_mask.nonzero(as_tuple=False).squeeze(-1)
    
    if len(env_ids) == 0:
        return
    
    # get current object root state for all environments
    root_state = random_object.data.default_root_state.clone()
    
    # get robot positions for the specified environments
    robot_pos = robot.data.root_pos_w[env_ids]  # [num_spawning_envs, 3]

    
    # generate random offsets around the chosen robot positions
    random_offsets = math_utils.sample_cylinder(
        radius=0.8,  # spawn within 0.8 meter radius of robot
        h_range=(0.3, 1.5),  # height between 0.3 and 1.5 meters above robot
        size=len(env_ids),
        device=device
    )
    
    # update positions for the specified environments (near chosen robot)
    spawn_positions = robot_pos + random_offsets
    root_state[env_ids, :3] = spawn_positions
    

    # calculate throwing velocities
    # direction vector from spawn position to target robot
    direction = robot_pos - spawn_positions
    distance = torch.norm(direction, dim=1, keepdim=True)
    direction_normalized = direction / distance.clamp(min=0.1)  # avoid division by zero
    
    # calculate projectile motion parameters
    # using physics: v = sqrt(g * d / sin(2*theta)) for optimal angle
    gravity = 9.81
    optimal_angle = torch.tensor(torch.pi / 4, device=device)  # 45 degrees for maximum range
    
    # calculate required speed for projectile motion
    horizontal_distance = torch.norm(direction[:, :2], dim=1, keepdim=True)  # x-y plane distance
    height_diff = direction[:, 2:3]  # z difference
    
    # simple ballistic trajectory calculation
    speed = torch.sqrt(gravity * horizontal_distance / torch.sin(2 * optimal_angle))
    speed = torch.clamp(speed, min=2.0, max=5.0) 
    
    # create velocity vector
    # horizontal component
    horizontal_vel = direction_normalized[:, :2] * speed * torch.cos(optimal_angle)
    # vertical component (upward)
    vertical_vel = speed.squeeze(-1) * torch.sin(optimal_angle) + height_diff.squeeze(-1) * 0.5  # adjust for height difference
    
    # combine into full velocity vector
    throw_velocity = torch.cat([horizontal_vel, vertical_vel.unsqueeze(-1)], dim=-1)
    
    # set linear velocities for the specified environments
    root_state[env_ids, 7:10] = throw_velocity  # linear velocity
    
    # add some random angular velocity for more realistic motion
    angular_velocity = torch.randn(len(env_ids), 3, device=device) * 2.0  # random spin
    root_state[env_ids, 10:13] = angular_velocity  # angular velocity
    
    # write the entire root state to simulation (but only the modified environments will change)
    random_object.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
    random_object.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)