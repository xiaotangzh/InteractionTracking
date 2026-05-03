import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
from typing import Union, TYPE_CHECKING
from isaaclab_tasks.direct.PhysicsProject.utils.math import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.reward_utils import *
from isaaclab_tasks.direct.PhysicsProject.utils.func import *
from isaaclab_tasks.direct.PhysicsProject.motions.utils.torch_utils import exp_map_to_quat, quat_to_tan_norm, calc_heading_quat_inv, calc_heading_quat
from isaaclab_tasks.direct.PhysicsProject.envs.utils.state import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.mimic import *
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.PhysicsProject.motions.motion_loader import MotionLoader


def build_tracking_observation(robot: "Articulation", target_states: State, ref_body_index: int, env_ids: torch.Tensor, key_body_indexes: list | None=None) -> torch.Tensor:
    if env_ids is None: env_ids = robot._ALL_INDICES
    num_envs = len(env_ids)

    # proprioception = pos_vel_proprioception_obs(robot, ref_body_index, env_ids).reshape(num_envs, -1)
    # target = PHC_target_obs(env_ids, robot, target_states).reshape(num_envs, -1)
    # target = keypoint_target_obs(env_ids, robot, target_states).reshape(num_envs, -1)
    # return torch.cat((proprioception, target), dim=-1)

    return DeepMimic.Obs.compute(robot, target_states, env_ids, key_body_indexes)





def deepmimic_obs(robot: "Articulation", target: State, env_ids: torch.Tensor, key_body_indexes: list | None=None) -> torch.Tensor:
    proprioception = deepmimic_proprioception_obs(robot, env_ids, key_body_indexes)
    target_obs = deepmimic_target_obs(robot, target, env_ids, key_body_indexes)
    
    return torch.cat((proprioception, target_obs), dim=-1)

def deepmimic_proprioception_obs(robot: Union["Articulation", "State"], env_ids: torch.Tensor | None=None, key_body_indexes: list | None=None) -> torch.Tensor:
    if env_ids is None: env_ids = robot._ALL_INDICES
    num_envs = len(env_ids)

    proprioception = torch.cat([
        robot.data.root_state_w[env_ids, 1:2], # root height
        # quat_to_tan_norm(robot.data.root_state_w[env_ids, 3:7]), # root orientation
        quat_to_forward_up(robot.data.root_state_w[env_ids, 3:7]),
        robot.data.root_state_w[env_ids, 7:10], # root linear velocity
        robot.data.root_state_w[env_ids, 10:13], # root angular velocity
        robot.data.joint_pos[env_ids],
        robot.data.joint_vel[env_ids],
    ], dim=1)

    if key_body_indexes is not None:
        key_body_pos = robot.data.body_pos_w[:, key_body_indexes] - robot.data.root_state_w[:, :3].unsqueeze(1)
        key_body_pos = key_body_pos[env_ids]
        proprioception = torch.cat([proprioception, key_body_pos.view(num_envs, -1)], dim=-1)

    return proprioception.view(num_envs, -1)

def deepmimic_target_obs(robot: Union["Articulation", "State"], target: State, env_ids: torch.Tensor, key_body_indexes: list | None=None) -> torch.Tensor:
    num_envs = len(env_ids)

    root_state_w = robot.data.root_state_w if isinstance(robot, Articulation) else robot.root_state_w

    heading_inv_rot = calc_heading_quat_inv(root_state_w[env_ids, 3:7])
    obs = torch.cat([
        quat_apply(heading_inv_rot, 
                   target.root_pos_w[env_ids] - root_state_w[env_ids, :3]),
        # quat_to_tan_norm(quat_mul(heading_inv_rot, target.root_state_w[env_ids, 3:7])),
        # quat_diff(robot.data.root_state_w[env_ids, 3:7], target_states["root_state"][env_ids, 3:7]),
        quat_to_forward_up(quat_mul(heading_inv_rot, target.root_quat_w[env_ids])), # relative root rotation
        # quat_to_forward_up(target.root_quat_w[env_ids]), # absolute root rotation
        quat_apply(heading_inv_rot, target.root_lin_vel_w[env_ids]),
        quat_apply(heading_inv_rot, target.root_ang_vel_w[env_ids]),
        target.joint_pos[env_ids],
        target.joint_vel[env_ids],
    ], dim=-1)

    if key_body_indexes is not None:
        if target.key_body_pos_w is not None:
            key_body_pos = target.key_body_pos_w - target.root_pos_w.unsqueeze(1)
        else:
            key_body_pos = target.body_pos_w[:, key_body_indexes] - target.root_pos_w.unsqueeze(1)

        key_body_pos = key_body_pos[env_ids]

        # normalize to current root 
        key_body_pos = quat_apply(heading_inv_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1), key_body_pos)

        # normalize to target root 
        # heading_inv_rot = calc_heading_quat_inv(target.root_quat_w[env_ids]).unsqueeze(1).repeat(1, key_body_pos.shape[1], 1)
        # key_body_pos = quat_apply(heading_inv_rot, key_body_pos)

        obs = torch.cat([obs, key_body_pos.view(num_envs, -1)], dim=-1)

    return obs.view(num_envs, -1)

def reverse_deepmimic_target_obs(robot: "Articulation", target_obs: torch.Tensor, env_ids: torch.Tensor | None=None, key_body_indexes: list | None=None, root_accumulate: bool=False) -> torch.Tensor:
    """
    Reverse the encoded target observation back to world space.
    
    The encoding process in deepmimic_target_obs is:
        heading_inv_rot = calc_heading_quat_inv(robot_quat)
        forward_up = quat_to_forward_up(quat_mul(heading_inv_rot, target_quat))
    
    So to reverse:
        target_quat = quat_mul(heading_rot, forward_up_to_quat(forward_up))
    where heading_rot = calc_heading_quat(robot_quat)
    """
    if env_ids is None: env_ids = robot._ALL_INDICES
    num_envs = len(env_ids)
    num_dofs = robot.num_joints
    length = target_obs.shape[1]
    
    # Get heading rotation for each env (will broadcast across length)
    heading_rot = calc_heading_quat(robot.data.root_state_w[env_ids, 3:7])  # [num_envs, 4]
    heading_rot = heading_rot.unsqueeze(1).repeat(1, length, 1)  # [num_envs, length, 4] 
    
    # Parse the flattened tensor
    idx = 0
    
    # 1. Root position offset (3D)
    root_pos_offset = target_obs[:, :, idx:idx+3]  # [num_envs, length, 3]
    idx += 3
    
    # 2. Root orientation as forward_up (6D)
    forward_up = target_obs[:, :, idx:idx+6]  # [num_envs, length, 6]
    idx += 6
    
    # 3. Root linear velocity (3D)
    root_lin_vel_local = target_obs[:, :, idx:idx+3]  # [num_envs, length, 3]
    idx += 3
    
    # 4. Root angular velocity (3D)
    root_ang_vel_local = target_obs[:, :, idx:idx+3]  # [num_envs, length, 3]
    idx += 3
    
    # 5. DOF positions
    dof_pos = target_obs[:, :, idx:idx+num_dofs]  # [num_envs, length, num_dofs]
    idx += num_dofs
    
    # 6. DOF velocities
    dof_vel = target_obs[:, :, idx:idx+num_dofs]  # [num_envs, length, num_dofs]
    idx += num_dofs
    
    # Transform back to world space
    # heading_rot is [num_envs, 1, 4]
    robot_root_pos = robot.data.root_state_w[env_ids, :3].unsqueeze(1)  # [num_envs, 1, 3]

    if not root_accumulate:
        root_pos_world = robot_root_pos + quat_apply(heading_rot, root_pos_offset)  # [num_envs, length, 3]

        # Convert forward_up back to quaternion, then rotate to world frame
        root_quat_local = forward_up_to_quat(forward_up)  # [num_envs, length, 4]
        root_quat_world = quat_mul(heading_rot, root_quat_local)  # [num_envs, length, 4]
    else:
        # Accumulate root positions over time
        root_pos_world = torch.zeros((num_envs, length, 3), device=target_obs.device)
        root_quat_world = torch.zeros((num_envs, length, 4), device=target_obs.device)
        root_pos_world[:, 0, :] = robot_root_pos[:, 0, :] + quat_apply(heading_rot[:, 0, :], root_pos_offset[:, 0, :])
        root_quat_local = forward_up_to_quat(forward_up)  # [num_envs, length, 4]
        root_quat_world[:, 0, :] = quat_mul(heading_rot[:, 0, :], root_quat_local[:, 0, :])

        # use absoulute rotation
        root_quat_world = root_quat_local

        for t in range(1, length):
            delta_pos = quat_apply(calc_heading_quat(root_quat_world[:, t-1, :]), root_pos_offset[:, t, :]) # relative to last step root
            root_pos_world[:, t, :] = root_pos_world[:, t-1, :] + delta_pos

        #     # to fix:
        #     _heading_rot = calc_heading_quat(root_quat_world[:, t-1, :])  # [num_envs, 4]
        #     root_quat_world[:, t, :] = quat_mul(_heading_rot, root_quat_local[:, t, :])  # [num_envs, 4]
    
    
    root_lin_vel_world = quat_apply(heading_rot, root_lin_vel_local)  # [num_envs, length, 3]
    root_ang_vel_world = quat_apply(heading_rot, root_ang_vel_local)  # [num_envs, length, 3]
    
    root_state = torch.cat([
        root_pos_world,
        root_quat_world,
        root_lin_vel_world,
        root_ang_vel_world
    ], dim=-1)  # [num_envs, length, 13]
    
    result = torch.cat([
        root_state,
        dof_pos,
        dof_vel,
    ], dim=-1)
    
    # 7. Key body positions (optional)
    if key_body_indexes is not None:
        num_key_bodies = len(key_body_indexes)
        key_body_pos_local = target_obs[:, :, idx:idx+num_key_bodies*3]  # [num_envs, length, num_key_bodies*3]
        key_body_pos_local = key_body_pos_local.reshape(num_envs, length, num_key_bodies, 3)

        # unnormalize from current root
        heading_rot = heading_rot.unsqueeze(2).repeat(1, 1, num_key_bodies, 1)  # [num_envs, length, num_key_bodies, 4]

        # unnormalize from target root
        # heading_rot = calc_heading_quat(root_quat_world)  # [num_envs, length, 4]
        # heading_rot = heading_rot.unsqueeze(2).repeat(1, 1, num_key_bodies, 1)  # [num_envs, length, num_key_bodies, 4]


        key_body_pos_world = quat_apply(heading_rot, key_body_pos_local)  # [num_envs, length, num_key_bodies, 3]
        key_body_pos_world = key_body_pos_world + root_pos_world.unsqueeze(2)  # [num_envs, length, num_key_bodies, 3]
        
        # Flatten last two dimensions
        key_body_pos_world = key_body_pos_world.reshape(num_envs, length, num_key_bodies * 3)
        result = torch.cat([
            result, 
            key_body_pos_world
        ], dim=-1)
    
    return result

def verify_deepmimic_target_obs_reversibility(robot: "Articulation", target_states: dict, env_ids: torch.Tensor | None=None, key_body_indexes: list | None=None) -> dict:
    """
    Verify that the encoding and decoding are correct by comparing original vs reconstructed.
    Returns a dictionary with error metrics.
    """
    if env_ids is None: env_ids = robot._ALL_INDICES
    
    # Encode
    target_obs = deepmimic_target_obs(robot, target_states, env_ids, key_body_indexes)
    
    # Reshape to 3D for reverse function [num_envs, 1, features]
    target_obs_3d = target_obs.unsqueeze(1)
    
    # Decode
    reconstructed = reverse_deepmimic_target_obs(robot, target_obs_3d, env_ids, key_body_indexes)
    
    # Extract components from reconstructed
    idx = 0
    recon_root_state = reconstructed[:, 0, idx:idx+13]  # [num_envs, 13]
    idx += 13
    num_dofs = robot.num_joints
    recon_dof_pos = reconstructed[:, 0, idx:idx+num_dofs]
    idx += num_dofs
    recon_dof_vel = reconstructed[:, 0, idx:idx+num_dofs]
    idx += num_dofs
    
    # Compare with original
    orig_root_state = target_states["root_state"][env_ids]
    orig_dof_pos = target_states["dof_pos"][env_ids]
    orig_dof_vel = target_states["dof_vel"][env_ids]
    
    # Compute errors
    pos_error = torch.norm(recon_root_state[:, :3] - orig_root_state[:, :3], dim=-1)
    
    # Quaternion angular error
    q_orig = F.normalize(orig_root_state[:, 3:7], dim=-1)
    q_recon = F.normalize(recon_root_state[:, 3:7], dim=-1)
    dot = torch.sum(q_orig * q_recon, dim=-1).clamp(-1.0, 1.0)
    quat_error = 2 * torch.acos(torch.abs(dot))  # radians
    quat_error_deg = torch.rad2deg(quat_error)
    
    lin_vel_error = torch.norm(recon_root_state[:, 7:10] - orig_root_state[:, 7:10], dim=-1)
    ang_vel_error = torch.norm(recon_root_state[:, 10:13] - orig_root_state[:, 10:13], dim=-1)
    dof_pos_error = torch.norm(recon_dof_pos - orig_dof_pos, dim=-1)
    dof_vel_error = torch.norm(recon_dof_vel - orig_dof_vel, dim=-1)
    
    return {
        "pos_error_mean": pos_error.mean().item(),
        "pos_error_max": pos_error.max().item(),
        "quat_error_deg_mean": quat_error_deg.mean().item(),
        "quat_error_deg_max": quat_error_deg.max().item(),
        "lin_vel_error_mean": lin_vel_error.mean().item(),
        "ang_vel_error_mean": ang_vel_error.mean().item(),
        "dof_pos_error_mean": dof_pos_error.mean().item(),
        "dof_vel_error_mean": dof_vel_error.mean().item(),
        "orig_quat_sample": q_orig[0].tolist(),
        "recon_quat_sample": q_recon[0].tolist(),
    }

def get_add_vectors(env_ids, robot: "Articulation", target: State, key_body_indexes: list | None=None) -> torch.Tensor:
    # Adversarial Differential Discriminator

    if env_ids is None: env_ids = robot._ALL_INDICES
    num_envs = len(env_ids)

    diff = torch.cat([
        # target_states["root_state"][env_ids, :3] - robot.data.root_state_w[env_ids, :3],
        # # quat_diff(robot.data.root_state_w[env_ids, 3:7], target_states["root_state"][env_ids, 3:7]),
        # target_states["root_state"][env_ids, 3:7] - robot.data.root_state_w[env_ids, 3:7],
        # target_states["root_state"][env_ids, 7:10] - robot.data.root_state_w[env_ids, 7:10],
        # target_states["root_state"][env_ids, 10:] - robot.data.root_state_w[env_ids, 10:],

        target.root_state_w[env_ids] - robot.data.root_state_w[env_ids],
        target.joint_pos[env_ids] - robot.data.joint_pos[env_ids],
        target.joint_vel[env_ids] - robot.data.joint_vel[env_ids],
    ], dim=-1)

    if key_body_indexes is not None:
        if target.key_body_pos_w is not None:
            key_body_diff = target.key_body_pos_w - robot.data.body_pos_w[:, key_body_indexes]
        else:
            key_body_diff = target.body_pos_w[:, key_body_indexes] - robot.data.body_pos_w[:, key_body_indexes]

        key_body_diff = key_body_diff[env_ids]
        diff = torch.cat([diff, key_body_diff.view(num_envs, -1)], dim=-1)

    return diff.view(num_envs, -1)

def keypoint_target_obs(env_ids, robot: "Articulation", target_states: dict) -> torch.Tensor:
    target_pos = target_states["body_pos"]
    target_lin_vel = target_states["body_lin_vel"]

    target = torch.cat([
        target_pos[env_ids] - robot.data.body_pos_w[env_ids], # pos offset
        target_lin_vel[env_ids] - robot.data.body_lin_vel_w[env_ids], # vel offset
    ], dim=-1)

    return target

def PHC_target_obs(env_ids, robot: "Articulation", target: State) -> torch.Tensor:
    num_envs = len(env_ids)
    root_state_w = robot.data.root_state_w if isinstance(robot, Articulation) else target.root_state_w
    heading_inv_rot = calc_heading_quat_inv(root_state_w[env_ids, 3:7])
    heading_inv_rot_expanded = heading_inv_rot.unsqueeze(1).repeat(1, target.body_pos_w.shape[1], 1)

    # obs = torch.cat([
    #     target_pos[env_ids] - robot.data.body_pos_w[env_ids],
    #     target_lin_vel[env_ids] - robot.data.body_lin_vel_w[env_ids],
    #     quat_diff(robot.data.body_quat_w[env_ids], target_rot[env_ids]),
    #     target_ang_vel[env_ids] - robot.data.body_ang_vel_w[env_ids],
    #     # target_rot[env_ids],
    #     # target_pos[env_ids]
    # ], dim=-1).view(num_envs, -1)

    obs = torch.cat([
        quat_apply(heading_inv_rot_expanded, 
                   target.body_pos_w[env_ids] - robot.data.body_pos_w[env_ids]).view(num_envs, -1),
        quat_to_forward_up(quat_mul(heading_inv_rot_expanded, target.body_quat_w[env_ids])).view(num_envs, -1),
        quat_apply(heading_inv_rot_expanded, target.body_lin_vel_w[env_ids]).view(num_envs, -1),
        quat_apply(heading_inv_rot_expanded, target.body_ang_vel_w[env_ids]).view(num_envs, -1),
        target.joint_pos[env_ids],
        target.joint_vel[env_ids],
    ], dim=-1)

    return obs

def pos_vel_proprioception_obs(robot: "Articulation", ref_body_index: int, env_ids: torch.Tensor | None=None, space: str="local") -> torch.Tensor:
    if env_ids is None: env_ids = robot._ALL_INDICES
    num_envs = len(env_ids)

    if space == "local":
        proprioception = torch.cat([
            robot.data.body_pos_w[env_ids] - robot.data.body_pos_w[env_ids, ref_body_index].unsqueeze(1), 
            robot.data.body_lin_vel_w[env_ids] - robot.data.body_lin_vel_w[env_ids, ref_body_index].unsqueeze(1),
        ], dim=-1).reshape(num_envs, -1)
    else:
        proprioception = torch.cat([
            robot.data.body_pos_w[env_ids],
            robot.data.body_lin_vel_w[env_ids],
        ], dim=-1).reshape(num_envs, -1)

    return proprioception
