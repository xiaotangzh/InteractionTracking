import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
from typing import Union, TYPE_CHECKING
from isaaclab_tasks.direct.PhysicsProject.utils.math import *
from isaaclab_tasks.direct.PhysicsProject.utils.func import *
from isaaclab_tasks.direct.PhysicsProject.motions.utils.torch_utils import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.state import *
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.PhysicsProject.motions.motion_loader import MotionLoader

class Mimic:
    """Base class for motion imitation observation generation"""
    
    class Obs:
        """Base observation generation methods"""

        @staticmethod
        def compute_diff(
            robot: "Articulation", 
            gt: State,
        ) -> torch.Tensor:

            root_pos, ref_root_pos = robot.data.root_state_w[:, :3], gt.root_pos_w
            root_rot, ref_root_rot = robot.data.root_state_w[:, 3:7], gt.root_quat_w
            root_vel, ref_root_vel = robot.data.root_state_w[:, 7:10], gt.root_lin_vel_w
            root_ang_vel, ref_root_ang_vel = robot.data.root_state_w[:, 10:13], gt.root_ang_vel_w
            
            diff = torch.cat([
                root_pos - ref_root_pos,
                root_rot, ref_root_rot,
                root_vel - ref_root_vel,
                root_ang_vel - ref_root_ang_vel,
                robot.data.joint_pos - gt.joint_pos,
                robot.data.joint_vel - gt.joint_vel,
            ], dim=-1)

            return diff
        
        @staticmethod
        def compute_mse(
            robot: "Articulation", 
            gt: State, 
        ) -> torch.Tensor:

            diff = Mimic.Obs.compute_diff(robot, gt)
            mse = torch.mean(diff ** 2, dim=-1)
            mse = torch.exp(-0.3 * mse)

            return mse
    
    class Reward:
        """Base reward computation methods"""
        
        @staticmethod
        def body_pos_reward(robot: "Articulation", gt: State, scale: float = -162.1) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            num_bodies = robot.data.body_pos_w.shape[1]

            robot_heading_inv = calc_heading_quat_inv(robot.data.root_state_w[:, 3:7]).unsqueeze(1).expand(-1, num_bodies, -1)
            gt_heading_inv = calc_heading_quat_inv(gt.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)

            pos = quat_apply(robot_heading_inv, robot.data.body_pos_w - robot.data.root_state_w[:, :3].unsqueeze(1))
            ref_pos = quat_apply(gt_heading_inv, gt.body_pos_w - gt.root_pos_w.unsqueeze(1))

            reward = torch.exp(mse(pos.reshape(num_envs, -1), ref_pos.reshape(num_envs, -1), start_dim=1) * scale)
            # print(f"Body Position Reward: {reward.mean().item()}")

            return reward

        @staticmethod
        def body_rot_reward(robot: "Articulation", gt: State, scale: float = -201) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            rot, ref_rot = robot.data.body_quat_w.reshape(num_envs, -1), gt.body_quat_w.reshape(num_envs, -1) 

            if gt.body_quat_w.shape[-1] == 4:
                reward = torch.exp(mse(quat_diff(rot, ref_rot), start_dim = 1) * scale)
            else:
                reward = torch.exp(mse(axis_angle_from_quat(robot.data.body_quat_w).reshape(num_envs, -1) - ref_rot, start_dim = 1) * scale)
            # print(f"Body Rotation Reward: {reward.mean().item()}")

            return reward

        @staticmethod
        def body_lin_vel_reward(robot: "Articulation", gt: State, scale: float = -3.17) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            num_bodies = robot.data.body_lin_vel_w.shape[1]

            robot_heading_inv = calc_heading_quat_inv(robot.data.root_state_w[:, 3:7]).unsqueeze(1).expand(-1, num_bodies, -1)
            gt_heading_inv = calc_heading_quat_inv(gt.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)

            vel = quat_apply(robot_heading_inv, robot.data.body_lin_vel_w)
            ref_vel = quat_apply(gt_heading_inv, gt.body_lin_vel_w)

            reward = torch.exp(mse(vel.reshape(num_envs, -1), ref_vel.reshape(num_envs, -1), start_dim=1) * scale)
            # print(f"Body Linear Velocity Reward: {reward.mean().item()}")

            return reward

        @staticmethod
        def body_angular_vel_reward(robot: "Articulation", gt: State, scale: float = -0.116) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            num_bodies = robot.data.body_ang_vel_w.shape[1]

            robot_heading_inv = calc_heading_quat_inv(robot.data.root_state_w[:, 3:7]).unsqueeze(1).expand(-1, num_bodies, -1)
            gt_heading_inv = calc_heading_quat_inv(gt.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1)

            ang_vel = quat_apply(robot_heading_inv, robot.data.body_ang_vel_w)
            ref_ang_vel = quat_apply(gt_heading_inv, gt.body_ang_vel_w)

            reward = torch.exp(mse(ang_vel.reshape(num_envs, -1), ref_ang_vel.reshape(num_envs, -1), start_dim=1) * scale)
            # print(f"Body Angular Velocity Reward: {reward.mean().item()}")

            return reward

        @staticmethod
        def compute_dof_pos_reward(robot: "Articulation", gt: State, scale: float = -2.0) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            dof_pos, ref_dof_pos = robot.data.joint_pos.reshape(num_envs, -1), gt.joint_pos.reshape(num_envs, -1)

            reward = torch.exp(mse(dof_pos, ref_dof_pos, start_dim = 1) * scale)
            # print(f"DoF Position Reward: {reward.mean().item()}")

            return reward

        @staticmethod
        def compute_dof_vel_reward(robot: "Articulation", gt: State, scale: float = -0.02) -> torch.Tensor:
            num_envs = robot.data.body_pos_w.shape[0]
            dof_vel, ref_dof_vel = robot.data.joint_vel.reshape(num_envs, -1), gt.joint_vel.reshape(num_envs, -1)

            reward = torch.exp(mse(dof_vel, ref_dof_vel, start_dim = 1) * scale)
            # print(f"DoF Velocity Reward: {reward.mean().item()}")

            return reward 

class DeepMimic(Mimic):

    class Reward(Mimic.Reward):

        @staticmethod
        def compute(
            robot: "Articulation", 
            gt: State, 
            key_body_indexes: torch.Tensor | list | None=None
        ) -> torch.Tensor:

            has_key_body = False
            if key_body_indexes is not None:
                if isinstance(key_body_indexes, torch.Tensor):
                    has_key_body = key_body_indexes.numel() > 0
                else:
                    has_key_body = len(key_body_indexes) > 0

            root_pos, ref_root_pos = robot.data.root_state_w[:, :3], gt.root_pos_w
            root_rot, ref_root_rot = robot.data.root_state_w[:, 3:7], gt.root_quat_w
            root_vel, ref_root_vel = robot.data.root_state_w[:, 7:10], gt.root_lin_vel_w
            root_ang_vel, ref_root_ang_vel = robot.data.root_state_w[:, 10:13], gt.root_ang_vel_w

            root_pos_reward = torch.exp(-10 * torch.norm(root_pos - ref_root_pos, dim=-1) ** 2)
            root_rot_reward = torch.exp(-2 * torch.norm(quat_diff(root_rot, ref_root_rot), dim=-1) ** 2)
            root_vel_reward = torch.exp(-0.5 * torch.norm(root_vel - ref_root_vel, dim=-1) ** 2)
            root_ang_vel_reward = torch.exp(-0.02 * torch.norm(root_ang_vel - ref_root_ang_vel, dim=-1) ** 2)
            dof_pos_reward = Mimic.Reward.compute_dof_pos_reward(robot, gt)
            dof_vel_reward = Mimic.Reward.compute_dof_vel_reward(robot, gt)

            reward = root_pos_reward + root_rot_reward + root_vel_reward + root_ang_vel_reward + dof_pos_reward + dof_vel_reward
            num_terms = 6

            if has_key_body:
                key_body_pos, ref_key_body_pos = robot.data.body_pos_w[:, key_body_indexes], gt.body_pos_w[:, key_body_indexes]
                key_body_pos_reward = torch.exp(-3.0 * (torch.norm(key_body_pos - ref_key_body_pos, dim=-1) ** 2).mean(dim=-1))
                reward = reward + key_body_pos_reward
                num_terms += 1

            # printl(f"DeepMimic Reward: root_pos {root_pos_reward.mean().item()},\n root_rot {root_rot_reward.mean().item()},\n root_vel {root_vel_reward.mean().item()},\n root_ang_vel {root_ang_vel_reward.mean().item()},\n dof_pos {dof_pos_reward.mean().item()},\n dof_vel {dof_vel_reward.mean().item()}\n, key_body_pos {key_body_pos_reward.mean().item() if key_body_indexes is not None else 'N/A'}")

            reward = reward / float(num_terms)

            return reward
        
        @staticmethod
        def compute_key_body_pos(robot: "Articulation", gt: State, key_body_indexes: torch.Tensor | list | None) -> torch.Tensor:
            key_body_pos, ref_key_body_pos = robot.data.body_pos_w[:, key_body_indexes], gt.body_pos_w[:, key_body_indexes]
            key_body_pos_reward = torch.exp(-3.0 * (torch.norm(key_body_pos - ref_key_body_pos, dim=-1) ** 2).mean(dim=-1))
            return key_body_pos_reward
        

    class Obs(Mimic.Obs):

        @staticmethod
        def compute(
            robot: "Articulation", 
            target: State, 
            env_ids: torch.Tensor, 
            key_body_indexes: list | None=None
        ) -> torch.Tensor:

            proprioception = DeepMimic.Obs.proprioception(robot, env_ids, key_body_indexes)
            target_obs = DeepMimic.Obs.target(robot, target, env_ids, key_body_indexes)

            return torch.cat((proprioception, target_obs), dim=-1)
        
        @staticmethod
        def proprioception(
            robot: Union["Articulation", "State"], 
            env_ids: torch.Tensor | None=None, 
            key_body_indexes: list | None=None
        ) -> torch.Tensor:

            if env_ids is None: env_ids = robot._ALL_INDICES
            num_envs = len(env_ids)

            heading_inv_rot = calc_heading_quat_inv(robot.data.root_state_w[env_ids, 3:7])
            obs = torch.cat([
                robot.data.root_state_w[env_ids, 2:3], # root height
                # quat_to_tan_norm(robot.data.root_state_w[env_ids, 3:7]), # root orientation
                quat_to_forward_up(robot.data.root_state_w[env_ids, 3:7]), # easier to reverse
                robot.data.root_state_w[env_ids, 7:10], # root linear velocity
                robot.data.root_state_w[env_ids, 10:13], # root angular velocity
                robot.data.joint_pos[env_ids],
                robot.data.joint_vel[env_ids],
            ], dim=1)

            if key_body_indexes:
                key_body_pos = robot.data.body_pos_w[:, key_body_indexes] - robot.data.root_state_w[:, :3].unsqueeze(1)
                key_body_pos = key_body_pos[env_ids]
                heading_inv_rot = calc_heading_quat_inv(robot.data.root_state_w[env_ids, 3:7]).unsqueeze(1).repeat(1, key_body_pos.shape[1], 1)
                key_body_pos = quat_apply(heading_inv_rot, key_body_pos)
                obs = torch.cat([obs, key_body_pos.view(num_envs, -1)], dim=-1)

            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            return obs.view(num_envs, -1)    
        
        @staticmethod
        def target(
            robot: Union["Articulation", "State"], 
            target: State, 
            env_ids: torch.Tensor, 
            key_body_indexes: list | None=None
        ) -> torch.Tensor:
            
            num_envs = len(env_ids)
            root_state_w = robot.data.root_state_w

            heading_inv_rot = calc_heading_quat_inv(root_state_w[env_ids, 3:7])
            obs = torch.cat([
                quat_apply(heading_inv_rot, 
                        target.root_pos_w[env_ids] - root_state_w[env_ids, :3]),
                # quat_to_tan_norm(quat_mul(heading_inv_rot, target.root_state_w[env_ids, 3:7])),
                quat_to_forward_up(quat_mul(heading_inv_rot, target.root_quat_w[env_ids])), # relative root rotation
                # quat_to_forward_up(target.root_quat_w[env_ids]), # absolute root rotation
                target.root_lin_vel_w[env_ids],
                target.root_ang_vel_w[env_ids],
                target.joint_pos[env_ids],
                target.joint_vel[env_ids],
            ], dim=-1)

            if key_body_indexes:
                key_body_pos = target.body_pos_w[:, key_body_indexes] - target.root_pos_w.unsqueeze(1)
                key_body_pos = key_body_pos[env_ids]

                # normalize to current root 
                # key_body_pos = quat_apply(heading_inv_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1), key_body_pos)

                # normalize to target root 
                heading_inv_rot = calc_heading_quat_inv(target.root_quat_w[env_ids]).unsqueeze(1).repeat(1, key_body_pos.shape[1], 1)
                key_body_pos = quat_apply(heading_inv_rot, key_body_pos)

                obs = torch.cat([obs, key_body_pos.view(num_envs, -1)], dim=-1)
            
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            return obs.view(num_envs, -1)

        @staticmethod
        def reverse_target(
            robot: Union["Articulation", "State"], 
            target_obs: torch.Tensor, 
            env_ids: torch.Tensor, 
            key_body_indexes: list | None=None, 
            root_accumulate: bool=False
        ) -> tuple[State, torch.Tensor]:
            """
            Reverse the encoded target observation back to world space.
            
            The encoding process in deepmimic_target_obs is:
                heading_inv_rot = calc_heading_quat_inv(robot_quat)
                forward_up = quat_to_forward_up(quat_mul(heading_inv_rot, target_quat))
            
            So to reverse:
                target_quat = quat_mul(heading_rot, forward_up_to_quat(forward_up))
            where heading_rot = calc_heading_quat(robot_quat)
            """

            num_envs = len(env_ids)
            num_dofs = robot.num_joints
            length = target_obs.shape[1]
            
            # Get heading rotation for each env (will broadcast across length for non-accumulate)
            # This is H_0 (Heading of reference frame)
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
                # Relative to the fixed robot reference (t=0)
                root_pos_world = robot_root_pos + quat_apply(heading_rot, root_pos_offset)  # [num_envs, length, 3]

                # Convert forward_up back to quaternion, then rotate to world frame
                root_quat_local = forward_up_to_quat(forward_up)  # [num_envs, length, 4]
                root_quat_world = quat_mul(heading_rot, root_quat_local)  # [num_envs, length, 4]
                root_lin_vel_world = quat_apply(heading_rot, root_lin_vel_local)  # [num_envs, length, 3]
                root_ang_vel_world = quat_apply(heading_rot, root_ang_vel_local)  # [num_envs, length, 3]

            else:
                # Root is relative to last frame instead of reference first frame
                root_pos_world = torch.zeros((num_envs, length, 3), device=target_obs.device)
                root_quat_world = torch.zeros((num_envs, length, 4), device=target_obs.device)
                root_lin_vel_world = torch.zeros((num_envs, length, 3), device=target_obs.device)
                root_ang_vel_world = torch.zeros((num_envs, length, 3), device=target_obs.device)
                
                # First frame
                root_pos_world[:, 0, :] = robot_root_pos[:, 0, :] + quat_apply(heading_rot[:, 0, :], root_pos_offset[:, 0, :])
                root_quat_local = forward_up_to_quat(forward_up)  # [num_envs, length, 4]
                root_quat_world[:, 0, :] = quat_mul(heading_rot[:, 0, :], root_quat_local[:, 0, :])
                root_lin_vel_world[:, 0, :] = quat_apply(heading_rot[:, 0, :], root_lin_vel_local[:, 0, :])
                root_ang_vel_world[:, 0, :] = quat_apply(heading_rot[:, 0, :], root_ang_vel_local[:, 0, :])

                for t in range(1, length):
                    new_heading_rot = calc_heading_quat(root_quat_world[:, t-1, :])  # [num_envs, 4]
                    root_pos_world[:, t, :] = root_pos_world[:, t-1, :] + quat_apply(new_heading_rot, root_pos_offset[:, t, :])
                    root_quat_world[:, t, :] = quat_mul(new_heading_rot, root_quat_local[:, t, :])
                    root_lin_vel_world[:, t, :] = quat_apply(new_heading_rot, root_lin_vel_local[:, t, :])
                    root_ang_vel_world[:, t, :] = quat_apply(new_heading_rot, root_ang_vel_local[:, t, :])

                    heading_rot[:, t, :] = new_heading_rot # update heading rot for next step
            
            
            root_state = torch.cat([
                root_pos_world,
                root_quat_world,
                root_lin_vel_world,
                root_ang_vel_world
            ], dim=-1)  # [num_envs, length, 13]
            
            tensors = torch.cat([
                root_state,
                dof_pos,
                dof_vel,
            ], dim=-1)
            
            # 7. Key body positions (optional)
            if key_body_indexes:
                num_key_bodies = len(key_body_indexes)
                key_body_pos_local = target_obs[:, :, idx:idx+num_key_bodies*3]  # [num_envs, length, num_key_bodies*3]
                key_body_pos_local = key_body_pos_local.reshape(num_envs, length, num_key_bodies, 3)

                # unnormalize from current root
                # heading_rot = heading_rot.unsqueeze(2).repeat(1, 1, num_key_bodies, 1)  # [num_envs, length, num_key_bodies, 4]

                # unnormalize from target root
                heading_rot = calc_heading_quat(root_quat_world)  # [num_envs, length, 4]
                heading_rot = heading_rot.unsqueeze(2).repeat(1, 1, num_key_bodies, 1)  # [num_envs, length, num_key_bodies, 4]


                key_body_pos_world = quat_apply(heading_rot, key_body_pos_local)  # [num_envs, length, num_key_bodies, 3]
                key_body_pos_world = key_body_pos_world + root_pos_world.unsqueeze(2)  # [num_envs, length, num_key_bodies, 3]
                
                # Flatten last two dimensions
                key_body_pos_world = key_body_pos_world.reshape(num_envs, length, num_key_bodies * 3)
                tensors = torch.cat([
                    tensors, 
                    key_body_pos_world
                ], dim=-1)

            states = State(
                root_state_w=root_state.reshape(-1, length, 13),
                joint_pos=dof_pos.reshape(-1, length, num_dofs),
                joint_vel=dof_vel.reshape(-1, length, num_dofs),
                key_body_pos_w=key_body_pos_world.reshape(-1, length, num_key_bodies, 3) if key_body_indexes else None,
            )
            
            return states, tensors


# class PHC(Mimic):
#     class Reward(Mimic.Reward):

#         @staticmethod
#         def compute(robot: "Articulation", gt: State, key_body_indexes: torch.Tensor | None=None) -> torch.Tensor:
#             reward = phc_reward(robot, gt)
#             return reward
    
#     class Obs(Mimic.Obs):

#         @staticmethod
#         def compute(robot: "Articulation", target: State, env_ids: torch.Tensor, key_body_indexes: list | None=None) -> torch.Tensor:
#             proprioception = phc_proprioception(robot, env_ids, key_body_indexes)
#             target_obs = phc_target(robot, target, env_ids, key_body_indexes)

#             return torch.cat((proprioception, target_obs), dim=-1)
