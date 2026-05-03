import torch
from torch.nn.functional import mse_loss
import isaaclab.utils.math as math_utils
from isaaclab_tasks.direct.PhysicsProject.utils.math import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.utils import *
from isaaclab_tasks.direct.PhysicsProject.utils.func import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.state import State
from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from isaaclab.assets import Articulation

def masked_mimic_reward(robot: "Articulation", gt: dict) -> torch.Tensor:
    # MaskedMimic: (0.5 * BodyPos) + (0.3 * BodyRot) + (0.1 * BodyVel) + (0.1 * BodyAngVel) + (0.2  * RootHeight) + (0.0005 * Energy)
    pass

def phc_reward(robot: "Articulation", gt: State) -> torch.Tensor:
    # PHC: (0.5 * BodyPos) + (0.3 * BodyRot) + (0.1 * BodyVel) + (0.1 * BodyAngVel)
    pos = body_pos_reward(robot, gt)
    rot = body_rot_reward(robot, gt)
    lin_vel = body_lin_vel_reward(robot, gt)
    ang_vel = body_angular_vel_reward(robot, gt)
    # printl(f"PHC Plus DoF Reward: pos {pos.mean().item()}, rot {rot.mean().item()}, lin_vel {lin_vel.mean().item()}, ang_vel {ang_vel.mean().item()}")
    return pos * 0.5 + rot * 0.3 + lin_vel * 0.1 + ang_vel * 0.1

def phc_plus_dof_reward(robot: "Articulation", gt: State) -> torch.Tensor:
    pos = body_pos_reward(robot, gt) 
    rot = body_rot_reward(robot, gt)
    lin_vel = body_lin_vel_reward(robot, gt)
    ang_vel = body_angular_vel_reward(robot, gt)
    dof_pos = dof_pos_reward(robot, gt)
    dof_vel = dof_vel_reward(robot, gt)
    # printl(f"PHC Plus DoF Reward: pos {pos.mean().item()}, rot {rot.mean().item()}, lin_vel {lin_vel.mean().item()}, ang_vel {ang_vel.mean().item()}, dof_pos {dof_pos.mean().item()}, dof_vel {dof_vel.mean().item()}")
    return pos * 0.4 + lin_vel * 0.1 + rot * 0.2 + ang_vel * 0.1 + dof_pos * 0.1 + dof_vel * 0.1

def tracking_pos_dof_reward(robot: "Articulation", gt: State) -> torch.Tensor:
    pos = body_pos_reward(robot, gt)
    dof_pos = dof_pos_reward(robot, gt, scale=-2.0) if gt.joint_pos is not None else torch.zeros_like(pos)
    dof_vel = dof_vel_reward(robot, gt, scale=-0.02) if gt.joint_vel is not None else torch.zeros_like(pos)
    return pos * 0.5 + dof_pos * 0.25 + dof_vel * 0.25

def tracking_pos_vel_reward(robot: "Articulation", gt: State) -> torch.Tensor:
    pos = body_pos_reward(robot, gt) 
    lin_vel = body_lin_vel_reward(robot, gt)
    return pos * 0.5 + lin_vel * 0.5

def energy_penalty(robot: "Articulation", scale: float = 0.03) -> torch.Tensor:
    dof_forces = robot.data.applied_torque # [envs, dofs]
    dof_vel = robot.data.joint_vel # [envs, dofs] # note: dof_vel is already in rad/s (angular velocity)
    power = (-1) * torch.abs(torch.multiply(dof_forces, dof_vel)).mean(dim=-1) 
    power = torch.exp(power * scale)
    return power

def compute_robots_avg_distance(robot1: "Articulation", robot2: "Articulation") -> torch.Tensor:
    pos1 = robot1.data.body_pos_w
    pos2 = robot2.data.body_pos_w
    distance = torch.mean(torch.norm(pos1 - pos2, dim=-1), dim=1, keepdim=False)
    return distance

# test best scale for each reward
list_pos = []
list_rot = []
list_lin_vel = []
list_ang_vel = []
list_dof_pos = []
list_dof_vel = []
import math 

# Section: Motion rewards normalized to [0,1]
def body_pos_reward(robot: "Articulation", gt: State, scale: float = -162.1) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    pos, ref_pos = robot.data.body_pos_w.reshape(num_envs, -1), gt.body_pos_w.reshape(num_envs, -1)

    # list_pos.append(mse(pos, ref_pos, start_dim = 1))
    # if len(list_pos) > 100:
    #     mean_mse = torch.cat(list_pos, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)

    reward = torch.exp(mse(pos, ref_pos, start_dim = 1) * scale) 
    # print(f"Body Position Reward: {reward.mean().item()}")

    return reward

def body_rot_reward(robot: "Articulation", gt: State, scale: float = -201) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    rot, ref_rot = robot.data.body_quat_w.reshape(num_envs, -1), gt.body_quat_w.reshape(num_envs, -1) 

    # list_rot.append(mse(quat_diff(rot, ref_rot), start_dim = 1))
    # if len(list_rot) > 100:
    #     mean_mse = torch.cat(list_rot, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)

    if gt.body_quat_w.shape[-1] == 4:
        reward = torch.exp(mse(quat_diff(rot, ref_rot), start_dim = 1) * scale)
    else:
        reward = torch.exp(mse(axis_angle_from_quat(robot.data.body_quat_w).reshape(num_envs, -1) - ref_rot, start_dim = 1) * scale)
    # print(f"Body Rotation Reward: {reward.mean().item()}")

    return reward

def body_lin_vel_reward(robot: "Articulation", gt: State, scale: float = -3.17) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    vel, ref_vel = robot.data.body_lin_vel_w.reshape(num_envs, -1), gt.body_lin_vel_w.reshape(num_envs, -1)

    # list_lin_vel.append(mse(vel, ref_vel, start_dim = 1))
    # if len(list_lin_vel) > 100:
    #     mean_mse = torch.cat(list_lin_vel, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)

    reward = torch.exp(mse(vel, ref_vel, start_dim = 1) * scale) 
    # print(f"Body Linear Velocity Reward: {reward.mean().item()}")

    return reward

def body_angular_vel_reward(robot: "Articulation", gt: State, scale: float = -0.116) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    ang_vel, ref_ang_vel = robot.data.body_ang_vel_w.reshape(num_envs, -1), gt.body_ang_vel_w.reshape(num_envs, -1)

    # list_ang_vel.append(mse(ang_vel, ref_ang_vel, start_dim = 1))
    # if len(list_ang_vel) > 100:
    #     mean_mse = torch.cat(list_ang_vel, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)

    reward = torch.exp(mse(ang_vel, ref_ang_vel, start_dim = 1) * scale) 
    # print(f"Body Angular Velocity Reward: {reward.mean().item()}")

    return reward

def dof_pos_reward(robot: "Articulation", gt: State, scale: float = -17.87) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    dof_pos, ref_dof_pos = robot.data.joint_pos.reshape(num_envs, -1), gt.joint_pos.reshape(num_envs, -1)

    # list_dof_pos.append(mse(dof_pos, ref_dof_pos, start_dim = 1))
    # if len(list_dof_pos) > 100:
    #     mean_mse = torch.cat(list_dof_pos, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)
    
    reward = torch.exp(mse(dof_pos, ref_dof_pos, start_dim = 1) * scale)
    # print(f"DoF Position Reward: {reward.mean().item()}")

    return reward

def dof_vel_reward(robot: "Articulation", gt: State, scale: float = -0.178) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]
    dof_vel, ref_dof_vel = robot.data.joint_vel.reshape(num_envs, -1), gt.joint_vel.reshape(num_envs, -1)

    # list_dof_vel.append(mse(dof_vel, ref_dof_vel, start_dim = 1))
    # if len(list_dof_vel) > 100:
    #     mean_mse = torch.cat(list_dof_vel, dim=0).mean().item()
    #     scale = math.log(0.001) / (mean_mse)
    #     print(scale)
    
    reward = torch.exp(mse(dof_vel, ref_dof_vel, start_dim = 1) * scale)
    # print(f"DoF Velocity Reward: {reward.mean().item()}")

    return reward 

def deepmimic_reward(robot: "Articulation", gt: State, key_body_indexes: torch.Tensor | None=None) -> torch.Tensor:
    num_envs = robot.data.body_pos_w.shape[0]

    root_pos, ref_root_pos = robot.data.root_state_w[:, :3], gt.root_pos_w
    root_rot, ref_root_rot = robot.data.root_state_w[:, 3:7], gt.root_quat_w
    root_vel, ref_root_vel = robot.data.root_state_w[:, 7:10], gt.root_lin_vel_w
    root_ang_vel, ref_root_ang_vel = robot.data.root_state_w[:, 10:13], gt.root_ang_vel_w

    root_pos_reward = torch.exp(-10 * torch.norm(root_pos - ref_root_pos, dim=-1) ** 2)
    root_rot_reward = torch.exp(-2 * torch.norm(quat_diff(root_rot, ref_root_rot), dim=-1) ** 2)
    root_vel_reward = torch.exp(-0.5 * torch.norm(root_vel - ref_root_vel, dim=-1) ** 2)
    root_ang_vel_reward = torch.exp(-0.02 * torch.norm(root_ang_vel - ref_root_ang_vel, dim=-1) ** 2)
    pos_reward = dof_pos_reward(robot, gt, scale=-2.0)
    vel_reward = dof_vel_reward(robot, gt, scale=-0.02)
    
    if key_body_indexes is not None:
        if gt.key_body_pos_w is not None:
            key_body_pos, ref_key_body_pos = robot.data.body_pos_w[:, key_body_indexes], gt.key_body_pos_w
        else:
            key_body_pos, ref_key_body_pos = robot.data.body_pos_w[:, key_body_indexes], gt.body_pos_w[:, key_body_indexes]
        key_body_pos_reward = torch.exp(-3.0 * (torch.norm(key_body_pos - ref_key_body_pos, dim=-1) ** 2).mean(dim=-1))

    # printl(f"DeepMimic Reward: root_pos {root_pos_reward.mean().item()},\n root_rot {root_rot_reward.mean().item()},\n root_vel {root_vel_reward.mean().item()},\n root_ang_vel {root_ang_vel_reward.mean().item()},\n dof_pos {pos_reward.mean().item()},\n dof_vel {vel_reward.mean().item()}\n, key_body_pos {key_body_pos_reward.mean().item() if key_body_indexes is not None else 'N/A'}")

    reward = root_pos_reward + root_rot_reward + root_vel_reward + root_ang_vel_reward + pos_reward + vel_reward + 0.0 * (key_body_pos_reward if key_body_indexes is not None else 0)
    reward = reward / (6 + (0.0 if key_body_indexes is not None else 0))
    return reward