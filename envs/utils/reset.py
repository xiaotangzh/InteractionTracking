import torch
from typing import Union, TYPE_CHECKING
from isaaclab_tasks.direct.PhysicsProject.utils.math import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.reward_utils import *
from isaaclab_tasks.direct.PhysicsProject.utils.func import *
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.PhysicsProject.motions.motion_loader import MotionLoader

def reset_robot(
    robot: "Articulation", 
    env_ids: torch.Tensor, 
    root_state: torch.Tensor, 
    joint_pos: torch.Tensor, 
    joint_vel: torch.Tensor
):
    robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)

def reset_strategy_default(
    env_origins, 
    env_ids: torch.Tensor, 
    robot: "Articulation", 
    lift_root_height: float=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env_origins[env_ids]
    root_state[:, 2] += lift_root_height  # lift the humanoid slightly to avoid collisions with the ground
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    return root_state, joint_pos, joint_vel

def reset_strategy_random(
    env_ids: torch.Tensor, 
    robot: "Articulation",
    sampled_motion_ids: torch.Tensor,
    sampled_times: torch.Tensor,
    motion_loader: "MotionLoader",
    ref_body_index: int,
    lift_root_height: float,
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # env_ids: the ids of envs to be reset
    
    num_samples = env_ids.shape[0]

    # sample random motions
    states = motion_loader.sample(num_samples=num_samples, motion_ids=sampled_motion_ids, local_times=sampled_times)
    
    # get root transforms (the humanoid torso)
    states.root_state_w[:, 0:3] = states.root_state_w[:, 0:3] + env_origins[env_ids]
    states.root_state_w[:, 2] += lift_root_height  # lift the humanoid slightly to avoid collisions with the ground

    return states.root_state_w, states.joint_pos, states.joint_vel

def reset_amp_buffer(amp_observation_buffer, num_amp_observations, env_ids: torch.Tensor, sampled_motion_ids: torch.Tensor, sampled_times: torch.Tensor, collect_references: callable):
    """ collect N>1 steps of observations and reset AMP observation buffer after resetting environments """
    num_samples = env_ids.shape[0]

    amp_observations = collect_references(num_samples, sampled_motion_ids, sampled_times)
    amp_observation_buffer[env_ids] = amp_observations.view(num_samples, num_amp_observations, -1)
    return amp_observation_buffer

def reset_nan_env(obs, reset_idx: callable, total_num_envs: int): # self.reset_idx is a bound method, no need to pass self
    nan_envs = find_nan(obs)
    nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
    # if nan_env_ids.shape[0] == total_num_envs:
    #     print("All environments are NaN, training process ends.")
    #     sys.exit(0)
    print(f"NaN detected in envs {nan_env_ids.tolist()}, resetting these envs.")
    reset_idx(nan_env_ids)
    return nan_env_ids