import torch.nn.functional as F
import torch
import math
from typing import Union, TYPE_CHECKING
from isaaclab_tasks.direct.InteractionTracking.utils.math import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward_utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.visualize import *
from isaaclab_tasks.direct.InteractionTracking.utils.func import *
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.InteractionTracking.motions.motion_loader import MotionLoader

class Killer:
    def __init__(self,
        num_envs: int,
        max_episode_length: int,
        device: torch.device | str="cpu",
    ):
        self.num_envs = num_envs
        self.device = device

        self.truncated = torch.full((num_envs,), False, dtype=torch.bool, device=device)
        self.terminated = torch.full((num_envs,), False, dtype=torch.bool, device=device)

        self.max_episode_length = max_episode_length
    
    def reset(self, env_ids: torch.Tensor):
        self.truncated[env_ids] = False
        self.terminated[env_ids] = False
    
    def truncate(self, *envs: torch.Tensor):
        truncates = [self.truncated] + list(envs)
        self.truncated = torch.max(torch.stack(truncates, dim=0), dim=0).values

    def terminate(self, *envs: torch.Tensor):
        terminates = [self.terminated] + list(envs)
        self.terminated = torch.max(torch.stack(terminates, dim=0), dim=0).values

    def truncate_episode(self, episode_length_buf, bias: int = 0) -> torch.Tensor:
        return episode_length_buf >= self.max_episode_length - 1 - bias
    
    @staticmethod
    def terminate_by_height(body_heights, termination_heights):
        return body_heights < termination_heights

    @staticmethod
    def terminate_by_angle(ref_body_index, num_envs, target_direction, robot: "Articulation", angle) -> torch.Tensor:
        device = robot.data.body_pos_w.device
        target_direction, idx = get_unit_vector(target_direction, device)
        target_direction = target_direction.expand(num_envs, -1)
        current_direction = transform_quat_to_target_direction(robot.data.body_quat_w[:, ref_body_index], target_direction)
        angle_offset = current_direction[:, idx]  # [-1, 1] from opposite to same direction as target
        return (angle_offset < angle).view(-1)