import torch
from isaaclab_tasks.direct.InteractionTracking.utils.func import printl
from isaaclab_tasks.direct.InteractionTracking.envs.utils.interaction import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.state import State
from isaaclab_tasks.direct.InteractionTracking.envs.utils.mimic import *
from isaaclab.assets import Articulation

class Reward:

    def __init__(self,
        num_envs: int,
        weights: dict,
        device: torch.device | str="cpu",
    ):
        self.num_envs = num_envs
        self.weights = weights
        self.device = device
        self.reset()

    def compute(self, **kwargs):
        for key, value in kwargs.items():
            value = value * self.weights[key]
            value = Reward.Func.zero_nan(value)

            setattr(self, key, value)
            self.total += value

        self.total = self.total / sum(self.weights.values()) if sum(self.weights.values()) > 0 else self.total

    def reset(self):
        self.total = torch.zeros([self.num_envs], device=self.device)
        for key in self.weights.keys():
            setattr(self, key, torch.zeros([self.num_envs], device=self.device))
    
    def print(self):
        reward_str = ", ".join([f"{key}: {getattr(self, key).mean().item():.4f}" for key in self.weights.keys()])
        printl(f"Rewards - {reward_str}, Total: {self.total.mean().item():.4f}")

    class Func:

        @staticmethod
        def tracking(robot: Articulation, gt: State, key_body_indexes: torch.Tensor | list | None=None) -> torch.Tensor:
            return DeepMimic.Reward.compute(robot, gt, key_body_indexes) 
        
        @staticmethod
        def energy_penalty(robot: Articulation) -> torch.Tensor:
            dof_forces = robot.data.applied_torque # [envs, dofs]
            dof_vel = robot.data.joint_vel # [envs, dofs] # dof_vel is already in rad/s (angular velocity)
            power = (-1) * torch.abs(torch.multiply(dof_forces, dof_vel)).mean(dim=-1) 
            power = torch.exp(power * 0.03)
            return power 
        
        @staticmethod
        def zero_nan(rewards: torch.Tensor):
            nan_envs = find_nan(rewards)
            if torch.any(nan_envs):
                nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
                print(f"NaN detected in rewards {nan_env_ids.tolist()}.")
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            return rewards
        