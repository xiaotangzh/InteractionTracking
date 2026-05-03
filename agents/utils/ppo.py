from ast import List
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.InteractionTracking.agents.base_agent import BaseAgent  # avoid circular import error

def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Generalized Advantage Estimator (GAE)

    :param rewards: Rewards obtained by the agent
    :type rewards: torch.Tensor
    :param dones: Signals to indicate that episodes have ended
    :type dones: torch.Tensor
    :param values: Values obtained by the agent
    :type values: torch.Tensor
    :param next_values: Next values obtained by the agent
    :type next_values: torch.Tensor
    :param discount_factor: Discount factor
    :type discount_factor: float
    :param lambda_coefficient: Lambda coefficient
    :type lambda_coefficient: float

    :return: Generalized Advantage Estimator
    :rtype: torch.Tensor
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def compute_policy_loss(
    agent: "BaseAgent",
    next_log_prob: torch.Tensor,
    sampled_log_prob: torch.Tensor,
    sampled_advantages: torch.Tensor
) -> torch.Tensor:
    ratio = torch.exp(next_log_prob - sampled_log_prob)
    surrogate = sampled_advantages * ratio
    surrogate_clipped = sampled_advantages * torch.clip(
        ratio, 1.0 - agent._ratio_clip, 1.0 + agent._ratio_clip
    )

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
    return policy_loss

def compute_value_loss(
    agent: "BaseAgent",
    value: "Model",
    sampled_states: torch.Tensor,
    sampled_values: torch.Tensor,
    sampled_returns: torch.Tensor,
) -> torch.Tensor:
    predicted_values, _, _ = value.act({"states": sampled_states}, role="value")

    if agent._clip_predicted_values:
        predicted_values = sampled_values + torch.clip(
            predicted_values - sampled_values, min=-agent._value_clip, max=agent._value_clip
        )
    value_loss = agent._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
    return value_loss
