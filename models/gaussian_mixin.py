from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Normal


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)
import time

class GaussianMixin:
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
        is_train: bool = True,
    ) -> None:
        self.is_train = is_train

        self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._g_clip_actions:
            self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std
        self._g_log_std_max = max_log_std

        self._g_log_std = None
        self._g_num_samples = None
        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        # map from states/observations to mean actions and log standard deviations
        mean_actions, log_std, outputs = self.compute(inputs, role)

        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]

        # distribution
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._g_distribution.rsample()

        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions
        return actions if self.is_train else mean_actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        return self._g_log_std.repeat(self._g_num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        return self._g_distribution