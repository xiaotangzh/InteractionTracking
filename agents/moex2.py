from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from .base_agent import BaseAgent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.memories.torch import RandomMemory

from agents.utils.ppo import compute_gae

# fmt: off
# [start-config-dict-torch]
MOEX2_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class MOEX2(BaseAgent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(MOEX2_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        self.memory_1 = self.cfg["memory_1"]
        self.memory_2 = self.cfg["memory_2"]

        assert type(self.memory_1) == RandomMemory and type(self.memory_2) == RandomMemory

        self.memory_1.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
        self.memory_1.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory_1.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory_1.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory_1.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self.memory_1.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory_1.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory_1.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory_1.create_tensor(name="advantages", size=1, dtype=torch.float32)
        self.memory_1.create_tensor(name="expert_idx", size=1, dtype=torch.long)
        
        self.memory_2.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
        self.memory_2.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory_2.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory_2.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory_2.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self.memory_2.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory_2.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory_2.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory_2.create_tensor(name="advantages", size=1, dtype=torch.float32)
        self.memory_2.create_tensor(name="expert_idx", size=1, dtype=torch.long)

        self.rewards_1 = torch.zeros((self.memory_1.num_envs), dtype=torch.float32, device=self.device)
        self.last_obs_1 = torch.zeros((self.memory_1.num_envs, self.observation_space), dtype=torch.float32, device=self.device)
        self.selected_expert_idx_1 = torch.zeros((self.memory_1.num_envs), dtype=torch.long, device=self.device)

        self.rewards_2 = torch.zeros((self.memory_2.num_envs), dtype=torch.float32, device=self.device)
        self.last_obs_2 = torch.zeros((self.memory_2.num_envs, self.observation_space), dtype=torch.float32, device=self.device)
        self.selected_expert_idx_2 = torch.zeros((self.memory_2.num_envs), dtype=torch.long, device=self.device)

        # tensors sampled during training
        self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "rewards", "expert_idx"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        
        states_1, states_2 = states.chunk(2, dim=1)  

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            obs1 = self._state_preprocessor(states_1)
            obs2 = self._state_preprocessor(states_2)
            actions_1, log_prob_1, outputs = self.policy.act(
                {"states": obs1, 
                 "timestep": timestep, 
                 "rewards": self.rewards_1,
                 "last_expert_idx": self.selected_expert_idx_1,
                 "last_obs": self.last_obs_1,
            }, role="policy")
            self.selected_expert_idx_1 = outputs["selected_expert_idx"].clone()
            self.last_obs_1 = obs1.clone()

            actions_2, log_prob_2, outputs = self.policy.act(
                {"states": obs2, 
                 "timestep": timestep, 
                 "rewards": self.rewards_2,
                 "last_expert_idx": self.selected_expert_idx_2,
                 "last_obs": self.last_obs_2,
            }, role="policy")
            self.selected_expert_idx_2 = outputs["selected_expert_idx"].clone()
            self.last_obs_2 = obs2.clone()

            actions = torch.cat([actions_1, actions_2], dim=1)
            log_prob = torch.cat([log_prob_1, log_prob_2], dim=1)
            self._current_log_prob = log_prob

            if 'log' in outputs and outputs['log'] and timestep % 100 == 0:
                checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
                if os.path.exists(checkpoints_dir):
                    with open(os.path.join(checkpoints_dir, f'training_logs.txt'), 'w', encoding='utf-8') as f:
                        f.write(outputs['log'])

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        self.rewards_1 = infos["rewards_1"]
        self.rewards_2 = infos["rewards_2"]

        if self.memory_1 is not None and self.memory_2 is not None:
            self._current_next_states = next_states

            states_1, states_2 = states.chunk(2, dim=1)
            actions_1, actions_2 = actions.chunk(2, dim=1)
            next_states_1, next_states_2 = next_states.chunk(2, dim=1)
            current_log_prob_1, current_log_prob_2 = self._current_log_prob.chunk(2, dim=1)

            self._current_next_states_1 = next_states_1
            self._current_next_states_2 = next_states_2

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values_1, _, _ = self.value.act({"states": self._state_preprocessor(states_1)}, role="value")
                values_1 = self._value_preprocessor(values_1, inverse=True)

                values_2, _, _ = self.value.act({"states": self._state_preprocessor(states_2)}, role="value")
                values_2 = self._value_preprocessor(values_2, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                infos["rewards_1"] += self._discount_factor * values_1.view(-1) * infos["truncated_1"]
                infos["rewards_2"] += self._discount_factor * values_2.view(-1) * infos["truncated_2"]

            # storage transition in memory
            self.memory_1.add_samples(
                states=states_1,
                actions=actions_1,
                rewards=infos["rewards_1"].view(-1, 1),
                next_states=next_states_1,
                terminated=infos["terminated_1"].view(-1, 1),
                truncated=infos["truncated_1"].view(-1, 1),
                log_prob=current_log_prob_1,
                values=values_1,
            )
            self.memory_2.add_samples(
                states=states_2,
                actions=actions_2,
                rewards=infos["rewards_2"].view(-1, 1),
                next_states=next_states_2,
                terminated=infos["terminated_2"].view(-1, 1),
                truncated=infos["truncated_2"].view(-1, 1),
                log_prob=current_log_prob_2,
                values=values_2,
            )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor, 
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
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
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            next_states_preprocessed_1 = self._state_preprocessor(self._current_next_states_1.float())
            next_states_preprocessed_2 = self._state_preprocessor(self._current_next_states_2.float())
            last_values_1, _, _ = self.value.act(
                {"states": next_states_preprocessed_1,
                "expert_idx": self.policy.select_expert(next_states_preprocessed_1)}, role="value"
            )
            last_values_2, _, _ = self.value.act(
                {"states": next_states_preprocessed_2,
                "expert_idx": self.policy.select_expert(next_states_preprocessed_2)}, role="value"
            )
            self.value.train(True)
            last_values_1 = self._value_preprocessor(last_values_1, inverse=True)
            last_values_2 = self._value_preprocessor(last_values_2, inverse=True)

        values_1 = self.memory_1.get_tensor_by_name("values")
        last_values = last_values_1.clone()
        returns_1, advantages_1 = compute_gae(
            rewards=self.memory_1.get_tensor_by_name("rewards"),
            dones=self.memory_1.get_tensor_by_name("terminated") | self.memory_1.get_tensor_by_name("truncated"),
            values=values_1,
            next_values=last_values_1,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        self.memory_1.set_tensor_by_name("values", self._value_preprocessor(values_1, train=True))
        self.memory_1.set_tensor_by_name("returns", self._value_preprocessor(returns_1, train=True))
        self.memory_1.set_tensor_by_name("advantages", advantages_1)

        values_2 = self.memory_2.get_tensor_by_name("values")
        last_values = last_values_2.clone()
        returns_2, advantages_2 = compute_gae(
            rewards=self.memory_2.get_tensor_by_name("rewards"),
            dones=self.memory_2.get_tensor_by_name("terminated") | self.memory_2.get_tensor_by_name("truncated"),
            values=values_2,
            next_values=last_values_2,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory_2.set_tensor_by_name("values", self._value_preprocessor(values_2, train=True))
        self.memory_2.set_tensor_by_name("returns", self._value_preprocessor(returns_2, train=True))
        self.memory_2.set_tensor_by_name("advantages", advantages_2)

        # sample mini-batches from memory
        sampled_batches_1 = self.memory_1.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        sampled_batches_2 = self.memory_2.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        sampled_batches = sampled_batches_1 + sampled_batches_2

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_adapter_usage_loss = 0  

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_rewards,
                sampled_expert_idx
            ) in sampled_batches:
                sampled_batch_size = sampled_states.shape[0]

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # note: if use multiple std, next_log_prob must be computed from expert selected when collecting samples to buffer
                    _, next_log_prob, policy_outputs = self.policy.act(
                        {"states": sampled_states, 
                         "taken_actions": sampled_actions,
                         "selected_experts": sampled_expert_idx,
                         }, role="policy"
                    ) 

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # extra loss
                    adapter_usage = policy_outputs["adapter_usage"]
                    adapter_usage_loss = (-torch.log(adapter_usage + 1e-6).mean() / 30) if adapter_usage is not None else torch.tensor(0.0, device=self.device)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + adapter_usage_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_adapter_usage_loss += adapter_usage_loss.item() 
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Adapter Usage loss", cumulative_adapter_usage_loss / (self._learning_epochs * self._mini_batches)) 
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        self.track_data("Policy / KL divergence", torch.stack(kl_divergences).mean().item())

        actions = self.memory_1.get_tensor_by_name("actions")
        self.track_data("Policy / Action mean", actions.mean().item())
        self.track_data("Policy / Action std", actions.std().item())

        log_probs = self.memory_1.get_tensor_by_name("log_prob")
        self.track_data("Policy / Log prob mean", log_probs.mean().item())

        advantages = self.memory_1.get_tensor_by_name("advantages")
        self.track_data("Stats / Advantage mean", advantages.mean().item())
        self.track_data("Stats / Advantage std", advantages.std().item())

        values = self.memory_1.get_tensor_by_name("values")
        self.track_data("Stats / Value mean", values.mean().item())
        self.track_data("Stats / Value std", values.std().item())

        returns = self.memory_1.get_tensor_by_name("returns")
        self.track_data("Stats / Return mean", returns.mean().item())
        self.track_data("Stats / Return std", returns.std().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.track_data("Learning / Grad norm", total_norm ** 0.5)
