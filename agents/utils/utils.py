from ast import List
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from packaging import version
from skrl import logger
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.InteractionTracking.agents.base_agent import BaseAgent  # avoid circular import error

def compute_entropy_loss(
    agent: "BaseAgent",
    policy: "Model",
):
    if agent._entropy_loss_scale:
        entropy_loss = agent._entropy_loss_scale * policy.get_entropy(role="policy").mean()
    else:
        entropy_loss = 0
    return entropy_loss

def compute_kl(
    next_log_prob: torch.Tensor,
    sampled_log_prob: torch.Tensor,
):
    with torch.no_grad():
        ratio = next_log_prob - sampled_log_prob
        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
    return kl_divergence

def sample_mini_batches(
    agent: "BaseAgent",
    tensors_names
) -> list:
    sampled_batches = agent.memory_1.sample_all(names=tensors_names, mini_batches=agent._mini_batches)
    return sampled_batches

def sample_mini_batches_for_discriminator(
    agent: "BaseAgent",
    tensors_names,
    motion_dataset,
    reply_buffer,
    sampled_batches,
    replay_tensor_name,
) -> tuple[list, list]:
    sampled_motion_batches = motion_dataset.sample(
        names=["states"], batch_size=agent.memory_1.memory_size * agent.memory_1.num_envs, mini_batches=agent._mini_batches
    )
    if len(reply_buffer):
        sampled_replay_batches = reply_buffer.sample(
            names=["states"],
            batch_size=agent.memory_1.memory_size * agent.memory_1.num_envs,
            mini_batches=agent._mini_batches,
        )
    else:
        sampled_replay_batches = [[batches[tensors_names.index(replay_tensor_name)]] for batches in sampled_batches]
    
    return sampled_motion_batches, sampled_replay_batches


def update_learning_rate(
    agent: "BaseAgent",
    scheduler,
    kl_divergences,
    config,
):
    if isinstance(scheduler, KLAdaptiveLR):
        kl = torch.tensor(kl_divergences, device=agent.device).mean()
        # reduce (collect from all workers/processes) KL in distributed runs
        if config.torch.is_distributed:
            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
            kl /= config.torch.world_size
        scheduler.step(kl.item())
    else:
        scheduler.step()

def load_from_agent_checkpoint(path: str, module_name: str, device = None) -> dict:
    if version.parse(torch.__version__) >= version.parse("1.13"):
        modules = torch.load(path, map_location=device, weights_only=False)
    else:
        modules = torch.load(path, map_location=device)

    if not isinstance(modules, dict):
        logger.error("The loaded checkpoint is not a dictionary of modules.")

    if module_name not in modules:
        logger.warning(f"Module '{module_name}' not found in checkpoint.")

    state_dict = modules[module_name]
    return state_dict