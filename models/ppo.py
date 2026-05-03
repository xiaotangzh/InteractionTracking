from skrl.models.torch import DeterministicMixin, Model
from models.gaussian_mixin import GaussianMixin
import torch
from torch import nn
from torch.nn import functional as F
from isaaclab_tasks.direct.PhysicsProject.utils.func import printl
from utils.math import reparameterize
from utils.func import disable_grads, enable_grads, set_grads
from agents.utils.utils import load_from_agent_checkpoint

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, hidden_dim=1024, initial_log_std=-2.9, fixed_log_std=False, device=None, num_layers=4, is_train=True, reduction: str="sum", clip_actions: bool=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, 
                              clip_actions=clip_actions,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                              is_train=is_train,
                              reduction=reduction
                              )

        # 10 layers
        if num_layers == 10:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, action_space)
            )
        
        # 7 layers
        elif num_layers == 7:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, action_space)
            )

        # 4 layers
        elif num_layers == 4:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, action_space)
            )

        if fixed_log_std: 
            self.log_std_parameter = torch.tensor(initial_log_std, device=device)
        else: 
            self.log_std_parameter = nn.Parameter(torch.full((action_space,), fill_value=initial_log_std)) 

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), self.log_std_parameter, {}

    
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space=1, hidden_dim=1024, device=None, num_layers=4):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # 10 layers
        if num_layers == 10:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            )
        
        # 7 layers
        elif num_layers == 7:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            )

        # 4 layers
        elif num_layers == 4:
            self.net = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            )

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {} 


class MixturePolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, num_experts=4, hidden_dim=1024, initial_log_std=-2.9, fixed_log_std=True, hard_weights: bool=False, is_train: bool=True, device=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                               clip_actions=False,
                               clip_log_std=True,
                               min_log_std=-20.0,
                               max_log_std=2.0,
                               is_train=is_train)

        self.num_experts = num_experts
        self.hard_weights = hard_weights

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(observation_space, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, action_space)
            ) for _ in range(num_experts)
        ])

        self.gating = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        if fixed_log_std:
            self.log_std_parameter = torch.tensor(initial_log_std, device=device)
        else:
            self.log_std_parameter = nn.Parameter(torch.full((action_space,), fill_value=initial_log_std))

    def compute(self, inputs, role):
        obs = inputs["states"]  # shape: (B, obs_dim)

        # Get expert outputs
        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=1)  # shape: (B, num_experts, action_dim)

        # Get expert weights from gating network
        weights = self.gating(obs)  # shape: (B, num_experts)

        if self.hard_weights:
            weights = F.gumbel_softmax(weights, tau=0.1, hard=True)

        # Weighted sum over experts
        mean_actions = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)  # shape: (B, action_dim)

        # expert_usage = weights.sum(dim=0) / mean_actions.shape[0]
        # printl("Usage", expert_usage)

        return mean_actions, self.log_std_parameter, {}
    

class MixtureValue(DeterministicMixin, Model):
    def __init__(self, observation_space, output_dim, num_experts=4, hidden_dim=1024, device=None):
        Model.__init__(self, observation_space, output_dim, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(observation_space, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(hidden_dim / 2)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim / 2), output_dim)
            ) for _ in range(num_experts)
        ])

        self.gating = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), num_experts),
        )

    def compute(self, inputs, role):
        obs = inputs["states"] 

        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=-2)  
        weights = nn.functional.softmax(self.gating(obs), dim=-1)

        outputs = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=-2) 
        return outputs, {} 