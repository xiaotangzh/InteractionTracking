from skrl.models.torch import DeterministicMixin, Model
from models.gaussian_mixin import GaussianMixin
import torch
from torch import nn


class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space=1, hidden_dim=1024, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1)  
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {}  
    