from skrl.models.torch import DeterministicMixin, Model
from models.gaussian_mixin_moe import GaussianMixin
import torch
from torch import nn
from torch.nn import functional as F
import math
from isaaclab_tasks.direct.PhysicsProject.utils.func import printl
from utils.math import reparameterize
from utils.func import disable_grads, enable_grads, set_grads, format_dict_to_string
from agents.utils.utils import load_from_agent_checkpoint
import time

class MoE_Unfreeze(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, hidden_dim=1024, device=None,
                 initial_log_std=-2.9, use_multiple_log_std=False,
                 eval_mode=False,
                 num_experts=4, init_expert=1, 
                 print_log=False,
                 reward_check_interval=20000,
                 min_reward_growth=0.005,
                 double_reward_check_interval=False,
                 zero_init_new_expert=False,
                 copy_parameters_to_new_expert=False,
                 reinit_std_when_activate_new_expert=False,
                 add_noise_when_copy_parameters=False,
                 selection_strategy='default',
                 load_first_expert: str | None = None,
                 load_reward_predictor: str | None = None,
                 load_log_std_parameter: str | None = None,
                 init_adapters_as: str = "zero",
                 adapter_usage_loss: bool = True,
        ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                               clip_actions=False,
                               clip_log_std=True,
                               min_log_std=-20.0,
                               max_log_std=2.0)
        self.eval_mode = eval_mode
        self.input_dim, self.output_dim = observation_space, action_space
        self.timestep = 0
        self.mean_reward = 0
        self.print_log = print_log
        self.num_experts = num_experts
        self.num_active_experts = init_expert
        self.selection_strategy = selection_strategy
        self.zero_init_new_expert = zero_init_new_expert
        self.copy_parameters_to_new_expert = copy_parameters_to_new_expert
        
        self.add_noise_when_copy_parameters = add_noise_when_copy_parameters

        # Section: Reward History
        self.reward_history = []
        self.reward_check_interval = reward_check_interval
        self.min_reward_growth = min_reward_growth
        self.double_reward_check_interval = double_reward_check_interval

        # Section: Experts and Gating Network
        experts = [
            nn.Sequential(
                nn.Linear(observation_space, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, int(hidden_dim / 2)), # --> + adapted outputs
                nn.LeakyReLU(),
                nn.Linear(int(hidden_dim / 2), action_space)
            ) for _ in range(num_experts)
        ]
        self.experts = nn.ModuleList(experts)
        self.expert_layers = [] 
        for expert in self.experts:
            layers = nn.ModuleList(expert.children())
            self.expert_layers.append(layers)

        self.gating = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # Section: PNN
        self.adapters = nn.ModuleDict()
        for i in range(1, self.num_experts): 
            for j in range(i):  
                self.adapters[f"adapter_{j}_to_{i}_layer2"] = nn.Sequential(
                    nn.Linear(int(hidden_dim / 2), int(hidden_dim / 2)),
                    nn.LeakyReLU()
                )
        for adapter in self.adapters.values():
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    if init_adapters_as == "zero":
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
                    elif init_adapters_as == "one":
                        nn.init.eye_(layer.weight)
                        nn.init.zeros_(layer.bias)

        self.init_adapters_as = init_adapters_as
        self.adapter_usage_loss = adapter_usage_loss

        # Section: Log Std Parameter
        self.initial_log_std = initial_log_std
        self.use_multiple_log_std = use_multiple_log_std
        self.reinit_std_when_activate_new_expert = reinit_std_when_activate_new_expert
        if self.use_multiple_log_std:
            self.log_std_parameter = nn.ParameterList([
                nn.Parameter(torch.full((action_space,), fill_value=initial_log_std))
                for _ in range(self.num_experts)
            ])
        else:
            self.log_std_parameter = nn.Parameter(torch.full((action_space,), fill_value=initial_log_std))

        # Section: Reward Predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), self.num_experts)
        )
        self.reward_predictor[-1].bias.data.fill_(-3.0)
        self.reward_predictor_lr = 1e-3

        # Section: Load Parameters
        if load_first_expert is not None:
            state_dict = load_from_agent_checkpoint(load_first_expert, 'policy')
            self.load_first_expert(state_dict)
            self.activate_new_expert(self.num_active_experts+1)  

        if load_reward_predictor is not None:
            state_dict = load_from_agent_checkpoint(load_reward_predictor, 'policy')
            self.load_reward_predictor(state_dict)

        if load_log_std_parameter is not None:
            state_dict = load_from_agent_checkpoint(load_log_std_parameter, 'policy')
            self.load_log_std_parameter(state_dict)
    
    @property
    def current_expert_idx(self):
        return self.num_active_experts - 1

    @property
    def prev_expert_idx(self):
        return self.current_expert_idx - 1 if self.current_expert_idx > 0 else 0

    @property
    def active_expert_mask(self):
        expert_mask = torch.zeros(self.num_experts, device=self.device)
        expert_mask[self.current_expert_idx] = 1.0
        return expert_mask

    @property
    def inactive_expert_mask(self):
        expert_mask = torch.ones(self.num_experts, device=self.device)
        expert_mask[self.current_expert_idx] = 0.0
        return expert_mask

    def active_expert_batch_mask(self, selected_expert_idx):
        return (selected_expert_idx == self.current_expert_idx).float()
    
    def training_samples_proportion(self, selected_expert_idx):
        current_mask = self.active_expert_batch_mask(selected_expert_idx)  # tensor of 0/1 floats, shape [batch_size]
        try:
            current_prop = float(current_mask.mean().item())
        except Exception:
            current_prop = float(current_mask.sum().item()) / max(1, getattr(self, 'batch_size', 1))
        return current_prop
    
    def lateral_forward(self, obs):
        batch = obs.shape[0]
        device = obs.device
        expert_outputs = torch.zeros(batch, self.num_experts, self.output_dim, device=device)
        adapter_outputs = torch.zeros(batch, self.output_dim // 2, device=device)
        adapter_usage = torch.zeros(batch, device=device)

        for i, expert in enumerate(self.experts):
            # current expert: include lateral adapters from previous experts (if any)
            if self.current_expert_idx == 0:
                expert_outputs[:, i, :] = expert(obs)
            else:
                layers = list(expert.children())
                # compute current expert hidden path
                h0 = layers[0](obs)
                h1 = layers[1](h0)
                h2_in = layers[2](h1)

                # collect adapted outputs from previous experts (detached)
                adapted_outputs_2 = []
                for prev_idx in range(self.current_expert_idx):
                    prev_layers = list(self.experts[prev_idx].children())
                    prev_h2 = prev_layers[3](prev_layers[2](prev_layers[1](prev_layers[0](obs)))).detach()
                    adapted = self.adapters[f"adapter_{prev_idx}_to_{self.current_expert_idx}_layer2"](prev_h2)
                    adapted_outputs_2.append(adapted)

                if len(adapted_outputs_2) > 0:
                    total_adapted = torch.sum(torch.stack(adapted_outputs_2), dim=0)
                    adapter_outputs = total_adapted
                    combined = layers[2](h1) + total_adapted
                    h2 = layers[3](combined)
                    adapter_usage = self.compute_adapter_usage(total_adapted, layers[2](h1))
                else:
                    h2 = layers[3](layers[2](h1))
                    adapter_usage = torch.zeros(batch, device=device)

                expert_outputs[:, i, :] = layers[4](h2) + expert_outputs[:, i-1, :].detach()

        return expert_outputs, adapter_outputs, adapter_usage

    def compute_adapter_usage(self, adapter_outputs, expert_hidden_outputs, eps=1e-6) -> torch.Tensor:
        usage = (adapter_outputs.norm(dim=-1) / (adapter_outputs.norm(dim=-1) + (expert_hidden_outputs.norm(dim=-1) + eps))).clamp(max=1.0)
        return usage

    def compute(self, inputs, role):
        obs = inputs["states"]  # shape: (B, obs_dim)
        self.batch_size = obs.shape[0]
        if 'timestep' in inputs:
            self.timestep = inputs['timestep']

        self.update_gradient_status()

        # Expert selection
        if "taken_actions" in inputs: # avoid contradiction of log_std in different timesteps
            assert "selected_experts" in inputs, "When providing 'taken_actions', 'selected_experts' must also be provided."
            selected_expert_idx = inputs["selected_experts"].view(-1)
        else:
            selected_expert_idx = self.select_expert(obs)

        # Get expert outputs
        expert_outputs, adapter_outputs, adapter_usage = self.lateral_forward(obs) # shape: (B, num_experts, action_dim) 

        # Combine expert outputs
        one_hot_weights = F.one_hot(selected_expert_idx, num_classes=self.num_experts).float() 
        mean_actions = torch.sum(expert_outputs * one_hot_weights.unsqueeze(-1), dim=1)  # shape: (B, action_dim)

        if inputs.get('rewards', None) is not None and inputs.get('last_obs', None) is not None and inputs.get('last_expert_idx', None) is not None:
            last_rewards = inputs["rewards"]
            last_obs = inputs["last_obs"]
            last_selected_expert_idx = inputs["last_expert_idx"]
            self.mean_reward = last_rewards.mean().item()

            self.update_reward_predictor(last_obs, last_rewards, last_selected_expert_idx)

            self.reward_history.append(self.mean_reward)
            if len(self.reward_history) > self.reward_check_interval * 2:
                self.reward_history = self.reward_history[-self.reward_check_interval:]

            if self.check_confidence_growth_stagnation():
                self.activate_new_expert(self.num_active_experts + 1)
        
        log = {}
        if self.timestep % 100 <= 10:
            log['Timestep'] = self.timestep
            log['Training Expert'] = self.current_expert_idx
            log['Selection Strategy'] = self.selection_strategy
            log['Expert Samples Proportion'] = self.training_samples_proportion(selected_expert_idx)
            log['Expert Usage'] = one_hot_weights.sum(dim=0) / self.batch_size
            log['Adapter Usage'] = adapter_usage.mean().item() 
            log['Average Reward'] = self.mean_reward
            log['Estimated Rewards'] = torch.sigmoid(self.reward_predictor(obs)).mean(dim=0).detach().cpu().numpy()
            log['Reward Growth'] = self.get_confidence_growth_info()
            log['Expert Statistics'] = self.expert_statistics()
            log['Log Std Statistics'] = self.log_std_statistics()
            log['Adapter Statistics'] = self.adapter_statistics()
        
            if self.print_log: print(format_dict_to_string(log))
        
        if self.eval_mode:
            printl(f"Expert Usage: {one_hot_weights.sum(dim=0) / self.batch_size}")

        return mean_actions, \
            self.select_log_std(selected_expert_idx), \
            {'use_multiple_log_std': self.use_multiple_log_std,
             'selected_expert_idx': selected_expert_idx,
             'adapter_usage': adapter_usage if self.adapter_usage_loss else None,
             'active_expert_batch_mask': self.active_expert_batch_mask(selected_expert_idx),
             'log': format_dict_to_string(log) if log else None}

    def select_log_std(self, selected_expert_idx: torch.Tensor):
        if self.use_multiple_log_std:
            # different log std for different samples
            if isinstance(selected_expert_idx, torch.Tensor):
                all_log_stds = torch.stack(list(self.log_std_parameter))   # [num_experts, action_dim]
                return all_log_stds[selected_expert_idx]                   # [batch, action_dim]
        else:
            return self.log_std_parameter

    def select_expert(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = obs.shape[0]

        if self.eval_mode:
            logits = self.reward_predictor(obs)  # (B, num_experts)
            estimated_rewards = torch.sigmoid(logits)  # (B, num_experts) 
            masked = estimated_rewards.clone()
            active_mask = torch.arange(self.num_experts, device=obs.device) < self.num_active_experts
            masked[:, ~active_mask] = -1.0
            max_confidence_expert = masked.argmax(dim=-1)  # (B,)
            return max_confidence_expert
        
        if self.selection_strategy == 'default':
            return torch.full((batch_size,), fill_value=(self.current_expert_idx), dtype=torch.long, device=self.device)
        
        elif self.selection_strategy == 'reward':
            logits = self.reward_predictor(obs)  # (B, num_experts)
            estimated_rewards = torch.sigmoid(logits)  # (B, num_experts) 
            max_confidence_expert = estimated_rewards.argmax(dim=-1)  # (B,)
            experts_average_reward = estimated_rewards.mean(dim=0)  # Average across all samples 
            
            if self.current_expert_idx == 0:
                selected_expert_idx = torch.full((batch_size,), fill_value=(self.current_expert_idx), dtype=torch.long, device=self.device)

            else:
                selected_expert_idx = max_confidence_expert.clone()

                if experts_average_reward[self.current_expert_idx] < experts_average_reward[self.prev_expert_idx]:
                    # Let current expert get at least same number of samples as previous expert
                    # Sort by confidence (achieves natural 50%, 25%, 12.5%... allocation ratio)
                    batch_size = obs.shape[0]
                    current_samples = torch.arange(batch_size, device=obs.device)  
                    
                    for expert_idx in range(self.num_active_experts):
                        if len(current_samples) == 0:
                            break
                        
                        expert_confidences = estimated_rewards[current_samples, expert_idx]
                        sorted_conf, sort_indices = torch.sort(expert_confidences, descending=True)
                        
                        # Current expert keeps top 50% (high confidence), remaining 50% passed to next expert
                        if expert_idx < self.current_expert_idx:  # Not the last expert
                            keep_count = len(current_samples) // 2  # Keep top 50%
                            if keep_count == 0 and len(current_samples) > 0:
                                keep_count = 1  # Keep at least 1 sample
                            
                            # Samples assigned to current expert
                            keep_indices = sort_indices[:keep_count]
                            assigned_samples = current_samples[keep_indices]
                            selected_expert_idx[assigned_samples] = expert_idx
                            
                            # Remaining samples passed to next expert
                            transfer_indices = sort_indices[keep_count:]
                            current_samples = current_samples[transfer_indices]
                        else:
                            # Last expert receives all remaining samples
                            selected_expert_idx[current_samples] = expert_idx
                            current_samples = torch.tensor([], device=obs.device, dtype=torch.long)
                    
            return selected_expert_idx
        
        elif self.selection_strategy == 'inverse':
            # 1) Predict per-sample rewards and restrict to ACTIVE experts
            logits = self.reward_predictor(obs)                     # (B, E)
            estimated = torch.sigmoid(logits)                       # (B, E) in (0,1)
            active_mask = torch.arange(self.num_experts, device=obs.device) < self.num_active_experts
            masked = estimated.clone()
            masked[:, ~active_mask] = -1.0                          # exclude inactive from argmax

            # 2) Start from normal hard routing (argmax among active experts)
            selected_expert_idx = masked.argmax(dim=-1)             # (B,)
            current_best_ratio = (selected_expert_idx == self.current_expert_idx).float().mean()

            # 3) If there is a previous expert, compute inverse ratio and re-route
            if self.current_expert_idx > 0:
                prev_idx = self.prev_expert_idx
                cur_idx  = self.current_expert_idx

                # Average estimated rewards for the two neighboring experts
                avg_prev = estimated[:, prev_idx].mean()
                avg_cur  = estimated[:, cur_idx].mean()

                # Ratio = avg_cur / avg_prev (e.g., 0.2/0.6=0.333 -> route 33%)
                eps = 1e-8
                # ratio = (avg_cur / (avg_prev + eps)).clamp(min=0.0, max=1.0)
                ratio = torch.max(torch.tensor(0.1, device=self.device), current_best_ratio)

                # Find samples currently assigned to prev expert
                prev_mask = (selected_expert_idx == prev_idx)
                prev_indices = torch.nonzero(prev_mask, as_tuple=False).flatten()

                if prev_indices.numel() > 0 and ratio.item() > 0.0:
                    # Sort those samples by how confident prev expert is (ascending),
                    # then take the weakest tail proportion (ratio) and send to current.
                    prev_conf = estimated[prev_indices, prev_idx]   # confidences under prev
                    sort_vals, sort_idx = torch.sort(prev_conf, descending=False)  # weakest first

                    transfer_count = int(torch.floor(ratio * prev_indices.numel()).item())
                    if transfer_count == 0 and ratio.item() > 0.0:  # ensure at least 1 if ratio>0
                        transfer_count = 1

                    if transfer_count > 0:
                        to_transfer = prev_indices[sort_idx[:transfer_count]]
                        selected_expert_idx[to_transfer] = cur_idx

            return selected_expert_idx

        elif self.selection_strategy == 'random':
            # 1) Predict per-sample rewards and restrict to ACTIVE experts
            logits = self.reward_predictor(obs)                     # (B, E)
            estimated = torch.sigmoid(logits)                       # (B, E) in (0,1)
            active_mask = torch.arange(self.num_experts, device=obs.device) < self.num_active_experts
            masked = estimated.clone()
            masked[:, ~active_mask] = -1.0                          # exclude inactive from argmax

            # 2) Start from normal hard routing (argmax among active experts)
            selected_expert_idx = masked.argmax(dim=-1)             # (B,)

            # 3) If there is a previous expert, randomly transfer a proportion of prev's samples to current
            if self.current_expert_idx > 0:
                prev_idx = self.prev_expert_idx
                cur_idx  = self.current_expert_idx

                # Average estimated rewards for the two neighboring experts
                avg_prev = estimated[:, prev_idx].mean()
                avg_cur  = estimated[:, cur_idx].mean()

                eps = 1e-8
                ratio = (avg_cur / (avg_prev + eps)).clamp(min=0.0, max=1.0)

                # Find samples currently assigned to prev expert
                prev_mask = (selected_expert_idx == prev_idx)
                prev_indices = torch.nonzero(prev_mask, as_tuple=False).flatten()

                if prev_indices.numel() > 0 and ratio.item() > 0.0:
                    # Decide how many to transfer (at least 1 if ratio>0)
                    transfer_count = int(torch.floor(ratio * prev_indices.numel()).item())
                    if transfer_count == 0 and ratio.item() > 0.0:
                        transfer_count = 1

                    # Randomly pick transfer_count samples from prev_indices
                    perm = torch.randperm(prev_indices.numel(), device=obs.device)
                    to_transfer = prev_indices[perm[:transfer_count]]
                    selected_expert_idx[to_transfer] = cur_idx

            return selected_expert_idx

        else:
            raise NotImplementedError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def update_gradient_status(self):
        for i in range(self.num_experts):
            if i == (self.current_expert_idx):
                enable_grads(self.experts[i])
            else:
                disable_grads(self.experts[i])

        for adapter_name, adapter in self.adapters.items():
            if f"to_{self.current_expert_idx}" in adapter_name:
                enable_grads(adapter)
            else:
                disable_grads(adapter)
        
        if self.use_multiple_log_std:
            for i in range(self.num_experts):
                if i == self.current_expert_idx:
                    self.log_std_parameter[i].requires_grad_(True)
                else:
                    self.log_std_parameter[i].requires_grad_(False)
    
    def check_confidence_growth_stagnation(self):
        if len(self.reward_history) < self.reward_check_interval:
            return False  
        
        recent_history = self.reward_history[-self.reward_check_interval:]
        
        mid_point = len(recent_history) // 2
        early_half_avg = sum(recent_history[:mid_point]) / mid_point
        late_half_avg = sum(recent_history[mid_point:]) / (len(recent_history) - mid_point)
        
        confidence_growth = late_half_avg - early_half_avg
        is_stagnant = confidence_growth < self.min_reward_growth
        
        if is_stagnant:
            print(f"[Growth Check] Expert {self.current_expert_idx} confidence growth: {confidence_growth:.4f} < {self.min_reward_growth}")
            print(f"Early half avg: {early_half_avg:.4f}, Late half avg: {late_half_avg:.4f}")
        
        return is_stagnant
    
    def get_confidence_growth_info(self):
        if len(self.reward_history) < 100: 
            return "insufficient data (< 100 steps)"
        
        # If insufficient data, use all available steps
        available_steps = min(len(self.reward_history), self.reward_check_interval)
        recent_history = self.reward_history[-available_steps:]
        
        # Compute average confidence of first half and second half
        mid_point = len(recent_history) // 2
        early_half_avg = sum(recent_history[:mid_point]) / mid_point
        late_half_avg = sum(recent_history[mid_point:]) / (len(recent_history) - mid_point)
        
        confidence_growth = late_half_avg - early_half_avg
        actual_steps = len(recent_history)
        
        if confidence_growth >= 0:
            growth_str = f"+{confidence_growth:.4f}"
        else:
            growth_str = f"{confidence_growth:.4f}"
        
        return f"{growth_str} (over {actual_steps}/{self.reward_check_interval} steps, threshold: {self.min_reward_growth})"
    
    def activate_new_expert(self, target_active_experts):
        if target_active_experts <= self.num_active_experts:
            return
            
        while self.num_active_experts < target_active_experts and self.num_active_experts < self.num_experts:
            new_expert_idx = self.current_expert_idx + 1

            if self.copy_parameters_to_new_expert:
                self.copy_parameters(self.current_expert_idx, new_expert_idx)
                
                if self.add_noise_when_copy_parameters:
                    self.add_noise_to_expert(new_expert_idx)
                    print(f"Added noise to expert {new_expert_idx} after parameter copying")
            
            if self.reinit_std_when_activate_new_expert:
                if not self.use_multiple_log_std:
                    self.log_std_parameter.data.fill_(self.initial_log_std)
                else:
                    self.log_std_parameter[new_expert_idx].data.fill_(self.initial_log_std)
            
            self.num_active_experts += 1
            self.reward_history = []
        
        print(f"Activated new expert{self.current_expert_idx}.")
        self.update_gradient_status()
        if self.double_reward_check_interval: self.reward_check_interval *= 2

    def copy_parameters(self, src_idx: int, dst_idx: int):
        src_expert_network = self.experts[src_idx]
        dst_expert_network = self.experts[dst_idx]
        for src_param, dst_param in zip(src_expert_network.parameters(), dst_expert_network.parameters()):
        # temp: copy to specific linear, 2 = 1 layer * (weight + bias)
        # for src_param, dst_param in zip(list(src_expert_network.parameters())[:2], list(dst_expert_network.parameters())[:2]): 
            dst_param.data.copy_(src_param.data)

        # Zero initialize the layers of the new expert
        linear_count = 0
        for layer in list(dst_expert_network.children()): 
            if isinstance(layer, nn.Linear): 
                if linear_count >= 1: # test: 
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                linear_count += 1
                # if linear_count >= 2:
                #     break
        
        if self.use_multiple_log_std and not self.reinit_std_when_activate_new_expert:
            self.log_std_parameter[dst_idx].data.copy_(self.log_std_parameter[src_idx].data)

    def add_noise_to_expert(self, expert_idx: int, noise_scale: float = 0.01):
        expert_network = self.experts[expert_idx]
        with torch.no_grad():
            for param in expert_network.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.data.add_(noise)
        
            log_std_noise_scale = noise_scale * 0.1
            if self.use_multiple_log_std:
                # add noise to the specific expert's log_std vector
                self.log_std_parameter[expert_idx].data.add_(torch.randn_like(self.log_std_parameter[expert_idx].data) * log_std_noise_scale)
            else:
                # single shared log_std: perturb the whole vector a little
                self.log_std_parameter.data.add_(torch.randn_like(self.log_std_parameter.data) * log_std_noise_scale)


    def update_reward_predictor(
        self, 
        last_obs: torch.Tensor | None, 
        last_rewards: torch.Tensor | None, 
        last_selected_expert_idx: torch.Tensor | None, 
        **kwargs
    ):

        if last_rewards is None or last_obs is None or last_selected_expert_idx is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.enable_grad():
            enable_grads(self.reward_predictor)
            last_obs.requires_grad_(True)

            one_hot_weights = F.one_hot(last_selected_expert_idx, num_classes=self.num_experts).float()
            one_hot_weights *= self.active_expert_mask.unsqueeze(0) # only the training expert has weight
            confidence_scores = torch.sigmoid(self.reward_predictor(last_obs))  # (B, num_experts))
            selected_confidences = (confidence_scores * one_hot_weights).sum(dim=-1)
            loss = F.mse_loss(selected_confidences, last_rewards.view(-1).to(self.device))
            
            for param in self.reward_predictor.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            loss.backward()
        
        with torch.no_grad():
            for param in self.reward_predictor.parameters():
                if param.grad is not None:
                    param.data -= self.reward_predictor_lr * param.grad
        
        disable_grads(self.reward_predictor)
        return f"Loss: {loss.item():.4f}, LR: {self.reward_predictor_lr}"

    def expert_statistics(self):
        stats = []
        for expert_idx in range(self.num_experts):
            stats.append(f"\n\tExpert {expert_idx}:")
            expert = self.experts[expert_idx]
            for name, param in expert.named_parameters():
                stats.append(f"\n\t\t{name}: \tmean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}, requires_grad={param.requires_grad}")
        return "".join(stats)

    def adapter_statistics(self):
        stats = []
        for adapter_name, adapter in self.adapters.items():
            stats.append(f"\n\tAdapter {adapter_name}:")
            for name, param in adapter.named_parameters():
                stats.append(f"\n\t\t{name}: \tmean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}, requires_grad={param.requires_grad}")
        return "".join(stats)
    
    def log_std_statistics(self):
        if self.use_multiple_log_std:
            stats = []
            for idx, log_std_param in enumerate(self.log_std_parameter):
                log_std = log_std_param.data
                stats.append(f"\n\tLog Std Expert {idx}: \tmean={log_std.mean().item():.4f}, std={log_std.std().item():.4f}, requires_grad={log_std_param.requires_grad}")
            return "".join(stats)
        else:
            log_std = self.log_std_parameter.data
            return f"\n\tLog Std: \tmean={log_std.mean().item():.4f}, std={log_std.std().item():.4f}, requires_grad={self.log_std_parameter.requires_grad}"

    def print_gradients(self):
        for idx in range(self.num_experts):
            print(f"Expert {idx} gradients:")
            for name, param in self.experts[idx].named_parameters():
                if param.grad is not None:
                    print(f"  {name}: grad_norm={param.grad.norm().item():.4f}")
                else:
                    print(f"  {name}: grad=None")

        for adapter_name in self.adapters.keys():
            print(f"{adapter_name} gradients:")
            for name, param in self.adapters[adapter_name].named_parameters():
                if param.grad is not None:
                    print(f"  {name}: grad_norm={param.grad.norm().item():.4f}")
                else:
                    print(f"  {name}: grad=None")
        
        for idx in range(self.num_experts):
            print(f"Log Std Expert {idx} gradients:")
            if self.use_multiple_log_std:
                log_std_param = self.log_std_parameter[idx]
            else:
                log_std_param = self.log_std_parameter

            if log_std_param.grad is not None:
                print(f"  grad_norm={log_std_param.grad.norm().item():.4f}")
            else:
                print("  grad=None")
    
    def load_state_dict(self, state_dict, strict=True):
        if 'log_std_parameter' in state_dict:
            loaded_log_std = state_dict.pop('log_std_parameter')
            if loaded_log_std.dim() == 2 and self.use_multiple_log_std:
                for i in range(min(self.num_experts, loaded_log_std.size(0))):
                    state_dict[f'log_std_parameter.{i}'] = loaded_log_std[i]
            elif loaded_log_std.dim() == 1 and self.use_multiple_log_std:
                for i in range(self.num_experts):
                    state_dict[f'log_std_parameter.{i}'] = loaded_log_std
            elif loaded_log_std.dim() == 1 and not self.use_multiple_log_std:
                state_dict['log_std_parameter'] = loaded_log_std

        super().load_state_dict(state_dict, strict=strict)

        # temp:
        if not self.eval_mode:
            self.reward_predictor[-1].bias.data[1:].fill_(-1.5) 
            for adapter in self.adapters.values():
                for layer in adapter:
                    if isinstance(layer, nn.Linear):
                        if self.init_adapters_as == "zero":
                            nn.init.zeros_(layer.weight)
                            nn.init.zeros_(layer.bias)
                        elif self.init_adapters_as == "one":
                            nn.init.eye_(layer.weight)
                            nn.init.zeros_(layer.bias)
            
        if not self.eval_mode:
            self.activate_new_expert(self.num_active_experts+1)

    def load_first_expert(self, state_dict):
        first_expert = self.experts[0]  
        missing_keys = []

        for name, param in first_expert.named_parameters():
            name = "experts.0." + name
            if name in state_dict:
                param.data.copy_(state_dict[name])
                print(f"Loaded parameter: {name}, mean={state_dict[name].mean().item():.4f}, std={state_dict[name].std().item():.4f}")
            else:
                missing_keys.append(name)

        if missing_keys:
            print(f"Warning: The following keys were not found in the state_dict and were skipped: {missing_keys}")
        else:
            print("Parameters loaded successfully into the first expert.")

    def load_log_std_parameter(self, state_dict):
        loaded_log_std = state_dict['log_std_parameter']

        if loaded_log_std.dim() == 1 and self.use_multiple_log_std:
            for i in range(self.num_experts):
                self.log_std_parameter[i].data.copy_(loaded_log_std)
        elif loaded_log_std.dim() == 1 and not self.use_multiple_log_std:
            self.log_std_parameter.data.copy_(loaded_log_std)
        else:
            print("Warning: log_std_parameter in state_dict has unexpected dimensions and was not loaded.")

    def load_reward_predictor(self, state_dict):
        missing_keys = []

        for name, param in self.reward_predictor.named_parameters():
            name = "reward_predictor." + name
            if name in state_dict:
                param.data.copy_(state_dict[name])
                print(f"Loaded parameter: {name}, mean={state_dict[name].mean().item():.4f}, std={state_dict[name].std().item():.4f}")
            else:
                missing_keys.append(name)

        if missing_keys:
            print(f"Warning: The following keys were not found in the state_dict and were skipped: {missing_keys}")
        else:
            print("Parameters loaded successfully into the reward predictor.")



class MoV(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, hidden_dim=1024, num_experts=4, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.device = device
        self.num_experts = num_experts
        self.current_expert_idx = 0
        
        experts = [
            nn.Sequential(
                nn.Linear(observation_space, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(hidden_dim/2)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim/2), 1) 
            ) for _ in range(num_experts)
        ]
        self.net = nn.ModuleList(experts)
        
    def compute(self, inputs, role):
        states = inputs["states"]
        
        if "expert_idx" not in inputs:
            raise ValueError("Value network requires expert_idx from policy.")
        
        expert_idx = inputs["expert_idx"].detach().view(-1) # [batch_size]
        self.current_expert_idx = expert_idx.max().item()
        self.update_gradient_status()
        expert_values = torch.stack([expert(states) for expert in self.net], dim=1) # [batch_size, num_experts, 1]

        weights = F.one_hot(expert_idx, num_classes=self.num_experts).float()
        selected_values = torch.sum(expert_values * weights.unsqueeze(-1), dim=1)

        return selected_values, {
            "selected_expert_idx": expert_idx,
            "all_expert_values": expert_values
        }

    def update_gradient_status(self):
        for i in range(self.num_experts):
            if i == (self.current_expert_idx):
                enable_grads(self.net[i])
            else:
                disable_grads(self.net[i])
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load parameters from a single value network into the first expert of MoV.
        """
        if "net.0.0.weight" not in state_dict:
            missing_keys = []
            first_expert = self.net[0]  # Only load parameters into the first expert

            for name, param in first_expert.named_parameters():
                value_key = f"net.{name}"
                if value_key in state_dict:
                    param.data.copy_(state_dict[value_key])
                else:
                    missing_keys.append(value_key)

            if strict and missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")

            if missing_keys:
                print(f"Warning: The following keys were not found in the state_dict and were skipped: {missing_keys}")

            print("State dict loaded successfully into the first expert of MoV class.")
            
        else:
            super().load_state_dict(state_dict, strict=strict)