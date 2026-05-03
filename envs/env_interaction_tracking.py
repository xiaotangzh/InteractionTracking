
# basic imports
from __future__ import annotations
import numpy as np
import torch
import sys
from collections import deque
from pathlib import Path

# isaac imports
import gymnasium as gym

# task imports
from isaaclab_tasks.direct.InteractionTracking.envs.base_env import BaseEnv
from isaaclab_tasks.direct.InteractionTracking.envs.env_cfgs import *
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

# utils
from isaaclab_tasks.direct.InteractionTracking.utils.func import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.visualize import *
from isaaclab_tasks.direct.InteractionTracking.motions.motion_loader import *

# environment utils
from isaaclab_tasks.direct.InteractionTracking.envs.utils.utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward_utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.interaction import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reset import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.done import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.observation import *


class Env_Interaction_Tracking(BaseEnv):
    cfg: EnvCfg_Interaction_Tracking

    def __init__(self, cfg: EnvCfg_Interaction_Tracking, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_dof_limits(self.robot1)

        self.single_observation_size = self.cfg.observation_space // 2
        self.single_action_size = self.cfg.action_space // 2

        # load motion
        motion_files = [os.path.join(self.cfg.motion_file, n) for n in os.listdir(self.cfg.motion_file)]
        motion_files = [f for f in motion_files if os.path.basename(f) not in get_excluding_list('InterHuman_SMPL')]
        motion_files_1 = [f for f in motion_files if (os.path.isfile(f) and '_1' in f and 'npz' in f)]
        motion_files_2 = [f for f in motion_files if (os.path.isfile(f) and '_2' in f and 'npz' in f)]
        motion_files_1, motion_files_2 = pairing_motion_files(motion_files_1, motion_files_2)

        self.motion_loader_1 = MotionLoader(files_path=motion_files_1, device=self.device, reference_body=self.cfg.reference_body)
        self.motion_loader_2 = MotionLoader(files_path=motion_files_2, device=self.device, reference_body=self.cfg.reference_body)

        assert self.motion_loader_1.total_frames == self.motion_loader_2.total_frames, "Two character motions must have same number of frames."
        self.sampled_times = torch.zeros(len(self._ALL_INDICES), device=self.device, dtype=torch.float) # synchronized sampling times for all robots
        self.sampled_motion_ids = torch.zeros(len(self._ALL_INDICES), device=self.device, dtype=torch.long) # ids of sampled motion clips
        self.local_frame_indexes = torch.zeros([self.num_envs], device=self.device, dtype=torch.long) # local indexes in each clip

        # DOF and key body indexes
        self.ref_body_index = self.robot1.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot1.data.body_names.index(name) for name in self.cfg.key_body_names]
        self.motion_loader_1.reset_dof_body_indexes(self.robot1.data.joint_names, self.robot1.data.body_names) 
        self.motion_loader_2.reset_dof_body_indexes(self.robot2.data.joint_names, self.robot2.data.body_names)

        # log tracking performance
        self.motion_terminate_count = torch.ones(self.motion_loader_1.num_motions, device=self.device, dtype=torch.int)
        self.env_episode_tracking = torch.zeros(self.num_envs, device=self.device)
        self.motion_episode_sum   = torch.zeros(self.motion_loader_1.num_motions, device=self.device)
        self.motion_episode_count = torch.zeros(self.motion_loader_1.num_motions, device=self.device)

        # random object spawning variables
        self.object_spawn_timer = torch.zeros(self.num_envs, device=self.device)
        self.object_spawn_interval = 3 # seconds

    @property
    def episode_tracking_rewards(self) -> torch.Tensor:
        return self.motion_episode_sum / self.motion_episode_count.clamp(min=1)

    @property
    def tracking_performance(self) -> dict:
        stats = {self.motion_loader_1.motion_names[i]: self.episode_tracking_rewards[i].item() for i in range(len(self.episode_tracking_rewards))}
        sorted_stats = dict(sorted(stats.items(), key=lambda item: item[1]))
        return sorted_stats
    
    @property
    def global_frame_indexes(self) -> torch.Tensor:
        return self.local_frame_indexes + self.motion_loader_1.start_frames[self.sampled_motion_ids]
    
    def _setup_scene(self):
        super()._setup_scene()
        
        self.robot1 = Articulation(self.cfg.robot1)
        self.robot2 = Articulation(self.cfg.robot2)
        self.scene.articulations["robot1"] = self.robot1
        self.scene.articulations["robot2"] = self.robot2
        
        if self.cfg.object_perturb:
            self.random_object = RigidObject(cfg=self.cfg.sphere_cfg)
            self.scene.rigid_objects["random_object"] = self.random_object

    # Section: Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)

        # get actions
        if self.cfg.action_clip is not None:
            actions = torch.clip(actions, min=-self.cfg.action_clip, max=self.cfg.action_clip) 
        self.actions1, self.actions2 = torch.chunk(actions.clone(), chunks=2, dim=-1)

        # handle random object spawning 
        if self.cfg.object_perturb:
            dt = self.step_dt
            self.object_spawn_timer += dt
            
            # check which environments need object respawning 
            spawn_mask = self.object_spawn_timer >= self.object_spawn_interval
            if spawn_mask.any():
                spawn_random_objects(self.random_object, spawn_mask, self.robot1 if random.random() < 0.5 else self.robot2, self.device)
                self.object_spawn_timer[spawn_mask] = 0.0

        # collect ground truth
        self.gt1 = collect_ground_truth(self.motion_loader_1, self.global_frame_indexes, self.scene.env_origins) 
        self.gt2 = collect_ground_truth(self.motion_loader_2, self.global_frame_indexes, self.scene.env_origins)

    def _apply_action(self):
        super()._apply_action()

        target1 = self.action_offset + self.action_scale * self.actions1
        self.robot1.set_joint_position_target(target1)
        target2 = self.action_offset + self.action_scale * self.actions2
        self.robot2.set_joint_position_target(target2)

    # Section: Post-physics step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # should return resets and time_out
        super()._get_dones()
            
        truncated = self.episode_length_buf >= self.max_episode_length - 1 # bools of envs that are time out

        exceeding_envs = check_exceeding_frame_index(self.local_frame_indexes, self.motion_loader_1.motion_frames[self.sampled_motion_ids], bias=self.cfg.future_step)
        
        if self.cfg.early_termination:
            terminated_1, terminated_2 = [], []

            terminated_1.extend([
                terminate_by_tracking_error(self.robot1.data.body_pos_w, self.gt1.body_pos_w)
            ])
            terminated_2.extend([
                terminate_by_tracking_error(self.robot2.data.body_pos_w, self.gt2.body_pos_w)
            ])

            terminated = torch.max(torch.stack(terminated_1 + terminated_2, dim=0), dim=0).values # [num_envs,]
            self.extras["terminated_1"] = torch.max(torch.stack(terminated_1, dim=0), dim=0).values
            self.extras["terminated_2"] = torch.max(torch.stack(terminated_2, dim=0), dim=0).values

            # for value bootstrapping
            self.extras["truncated_1"] = torch.max(
                torch.cat([torch.stack([truncated, exceeding_envs], dim=0), torch.stack(terminated_2, dim=0)], dim=0)
            , dim=0).values
            self.extras["truncated_2"] = torch.max(
                torch.cat([torch.stack([truncated, exceeding_envs], dim=0), torch.stack(terminated_1, dim=0)], dim=0)
            , dim=0).values
            
        else: # no early termination until time out
            terminated = torch.zeros_like(truncated)
            self.extras["terminated_1"] = torch.zeros_like(truncated)
            self.extras["terminated_2"] = torch.zeros_like(truncated)
            self.extras["truncated_1"] = torch.max(torch.stack([truncated, exceeding_envs], dim=0), dim=0).values
            self.extras["truncated_2"] = torch.max(torch.stack([truncated, exceeding_envs], dim=0), dim=0).values
        
        self.motion_terminate_count = count_motion_terminate(terminated, self.motion_terminate_count, self.sampled_motion_ids)

        truncated = torch.max(torch.stack([truncated, exceeding_envs], dim=0), dim=0).values

        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        super()._get_rewards()

        rewards_1 = compute_rewards(self.cfg.rewards, self.num_envs, robot=self.robot1, _robot=self.robot2, gt=self.gt1, key_body_indexes=self.key_body_indexes)
        rewards_2 = compute_rewards(self.cfg.rewards, self.num_envs, robot=self.robot2, _robot=self.robot1, gt=self.gt2, key_body_indexes=self.key_body_indexes)
        rewards = (rewards_1['total'] + rewards_2['total']) / 2.0

        # compute tracking rewards for each motion
        self.env_episode_tracking += (rewards_1['tracking'] + rewards_2['tracking']) / 2.0

        self.wandb['motion_episode_min_tracking_reward'] = self.episode_tracking_rewards.min().cpu()
        self.wandb['motion_episode_mean_tracking_reward'] = self.episode_tracking_rewards.mean().cpu()
        self.wandb['motion_episode_max_tracking_reward'] = self.episode_tracking_rewards.max().cpu()
        self.wandb['motion_episode_weighted_sum_tracking_reward'] = (self.episode_tracking_rewards.cpu() * self.motion_loader_1.motion_duration_weights.cpu()).sum()
        self.wandb['motion_episode_median_tracking_reward'] = self.episode_tracking_rewards.median().cpu()

        if self.timestep % 1000 == 0:
            self.log['tracking_performance'] = self.tracking_performance

        self.extras["rewards_1"] = rewards_1['total']
        self.extras["rewards_2"] = rewards_2['total']

        return rewards
    
    def _reset_idx(self, env_ids: torch.Tensor | None): # env_ids: the ids of envs needed to be reset
        # reset motion episode rewards (BEFORE resetting self.episode_length_buf)
        mids   = self.sampled_motion_ids[env_ids]
        ep_rew = self.env_episode_tracking[env_ids] / (self.episode_length_buf[env_ids]+1) # use mean reward in episode
        self.motion_episode_sum  .index_add_(0, mids, ep_rew)
        self.motion_episode_count.index_add_(0, mids, torch.ones_like(ep_rew))
        self.env_episode_tracking[env_ids] = 0.0

        super()._reset_idx(env_ids)
        assert env_ids is not None
        self.robot1.reset(env_ids)
        self.robot2.reset(env_ids)
        
        # reset object spawn timer for reset environments
        if self.cfg.object_perturb:
            self.object_spawn_timer[env_ids] = 0.0

        # reset robot states
        if self.cfg.reset_strategy == "default":
            root_state_1, joint_pos_1, joint_vel_1 = reset_strategy_default(self.scene.env_origins, env_ids, self.robot1, self.cfg.lift_root_height)
            root_state_2, joint_pos_2, joint_vel_2 = reset_strategy_default(self.scene.env_origins, env_ids, self.robot2, self.cfg.lift_root_height)
            root_state_2 = root_state_face2face(root_state_2) 

        elif self.cfg.reset_strategy.startswith("random"):
            sampled_motion_ids = self.motion_loader_1.sample_motion_ids(env_ids.shape[0], self.cfg.sample_strategy, failure_counts=self.motion_terminate_count, tracking_rewards=self.episode_tracking_rewards, timestep=self.timestep)
            
            if "start" in self.cfg.reset_strategy:
                sampled_times = torch.zeros(env_ids.shape[0]).to(env_ids.device)
            else:
                sampled_times = self.motion_loader_1.sample_times(env_ids.shape[0], sampled_motion_ids, bias_end=self.cfg.future_step) 
            
            self.sampled_motion_ids[env_ids], self.sampled_times[env_ids] = sampled_motion_ids, sampled_times

            root_state_1, joint_pos_1, joint_vel_1 = reset_strategy_random(env_ids, self.robot1, sampled_motion_ids, sampled_times, self.motion_loader_1, self.ref_body_index, self.cfg.lift_root_height, self.scene.env_origins)
            root_state_2, joint_pos_2, joint_vel_2 = reset_strategy_random(env_ids, self.robot2, sampled_motion_ids, sampled_times, self.motion_loader_2, self.ref_body_index, self.cfg.lift_root_height, self.scene.env_origins)

        reset_robot(self.robot1, env_ids, root_state_1, joint_pos_1, joint_vel_1)
        reset_robot(self.robot2, env_ids, root_state_2, joint_pos_2, joint_vel_2)

        # spawn new objects for reset environments (after robots are reset)
        if self.cfg.object_perturb:
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[env_ids] = True
            spawn_random_objects(self.random_object, reset_mask, self.robot1 if random.random() < 0.5 else self.robot2, self.device)

        # reset frame indexes
        reset_frame_indexes(
            self.local_frame_indexes, 
            self.sampled_times, 
            self.sampled_motion_ids, 
            env_ids, 
            self.cfg.reset_strategy, 
            self.motion_loader_1
        )
   
    def _get_observations(self) -> dict:
        super()._get_observations()

        self.target_states_1 = collect_ground_truth(
            self.motion_loader_1,
            self.global_frame_indexes+self.cfg.future_step, 
            self.scene.env_origins,
            add_noise=self.cfg.noise_perturb,
            key_body_indexes=self.key_body_indexes
        )
        self.target_states_2 = collect_ground_truth(
            self.motion_loader_2,
            self.global_frame_indexes+self.cfg.future_step, 
            self.scene.env_origins,
            add_noise=self.cfg.noise_perturb,
            key_body_indexes=self.key_body_indexes
        )

        if self.cfg.visualize:
            positions = torch.cat([self.target_states_1.body_pos_w, self.target_states_2.body_pos_w], dim=0)
            visualize_markers(self.green_markers_small, positions)

            contact_positions = detect_close_joints(self.robot1.data.body_pos_w, self.robot2.data.body_pos_w)
            visualize_markers(self.red_markers, contact_positions)

        # build observations
        obs = torch.cat([
            build_tracking_observation(self.robot1, self.target_states_1, self.ref_body_index, self._ALL_INDICES, key_body_indexes=self.key_body_indexes),
            build_tracking_observation(self.robot2, self.target_states_2, self.ref_body_index, self._ALL_INDICES, key_body_indexes=self.key_body_indexes)
        ], dim=-1)

        # reset NaN environments
        tries = 0
        while has_nan(obs):
            nan_env_ids = reset_nan_env(obs, self._reset_idx, self.num_envs)
            if tries >= 10:
                print(f"Maximum reset tries (10) reached, exiting.")
                sys.exit(0)
            tries += 1
            
            # reset observations
            self.target_states_1 = collect_ground_truth(
                self.motion_loader_1, 
                self.global_frame_indexes+self.cfg.future_step, 
                self.scene.env_origins, 
                key_body_indexes=self.key_body_indexes)
            self.target_states_2 = collect_ground_truth(
                self.motion_loader_2, 
                self.global_frame_indexes+self.cfg.future_step, 
                self.scene.env_origins, 
                key_body_indexes=self.key_body_indexes)
            
            obs[nan_env_ids] = torch.cat([
                build_tracking_observation(self.robot1, self.target_states_1, self.ref_body_index, env_ids=nan_env_ids, key_body_indexes=self.key_body_indexes),
                build_tracking_observation(self.robot2, self.target_states_2, self.ref_body_index, env_ids=nan_env_ids, key_body_indexes=self.key_body_indexes)
            ], dim=-1)

        self._end_step()

        return {"policy": obs} 
    
    def _end_step(self) -> None:
        super()._end_step()
        self.local_frame_indexes += 1 
