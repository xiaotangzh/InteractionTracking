
# basic imports
from __future__ import annotations
import numpy as np
import torch
import sys

# isaac imports
import gymnasium as gym

# task imports
from isaaclab_tasks.direct.PhysicsProject.envs.base_env import BaseEnv
from isaaclab_tasks.direct.PhysicsProject.envs.env_cfgs import *
from isaaclab.assets import Articulation

# utils
from isaaclab_tasks.direct.PhysicsProject.utils.func import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.visualize import *
from isaaclab_tasks.direct.PhysicsProject.motions.motion_loader import MotionLoader

# environment utils
from isaaclab_tasks.direct.PhysicsProject.envs.utils.utils import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.reward import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.reward_utils import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.interaction import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.reset import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.done import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.observation import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.amp import *
from isaaclab_tasks.direct.PhysicsProject.envs.utils.state import State
from isaaclab_tasks.direct.PhysicsProject.envs.utils.tracking import Tracker

class Env_Tracking(BaseEnv):
    cfg: EnvCfg_Tracking

    def __init__(self, cfg: EnvCfg_Tracking, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_dof_limits(self.robot)

        self.motion_loader = MotionLoader(
            files_path=self.cfg.motion_file, 
            device=self.device, 
            reference_body=self.cfg.reference_body,
            root_height=self.cfg.root_height,
            max_motions=None if self.is_train else 50, # cuda out of memory
        ) 
        self.motion_loader.reset_dof_body_indexes(self.robot.data.joint_names, self.robot.data.body_names)
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in self.cfg.key_body_names]

        self.tracker = Tracker(self.num_envs, self.motion_loader, self.scene.env_origins, self.key_body_indexes, self.device)

        # AMP 
        self.amp_observation_size = self.cfg.amp_observation_size
        self.num_amp_observations = self.cfg.num_amp_observations
        self.amp = AdversarialMotionPrior(
            self.num_envs, 
            self.amp_observation_size, 
            self.num_amp_observations, 
            self.motion_loader.body_names,
            self.motion_loader.dof_names,
            self.device,
        )
        self.gt_amp = AdversarialMotionPrior(
            self.num_envs, 
            self.amp_observation_size, 
            self.num_amp_observations,
            self.motion_loader.body_names,
            self.motion_loader.dof_names,
            self.device,
        )
        self.amp_total_size = self.amp.total_size

    def _setup_scene(self):
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations['robot'] = self.robot

        if self.cfg.object_perturb:
            self.random_object = RigidObject(cfg=self.cfg.sphere_cfg)
            self.scene.rigid_objects["random_object"] = self.random_object
    


    # Section: Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)

        if self.cfg.action_clip is not None:
            actions = torch.clip(actions, min=-self.cfg.action_clip, max=self.cfg.action_clip) 

        self.actions = actions.clone()
        self.tracker.collect_ground_truth(self.motion_loader) # frame index already stepped forward

        # Step AMP buffers once per policy step (not inside decimation loop) so that
        # amp and gt_amp share the same policy-step-level temporal resolution.
        # Stepping gt_amp inside _apply_action would repeat the same tracker.gt frame
        # `decimation` times, producing stuttering sequences that diverge from training data.
        self.amp.step(self.robot)
        self.gt_amp.step(self.tracker.gt)

    def _apply_action(self):
        super()._apply_action()

        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

        # use ground truth
        # write_ref_state(self.robot, self.motion_loader, self.global_frame_indexes, self.scene.env_origins) 

        # self.extras['diff'] = Mimic.Obs.compute_diff(self.robot, self.tracker.gt)

    # Section: Post-physics step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # bool
        super()._get_dones()

        self.killer.truncate(
            self.killer.truncate_episode(self.episode_length_buf),
            self.tracker.truncate_out_of_bound(),
        )

        if self.cfg.early_termination:
            self.killer.terminate(
                self.tracker.terminate_by_tracking_error(self.robot.data.body_pos_w, self.tracker.gt.body_pos_w), 
                # Killer.terminate_by_height(self.robot.data.body_pos_w[:, 0, 2], self.cfg.terminate_root_height) # test:
            )

        self.tracker.count_failure(self.killer.terminated, self.tracker.motion_ids)
        return self.killer.terminated, self.killer.truncated
    
    def _get_rewards(self) -> torch.Tensor:
        super()._get_rewards()

        self.reward.compute(
            tracking=DeepMimic.Reward.compute(self.robot, self.tracker.gt, self.key_body_indexes),
            # tracking=Mimic.Obs.compute_mse(self.robot, self.tracker.gt), # test:
            energy_penalty=Reward.Func.energy_penalty(self.robot),
        )

        self.tracker.env_episode_tracking += self.reward.tracking

        self.amp.step_rewards(self.reward.total)
        self.gt_amp.step_rewards(torch.ones_like(self.reward.total, device=self.device))

        if self.timestep % 1000 == 0:
            self.log['tracking_performance'] = self.tracker.tracking_performance

        # self.reward.print()
        return self.reward.total
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        self.tracker.reset_episode_tracking(env_ids, self.episode_length_buf) # before resetting episode_length_buf
        
        super()._reset_idx(env_ids)
        assert env_ids is not None
        self.robot.reset(env_ids)
        
        assert self.cfg.reset_strategy.startswith('random')
        motion_ids = self.motion_loader.sample_motion_ids(
            env_ids.shape[0], 
            self.cfg.sample_strategy, 
            failure_counts=self.tracker.motion_terminate_count, 
            tracking_rewards=self.tracker.motion_avg_performance, 
            timestep=self.timestep
        )
        
        if "start" in self.cfg.reset_strategy:
            timestamps = torch.zeros(env_ids.shape[0]).to(env_ids.device)
        else:
            timestamps = self.motion_loader.sample_times(
                env_ids.shape[0], 
                motion_ids, 
                bias_end=self.num_amp_observations, 
                bias_start=self.num_amp_observations
            ) 
        
        self.tracker.update_sampling(env_ids, motion_ids, timestamps)

        root_state, joint_pos, joint_vel = reset_strategy_random(
            env_ids, 
            self.robot, 
            motion_ids, 
            timestamps, 
            self.motion_loader, 
            self.ref_body_index, 
            self.cfg.lift_root_height, 
            self.scene.env_origins
        )

        self.tracker.reset_frame_indexes(
            env_ids, 
            self.cfg.reset_strategy, 
            self.motion_loader.get_frame_index_from_time(timestamps, env_ids.shape[0], motion_ids)[0]
        )
        
        reset_robot(self.robot, env_ids, root_state, joint_pos, joint_vel)

        # reset amp observations
        self.amp.reset(
            self.motion_loader,
            env_ids=env_ids,
            motion_ids=motion_ids,
            timestamps=timestamps,
        )
        self.gt_amp.reset(
            self.motion_loader,
            env_ids=env_ids,
            motion_ids=motion_ids,
            timestamps=timestamps,
        )

    def _get_observations(self) -> dict:
        super()._get_observations()
  
        next_states = self.tracker.collect_states(
            self.motion_loader, 
            bias=1, 
        )

        if self.cfg.visualize: 
            visualize_markers(self.green_markers_small, next_states.body_pos_w)

        obs = DeepMimic.Obs.compute(
            self.robot,
            next_states,
            self._ALL_INDICES,
            self.key_body_indexes,
        )

        # reset NaN environments
        tries = 0
        while torch.any(find_nan(obs)):
            nan_env_ids = reset_nan_env(obs, self._reset_idx, self.num_envs)
            if tries >= 10:
                print(f'Maximum reset tries (10) reached, exiting.')
                sys.exit(0)
            tries += 1
            
            # reset observations
            obs[nan_env_ids] = DeepMimic.Obs.compute(
                self.robot,
                next_states,
                nan_env_ids,
                self.key_body_indexes,
            )
        
        valid_amp_envs = self.episode_length_buf > self.num_amp_observations
        if torch.any(valid_amp_envs):
            self.extras['amp_rewards'] = self.amp.mean_rewards[valid_amp_envs]
            self.extras['gt_amp_rewards'] = self.gt_amp.mean_rewards[valid_amp_envs]
            self.extras['amp_obs'] = self.amp.flat_data[valid_amp_envs]
            self.extras['gt_amp_obs'] = self.gt_amp.flat_data[valid_amp_envs]
        self.extras['valid_amp_envs'] = valid_amp_envs

        self._end_step()
        return {'policy': obs}
    
    def _end_step(self) -> None:
        super()._end_step()
        self.tracker.step()

    # Section: Functions needed to be called by agent
    def collect_reference_motions(self, 
        num_samples: int, 
        sampled_motion_ids: torch.Tensor | None=None, 
        sampled_times: torch.Tensor | None = None, 
        motion_loader: MotionLoader | None=None
    ) -> torch.Tensor:
        
        if sampled_times is None or sampled_motion_ids is None:
            sampled_motion_ids = self.motion_loader.sample_motion_ids(num_samples, strategy="duration")
            sampled_times = self.motion_loader.sample_times(num_samples, sampled_motion_ids)

        assert sampled_motion_ids is not None and sampled_times is not None

        if motion_loader is None:
            motion_loader = self.motion_loader

        return self.amp.collect_observations(
            motion_loader, 
            num_samples=num_samples,
            motion_ids=sampled_motion_ids, 
            times=AdversarialMotionPrior.compute_times(sampled_times, motion_loader, self.amp.num_steps)
        ).view(num_samples, self.amp.total_size)
    