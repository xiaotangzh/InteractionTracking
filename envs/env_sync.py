
# basic imports
from __future__ import annotations
import numpy as np
import torch
import sys
from collections import deque

# task imports
from isaaclab_tasks.direct.InteractionTracking.envs.base_env import BaseEnv
from isaaclab_tasks.direct.InteractionTracking.envs.env_cfgs import *

# utils
from isaaclab_tasks.direct.InteractionTracking.utils.func import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.visualize import *
from isaaclab_tasks.direct.InteractionTracking.motions.motion_loader import MotionLoader

# environment utils
from isaaclab_tasks.direct.InteractionTracking.envs.utils.utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward_utils import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.interaction import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reset import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.done import *
from isaaclab_tasks.direct.InteractionTracking.envs.utils.tracking import Tracker
from isaaclab_tasks.direct.InteractionTracking.envs.utils.bmp import BioMechanicalPrior

class Env_Sync(BaseEnv):
    cfg: EnvCfg_Sync

    def __init__(self, cfg: EnvCfg_Sync, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_dof_limits(self.robot)

        # load motion
        self.motion_loader = MotionLoader(
            files_path=self.cfg.motion_file, 
            device=self.device, 
            reference_body=self.cfg.reference_body, 
            min_duration=0.0, 
            max_motions=500
        )
        self.motion_loader.reset_dof_body_indexes(self.robot.data.joint_names, self.robot.data.body_names) 

        self.tracker = Tracker(self.num_envs, self.motion_loader, self.scene.env_origins, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()

        # add articulation to scene
        self.robot = Articulation(self.cfg.robot) # init_root_height = self.robot.cfg.init_state.pos[2]
        self.scene.articulations["robot"] = self.robot
        
    # Section: Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)

    def _apply_action(self):
        super()._apply_action()

        write_ref_state(self.robot, self.motion_loader, self.tracker.global_frame_indexes, self.scene.env_origins) 

        self.cur_motion_ids = torch.searchsorted(self.motion_loader.start_frames, self.tracker.global_frame_indexes, right=True) - 1
        printl(f"Current motion: {self.cur_motion_ids[0]+1}/{len(self.motion_loader.motion_names)}: {self.motion_loader.motion_names[self.cur_motion_ids[0].item()]}")

        # bmp = BioMechanicalPrior.compute(self.robot)
        # print(f"BMP: {bmp}")

    # Section: Post-physics step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # should return resets and time_out
        super()._get_dones()

        self.killer.truncate(
            self.tracker.truncate_out_of_bound(mode='global'),
        )

        return self.killer.terminated, self.killer.truncated
    
    def _get_rewards(self) -> torch.Tensor:
        super()._get_rewards()
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

    def _reset_idx(self, env_ids: torch.Tensor | None): # env_ids: the ids of envs needed to be reset
        super()._reset_idx(env_ids)
        assert env_ids is not None
        self.robot.reset(env_ids)

        root_state, joint_pos, joint_vel = reset_strategy_default(self.scene.env_origins, env_ids, self.robot)

        self.tracker.reset_frame_indexes(env_ids)
        
        reset_robot(self.robot, env_ids, root_state, joint_pos, joint_vel)
            
    def _get_observations(self) -> dict:
        super()._get_observations()

        next_states = self.tracker.collect_states(
            self.motion_loader, 
            bias=1, 
        )

        visualize_markers(self.green_markers_small, next_states.body_pos_w)

        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device, dtype=torch.float)

        self._end_step()
        return {"policy": obs}
    
    def _end_step(self) -> None:
        super()._end_step()
        self.tracker.step()