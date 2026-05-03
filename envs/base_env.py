
# basic imports
from __future__ import annotations
import os
import torch

# isaac imports
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

# task imports
from isaaclab_tasks.direct.InteractionTracking.envs.env_cfgs import *

# utils
from isaaclab_tasks.direct.InteractionTracking.envs.utils.visualize import reset_markers
from isaaclab_tasks.direct.InteractionTracking.envs.utils.done import Killer
from isaaclab_tasks.direct.InteractionTracking.envs.utils.reward import Reward

# marker
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

# terrain
from isaaclab.terrains import TerrainImporter

class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.is_train = kwargs.get("is_train", True)
        self.timestep, self.timesteps = 0, kwargs.get("timesteps", 1000)
        self.task_name = type(self.cfg).__name__
        self._ALL_INDICES = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        # video tracking for wandb logging
        self.video_path = kwargs.get("video_path", None)
        self.video_interval = kwargs.get("video_interval", 0)
        self.last_logged_video = None 

        # logging
        self.wandb = {} # weight and bias
        self.log = {} # save .txt log to checkpoint folder

        # avoid using gym spaces
        self.observation_size = self.cfg.observation_space
        self.action_size = self.cfg.action_space

        # terminate or truncate
        self.killer = Killer(
            num_envs=self.num_envs,
            max_episode_length=self.max_episode_length,
            device=self.device,
        )

        # reward
        self.reward = Reward(self.num_envs, self.cfg.rewards, self.device)

    def _setup_scene(self):
        # add ground plane
        if self.cfg.terrain == "irregular":
            TerrainImporter(self.cfg.irregular_terrain_cfg)
        else:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # markers
        self.green_markers = VisualizationMarkers(self.cfg.marker_green_cfg)
        self.red_markers = VisualizationMarkers(self.cfg.marker_red_cfg)
        self.green_markers_small = VisualizationMarkers(self.cfg.marker_green_small_cfg)
        self.red_markers_small = VisualizationMarkers(self.cfg.marker_red_small_cfg)
        self.markers = [self.green_markers, self.red_markers, self.green_markers_small, self.red_markers_small]
        # self.arrow_green = VisualizationMarkers(self.cfg.arrow_green_cfg)
        reset_markers(self.markers)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, 
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

        # light_cfg = sim_utils.DomeLightCfg(
        #     intensity=1000.0,
        #     texture_file="./assets/environment/qwantani_moonrise_puresky_2k.hdr"
        # )
        # light_cfg.func("/World/Light", light_cfg)
    
    def _setup_dof_limits(self, robot: Articulation):
        # action offset and scale
        dof_lower_limits = robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        
    # Section: Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        pass
        
    def _apply_action(self):
        pass

    # Section: Post-physics step
    def post_step(self) -> None:
        self.extras = {} # reset info dictionary, is passed to agent
        
        # check for new video files and log to wandb at video_interval
        if self.video_path and self.video_interval > 0 and self.timestep % self.video_interval == 0:
            self._log_latest_video()

    def _get_dones(self):
        self.post_step()
    
    def _get_rewards(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES

        super()._reset_idx(env_ids)
        self.killer.reset(env_ids)
            
    def _get_observations(self):
        pass
    
    def _log_latest_video(self) -> None:
        """Find and log the latest video file to wandb extras"""
        if not self.video_path:
            return
        import glob
        try:
            video_files = glob.glob(os.path.join(str(self.video_path), "*.mp4"))
            if video_files:
                latest_video = max(video_files, key=os.path.getctime)
                # Only log if this is a new video
                if latest_video != self.last_logged_video:
                    self.last_logged_video = latest_video
                    # Initialize wandb dict if not present
                    if not self.wandb:
                        self.wandb = {}
                    self.wandb['Video'] = latest_video
        except Exception as e:
            pass  # silently ignore errors in video logging
    
    def _end_step(self) -> None:
        self.reward.reset()
        self.extras['wandb'] = self.wandb
        self.extras['log'] = self.log
        self.timestep += 1
