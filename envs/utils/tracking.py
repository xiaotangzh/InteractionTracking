import torch
from isaaclab_tasks.direct.PhysicsProject.motions.motion_loader import MotionLoader
from isaaclab_tasks.direct.PhysicsProject.envs.utils.state import State
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets import Articulation

class Tracker:
    def __init__(self,
        num_envs: int,
        motion_loader: MotionLoader,
        env_origins: torch.Tensor,
        key_body_indexes: list | None=None,
        device: torch.device | str="cpu",
    ):
        self.num_envs = num_envs
        self.device = device

        # motion sampling
        self.timestamps = torch.zeros(num_envs, device=device, dtype=torch.float) # sampled timestamps in each motion clip
        self.motion_ids = torch.zeros(num_envs, device=device, dtype=torch.long) # ids of sampled motion clips
        self.local_frame_indexes = torch.zeros([num_envs], device=device, dtype=torch.long) # local indexes in each clip

        # motion info
        self.num_motions = motion_loader.num_motions
        self.start_frames = motion_loader.start_frames
        self.end_frames = motion_loader.end_frames
        self.motion_frames = motion_loader.motion_frames
        self.motion_names = motion_loader.motion_names

        # log tracking performance
        self.motion_terminate_count = torch.ones(self.num_motions, device=device, dtype=torch.int)
        self.env_episode_tracking = torch.zeros(num_envs, device=device)
        self.motion_episode_sum   = torch.zeros(self.num_motions, device=device)
        self.motion_episode_count = torch.zeros(self.num_motions, device=device)

        # for collecting ground truth
        self.env_origins = env_origins
        self.key_body_indexes = key_body_indexes

    @property
    def global_frame_indexes(self) -> torch.Tensor:
        return self.local_frame_indexes + self.start_frames[self.motion_ids]

    @property
    def motion_avg_performance(self) -> torch.Tensor:
        return self.motion_episode_sum / self.motion_episode_count.clamp(min=1)

    @property
    def tracking_performance(self) -> dict:
        stats = {self.motion_names[i]: self.motion_avg_performance[i].item() for i in range(len(self.motion_avg_performance))}
        sorted_stats = dict(sorted(stats.items(), key=lambda item: item[1]))
        return sorted_stats
    
    def step(self):
        self.local_frame_indexes += 1
    
    def count_failure(self, terminates: torch.Tensor, sampled_motion_ids: torch.Tensor):
        terminated_motion_ids = sampled_motion_ids[terminates]  # shape: [terminate_envs,]
        motion_counts = torch.bincount(terminated_motion_ids, minlength=self.motion_terminate_count.shape[0])
        self.motion_terminate_count += motion_counts

    def compute_episode_avg_reward(self, episode_length_buf) -> torch.Tensor:
        return self.env_episode_tracking / (episode_length_buf+1)
    
    def reset_episode_tracking(self, env_ids: torch.Tensor | None, episode_length_buf: torch.Tensor):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) 

        mids = self.motion_ids[env_ids]
        ep_rew = self.env_episode_tracking[env_ids] / (episode_length_buf[env_ids]+1) # use mean reward in episode
        self.motion_episode_sum.index_add_(0, mids, ep_rew)
        self.motion_episode_count.index_add_(0, mids, torch.ones_like(ep_rew))
        self.env_episode_tracking[env_ids] = 0.0

    def update_sampling(self, env_ids: torch.Tensor, sampled_motion_ids: torch.Tensor, sampled_times: torch.Tensor):
        self.motion_ids[env_ids], self.timestamps[env_ids] = sampled_motion_ids, sampled_times

    def reset_frame_indexes(self, env_ids: torch.Tensor, reset_strategy: str="default", local_frame_indexes: torch.Tensor | None=None):
        if reset_strategy == "random":
            assert local_frame_indexes is not None
            self.local_frame_indexes[env_ids] = local_frame_indexes
        else:
            self.local_frame_indexes[env_ids] = 0
    
    def truncate_out_of_bound(self, mode: str='local', bias=1) -> torch.Tensor:
        if mode == 'local':
            invalid_mask = self.local_frame_indexes > (self.motion_frames[self.motion_ids] - 1 - bias)
        else:
            invalid_mask = self.global_frame_indexes > (self.end_frames[-1] - 1 - bias)
        return invalid_mask.view(-1)

    @staticmethod
    def terminate_by_tracking_error(cur_pos: torch.Tensor, target_pos: torch.Tensor, dist: float=0.5) -> torch.Tensor:
        dis_to_target_pos = torch.mean(torch.norm(cur_pos - target_pos, dim=-1, keepdim=True), dim=1)
        return (dis_to_target_pos > dist).view(-1)

    def collect_states(self,
        motion_loader: MotionLoader, 
        bias: int=0, # 0 if collect current frame, 1 if collect next frame
    ) -> State:
        
        root_states = motion_loader.root_states[self.global_frame_indexes+bias]
        root_states[:, :3] += self.env_origins
        states = State(
            root_state_w=root_states,
            joint_pos=motion_loader.dof_positions[self.global_frame_indexes+bias],
            joint_vel=motion_loader.dof_velocities[self.global_frame_indexes+bias],
            body_pos_w=motion_loader.body_positions[self.global_frame_indexes+bias] + self.env_origins.unsqueeze(1),
            body_quat_w=motion_loader.body_rotations[self.global_frame_indexes+bias],
            body_lin_vel_w=motion_loader.body_linear_velocities[self.global_frame_indexes+bias],
            body_ang_vel_w=motion_loader.body_angular_velocities[self.global_frame_indexes+bias],
            num_joints=motion_loader.num_dofs,
            num_bodies=motion_loader.num_bodies,
        )

        if self.key_body_indexes is not None:
            states.key_body_pos_w = motion_loader.body_positions[self.global_frame_indexes+bias][:, self.key_body_indexes] + self.env_origins.unsqueeze(1)
        
        return states
    
    def collect_ground_truth(self,motion_loader: "MotionLoader"):
        self.gt = self.collect_states(motion_loader, bias=0)