import torch
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.assets import Articulation

class State:
    def __init__(self,
        root_state_w: torch.Tensor = torch.tensor([]),
        joint_pos: torch.Tensor = torch.tensor([]),
        joint_vel: torch.Tensor = torch.tensor([]),
        body_pos_w: torch.Tensor = torch.tensor([]),
        body_quat_w: torch.Tensor = torch.tensor([]),
        body_lin_vel_w: torch.Tensor = torch.tensor([]),
        body_ang_vel_w: torch.Tensor = torch.tensor([]),
        key_body_pos_w: torch.Tensor | None = None,

        num_joints: int = 0,
        num_bodies: int = 0,
    ):
        # root
        self.root_state_w = root_state_w
        if self.root_state_w.numel() > 0:
            self.root_pos_w = self.root_state_w[..., :3]
            self.root_quat_w = self.root_state_w[..., 3:7]
            self.root_lin_vel_w = self.root_state_w[..., 7:10]
            self.root_ang_vel_w = self.root_state_w[..., 10:13]

        # joints
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel

        # rigid bodies
        self.body_pos_w = body_pos_w
        self.body_quat_w = body_quat_w
        self.body_lin_vel_w = body_lin_vel_w
        self.body_ang_vel_w = body_ang_vel_w

        # key rigid bodies
        self.key_body_pos_w = key_body_pos_w

        # metadata
        self.num_joints = num_joints
        self.num_bodies = num_bodies

    @property
    def data(self):
        return self
    
    @property
    def _ALL_INDICES(self):
        return torch.arange(self.root_state_w.shape[0])
    
    @property
    def has_bodies_data(self):
        return self.body_pos_w.numel() > 0 and self.body_quat_w.numel() > 0 and self.body_lin_vel_w.numel() > 0 and self.body_ang_vel_w.numel() > 0
    
    # def check_dimensions(self):
    #     # check that all tensors have the same dimensionality except for the last dimension
    #     dims = []
        
    #     if self.root_state_w.numel() > 0:
    #         dims.append(self.root_state_w.shape[:-1])
    #     if self.joint_pos.numel() > 0:
    #         dims.append(self.joint_pos.shape[:-1])
    #     if self.joint_vel.numel() > 0:
    #         dims.append(self.joint_vel.shape[:-1])
    #     if self.body_pos_w.numel() > 0:
    #         dims.append(self.body_pos_w.shape[:-1])
    #     if self.body_quat_w.numel() > 0:
    #         dims.append(self.body_quat_w.shape[:-1])
    #     if self.body_lin_vel_w.numel() > 0:
    #         dims.append(self.body_lin_vel_w.shape[:-1])
    #     if self.body_ang_vel_w.numel() > 0:
    #         dims.append(self.body_ang_vel_w.shape[:-1])
    #     if self.key_body_pos_w is not None and self.key_body_pos_w.numel() > 0:
    #         dims.append(self.key_body_pos_w.shape[:-2])
        
    #     if len(dims) > 1:
    #         first_dim = dims[0]
    #         for dim in dims[1:]:
    #             if dim != first_dim:
    #                 raise ValueError(f"Dimension mismatch: expected {first_dim}, got {dim}")
        
    #     return True
    
    # def reduce_dimension(self, dim: int=0) -> list["State"]:
    #     self.check_dimensions()

    #     # get the size of the dimension to reduce
    #     size = None
    #     if self.root_state_w.numel() > 0:
    #         size = self.root_state_w.shape[dim]
    #     elif self.joint_pos.numel() > 0:
    #         size = self.joint_pos.shape[dim]
    #     elif self.body_pos_w.numel() > 0:
    #         size = self.body_pos_w.shape[dim]
    #     else:
    #         raise ValueError("All tensors are empty")
        
    #     states = []
    #     for i in range(size):
    #         state = State(
    #             root_state_w=self.root_state_w.select(dim, i) if self.root_state_w.numel() > 0 else torch.tensor([]),
    #             joint_pos=self.joint_pos.select(dim, i) if self.joint_pos.numel() > 0 else torch.tensor([]),
    #             joint_vel=self.joint_vel.select(dim, i) if self.joint_vel.numel() > 0 else torch.tensor([]),
    #             body_pos_w=self.body_pos_w.select(dim, i) if self.body_pos_w.numel() > 0 else torch.tensor([]),
    #             body_quat_w=self.body_quat_w.select(dim, i) if self.body_quat_w.numel() > 0 else torch.tensor([]),
    #             body_lin_vel_w=self.body_lin_vel_w.select(dim, i) if self.body_lin_vel_w.numel() > 0 else torch.tensor([]),
    #             body_ang_vel_w=self.body_ang_vel_w.select(dim, i) if self.body_ang_vel_w.numel() > 0 else torch.tensor([]),
    #             key_body_pos_w=self.key_body_pos_w.select(dim, i) if self.key_body_pos_w is not None and self.key_body_pos_w.numel() > 0 else None,
    #         )
    #         states.append(state)
        
    #     return states

    def from_articulation(self, robot: "Articulation", key_body_indices: torch.Tensor | None=None):
        self.root_state_w = robot.data.root_state_w
        self.root_pos_w = self.root_state_w[:, :3]
        self.root_quat_w = self.root_state_w[:, 3:7]
        self.root_lin_vel_w = self.root_state_w[:, 7:10]
        self.root_ang_vel_w = self.root_state_w[:, 10:13]

        self.joint_pos = robot.data.joint_pos
        self.joint_vel = robot.data.joint_vel

        self.body_pos_w = robot.data.body_pos_w
        self.body_quat_w = robot.data.body_quat_w
        self.body_lin_vel_w = robot.data.body_lin_vel_w
        self.body_ang_vel_w = robot.data.body_ang_vel_w

        if key_body_indices is not None:
            self.key_body_pos_w = robot.data.body_pos_w[:, key_body_indices]

        return self

    # def reset(self, state: "State", env_ids: torch.Tensor):
    #     self.root_state_w[env_ids] = state.root_state_w[env_ids]
    #     self.root_pos_w[env_ids] = state.root_pos_w[env_ids]
    #     self.root_quat_w[env_ids] = state.root_quat_w[env_ids]
    #     self.root_lin_vel_w[env_ids] = state.root_lin_vel_w[env_ids]
    #     self.root_ang_vel_w[env_ids] = state.root_ang_vel_w[env_ids]

    #     self.joint_pos[env_ids] = state.joint_pos[env_ids]
    #     self.joint_vel[env_ids] = state.joint_vel[env_ids]

    #     self.body_pos_w[env_ids] = state.body_pos_w[env_ids] 
    #     self.body_quat_w[env_ids] = state.body_quat_w[env_ids]
    #     self.body_lin_vel_w[env_ids] = state.body_lin_vel_w[env_ids]
    #     self.body_ang_vel_w[env_ids] = state.body_ang_vel_w[env_ids]

    #     if self.key_body_pos_w is not None and state.key_body_pos_w is not None:
    #         self.key_body_pos_w[env_ids] = state.key_body_pos_w[env_ids]