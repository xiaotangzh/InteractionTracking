import torch
from isaaclab.utils.math import *

def get_unit_vector(target_direction, device):
    if target_direction == "forward":
        target_direction = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0)
        idx = 0
    elif target_direction == "leftward":
        target_direction = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)
        idx = 1
    elif target_direction == "upward":
        target_direction = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0)
        idx = 2
    return target_direction, idx

def transform_quat_to_target_direction(current_quat, target_direction) -> torch.Tensor:
    current_quat = current_quat / torch.norm(current_quat, dim=-1, keepdim=True)
    current_direction = quat_apply(current_quat, target_direction)
    return current_direction

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

def quat_diff(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_rot = quat_mul(target, quat_inv(input))
    diff_angle = axis_angle_from_quat(diff_rot)
    return diff_angle

@torch.jit.script
def quat_to_forward_up(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w first) to forward (x) and upward (z) vectors.

    Args:
        q: (*, 4) quaternion tensor [w, x, y, z]
    Returns:
        forward_up: (*, 6) tensor concatenating forward (x) and upward (z) vectors
    """

    # Normalize quaternion to ensure unit length
    q = torch.nn.functional.normalize(q, dim=-1)

    w, x, y, z = q.unbind(-1)

    # Build rotation matrix (column vectors are rotated basis axes)
    # This follows the standard quaternion-to-rotation-matrix conversion
    R = torch.stack([
        torch.stack([
            1 - 2*(y**2 + z**2),  2*(x*y - z*w),      2*(x*z + y*w)
        ], dim=-1),
        torch.stack([
            2*(x*y + z*w),        1 - 2*(x**2 + z**2), 2*(y*z - x*w)
        ], dim=-1),
        torch.stack([
            2*(x*z - y*w),        2*(y*z + x*w),      1 - 2*(x**2 + y**2)
        ], dim=-1),
    ], dim=-2)  # shape (..., 3, 3)

    # Extract desired rotated basis vectors
    # Assuming local axes:
    #   x-axis = forward direction
    #   z-axis = upward direction
    forward = R[..., 0]   # rotated X-axis
    upward  = R[..., 2]   # rotated Z-axis

    # Concatenate into 6D representation
    forward_up = torch.cat([forward, upward], dim=-1)
    return forward_up

@torch.jit.script
def forward_up_to_quat(forward_up: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D [forward(+X), up(+Z)] to quaternion [w, x, y, z].
    Assumes columns of R are [X, Y, Z] and we set:
        R[:,0] = forward, R[:,2] = up, R[:,1] = up × forward
    """
    f = forward_up[..., :3]
    u = forward_up[..., 3:]

    # orthonormalize
    f = torch.nn.functional.normalize(f, dim=-1)
    u = u - (f * u).sum(-1, keepdim=True) * f
    u = torch.nn.functional.normalize(u, dim=-1)

    # Y-axis (right/side) to complete a right-handed basis
    y = torch.cross(u, f, dim=-1)        # y = z × x

    # rotation matrix with columns [x, y, z] = [f, y, u]
    R = torch.stack([f, y, u], dim=-1)   # (..., 3, 3)

    # robust matrix → quat (w first)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    denom = 4.0 * qw + 1e-8
    qx = (R[..., 2, 1] - R[..., 1, 2]) / denom
    qy = (R[..., 0, 2] - R[..., 2, 0]) / denom
    qz = (R[..., 1, 0] - R[..., 0, 1]) / denom

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return torch.nn.functional.normalize(q, dim=-1)

def quat_diff_inverse(input: torch.Tensor, diff_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(diff_angle, dim=-1)
    axis = torch.where(
        angle.unsqueeze(-1) > 1e-6,
        diff_angle / angle.unsqueeze(-1),
        torch.zeros_like(diff_angle)
    )
    diff_rot = quat_from_angle_axis(angle, axis)
    target = quat_mul(diff_rot, input)
    return target


def rotation_around_up_axis(q_from: torch.Tensor, q_to: torch.Tensor, up_axis: int = 2) -> torch.Tensor:
    """Compute signed rotation about the global up axis between two orientations.

    Args:
        q_from: Initial orientation quaternion(s) in ``(w, x, y, z)`` format. Shape ``(..., 4)``.
        q_to: Target orientation quaternion(s) in ``(w, x, y, z)`` format. Shape ``(..., 4)``.
        up_axis: Index of the global up axis. Defaults to ``2`` (Z-up).

    Returns:
        Tensor of shape ``(...)`` containing yaw angles in radians within ``[-pi, pi]`` that
        rotate ``q_from`` around the up axis to align with ``q_to``.
    """

    if up_axis != 2:
        raise NotImplementedError("rotation_around_up_axis currently supports only Z-up (axis index 2).")

    squeeze_output = False
    if q_from.ndim == 1:
        q_from = q_from.unsqueeze(0)
        squeeze_output = True

    if q_to.ndim == 1:
        expand_shape = q_from.shape
        view_shape = (1,) * (q_from.ndim - 1) + (4,)
        q_to = q_to.view(view_shape).expand(expand_shape)
    elif q_to.shape[:-1] != q_from.shape[:-1]:
        raise ValueError("q_to must either be a single quaternion or match q_from's batch shape.")

    q_from = q_from / q_from.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    q_to = q_to / q_to.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    delta = quat_mul(q_to, quat_inv(q_from))
    w, x, y, z = torch.unbind(delta, dim=-1)

    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = wrap_to_pi(yaw)

    if squeeze_output:
        return yaw.squeeze(0)
    return yaw


def rotate_around_up_axis(q_from: torch.Tensor, q_to: torch.Tensor, up_axis: int = 2) -> torch.Tensor:
    """Rotate ``q_from`` around the up axis so its yaw aligns with ``q_to``.

    Args:
        q_from: Source orientation quaternion(s) in ``(w, x, y, z)`` format. Shape ``(..., 4)``.
        q_to: Target orientation quaternion(s) in ``(w, x, y, z)`` format. Shape ``(..., 4)``.
        up_axis: Index of the up axis (0=x, 1=y, 2=z). Defaults to 2 (Z-up).

    Returns:
        Quaternions with the same roll and pitch as ``q_from`` but yaw aligned to ``q_to``.
    """

    if up_axis != 2:
        raise NotImplementedError("rotate_around_up_axis currently supports only Z-up (axis index 2).")

    single_input = False
    if q_from.ndim == 1:
        q_from = q_from.unsqueeze(0)
        single_input = True

    if q_to.ndim == 1:
        expand_shape = q_from.shape
        view_shape = (1,) * (q_from.ndim - 1) + (4,)
        q_to = q_to.view(view_shape).expand(expand_shape)
    elif q_to.shape[:-1] != q_from.shape[:-1]:
        raise ValueError("q_to must either be a single quaternion or match q_from's batch shape.")

    yaw = rotation_around_up_axis(q_from, q_to, up_axis)

    half = 0.5 * yaw
    rot = torch.zeros_like(q_from)
    rot[..., 0] = torch.cos(half)

    axis = torch.zeros_like(q_from[..., 1:])
    axis[..., up_axis] = torch.sin(half)
    rot[..., 1:] = axis

    rotated = quat_mul(rot, q_from)

    if single_input:
        return rotated.squeeze(0)
    return rotated


def mse(input: torch.Tensor, target: torch.Tensor | None = None, start_dim: int = 0) -> torch.Tensor:
    if target is None:
        squared_error = input ** 2
    else:
        squared_error = (input - target) ** 2
        
    reduce_dims = tuple(range(start_dim, input.ndim))
    return torch.mean(squared_error, dim=reduce_dims)

def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mean + std * eps
    return z

def axis_angle_add(original: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Compute target axis-angle by applying an offset rotation to an original rotation.
    
    Args:
        original: Original rotation as axis-angle vector(s). Shape ``(..., 3)``.
        offset: Offset rotation as axis-angle vector(s). Shape ``(..., 3)``.
        
    Returns:
        Target axis-angle vector(s) representing the combined rotation. Shape ``(..., 3)``.
    """
    # Get angle magnitudes
    original_angle = torch.norm(original, dim=-1, keepdim=True)
    offset_angle = torch.norm(offset, dim=-1, keepdim=True)
    
    # Get normalized axes (handle zero case)
    original_axis = torch.where(
        original_angle > 1e-6,
        original / original_angle,
        torch.zeros_like(original)
    )
    offset_axis = torch.where(
        offset_angle > 1e-6,
        offset / offset_angle,
        torch.zeros_like(offset)
    )
    
    # Convert to quaternions manually: q = [cos(θ/2), sin(θ/2) * axis]
    original_quat = torch.cat([
        torch.cos(original_angle / 2),
        torch.sin(original_angle / 2) * original_axis
    ], dim=-1)
    
    offset_quat = torch.cat([
        torch.cos(offset_angle / 2),
        torch.sin(offset_angle / 2) * offset_axis
    ], dim=-1)
    
    # Apply the rotation: target = offset * original
    target_quat = quat_mul(offset_quat, original_quat)
    
    # Convert quaternion back to axis-angle
    # For quaternion [w, x, y, z], angle = 2*acos(w), axis = [x,y,z]/sin(angle/2)
    w = target_quat[..., 0:1]
    xyz = target_quat[..., 1:]
    
    target_angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    sin_half = torch.sin(target_angle / 2)
    
    target_axis = torch.where(
        sin_half.abs() > 1e-6,
        xyz / sin_half,
        torch.zeros_like(xyz)
    )
    
    target_axis_angle = target_angle * target_axis
    
    return target_axis_angle