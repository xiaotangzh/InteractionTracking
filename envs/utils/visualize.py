import torch
import torch
import matplotlib.pyplot as plt
import numpy as np
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.InteractionTracking.motions.assets.skeletons import *
from isaaclab_tasks.direct.InteractionTracking.utils.func import *

def reset_markers(markers: list):
    for marker in markers:
        marker.visualize(translations=torch.tensor([0,0,-1]).unsqueeze(0))

def visualize_markers(markers: VisualizationMarkers, target_pos: torch.Tensor | None):
    if target_pos is None:
        return
    elif all_zeros(target_pos): 
        reset_markers([markers])
        return
    markers.visualize(translations=target_pos.reshape(-1, 3))

def detect_contact(sensor: ContactSensor, robot: Articulation, body_order: list | None=None) -> torch.Tensor:
    # forces = sensor.data.net_forces_w # [num_envs, num_bodies, 3]
    forces = sensor.data.force_matrix_w # [num_envs, num_bodies, filtered, 3]
    force_mag = torch.linalg.norm(forces[:,:,0], dim=-1)  # [num_envs, num_bodies]

    if body_order is not None:
        force_mag = force_mag[:, body_order]

    threshold = 1e-6
    contact_mask = force_mag != 0.0  # bool  [num_envs, num_bodies]

    contact_pairs = torch.nonzero(contact_mask, as_tuple=False)  # [K, 2]
    contact_positions = robot.data.body_pos_w[contact_pairs[:, 0], contact_pairs[:, 1]]  # [K, 3]
    return contact_positions

def detect_close_joints(joints1: torch.Tensor, joints2: torch.Tensor, threshold: float=0.15):
    # pairwise distances [E, J, J]
    dists = torch.cdist(joints1, joints2, p=2)  

    # mask of close pairs [E, J, J]
    mask = dists < threshold  

    # mark joints in joints1 that have at least one close partner in joints2
    contact_mask = mask.any(dim=2)   # [E, J]

    # how many contacts in each env
    max_contacts = contact_mask.sum(dim=1).max().item()

    # expand mask for gathering
    expanded_mask = contact_mask.unsqueeze(-1).expand(-1, -1, 3)

    # zero out non-contact joints
    filtered = joints1 * expanded_mask  

    # sort so that nonzero contacts come first
    order = contact_mask.int().argsort(dim=1, descending=True)  # [E, J]
    gather_idx = order.unsqueeze(-1).expand(-1, -1, 3)
    sorted_contacts = torch.gather(filtered, 1, gather_idx)

    # keep only up to max_contacts
    contacts = sorted_contacts[:, :max_contacts, :]  

    return contacts

fig_motion_pie, ax_motion_pie = plt.subplots()
fig_tensor_norm, ax_tensor_norm = plt.subplots()
plt.ion()  # turn on interactive mode

def animate_tensor_norm_distribution(frame: torch.Tensor, joint_names: list[str] | None = None, title: str = "Tensor Norm Distribution"):
    """
    Animate the distribution of norms of a [24, 3] tensor in real-time.
    
    Args:
        tensor_24x3 (torch.Tensor): Tensor of shape [24, 3] 
        joint_names (list[str], optional): Names for the 24 joints/elements
        title (str): Title for the plot
    """
    if frame.shape != (24, 3):
        print(f"Warning: Expected tensor shape [24, 3], got {frame.shape}")
        return
    
    # Compute norms along the last dimension (3D vectors)
    norms = torch.norm(frame, dim=-1)  # Shape: [24]
    values = norms.detach().cpu().float().numpy()
    
    # Check for invalid values
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return
    
    # Generate labels for the 24 elements
    if joint_names is not None and len(joint_names) == 24:
        labels = joint_names
    else:
        labels = [f"Joint {i}" for i in range(24)]
    
    indices = np.arange(24)
    
    # Clear and update the plot
    ax_tensor_norm.clear()
    bars = ax_tensor_norm.bar(indices, values, alpha=0.7, color='steelblue')
    
    # Customize the plot
    ax_tensor_norm.set_xlabel('Joint Index')
    ax_tensor_norm.set_ylabel('Norm Value')
    ax_tensor_norm.set_title(title)
    ax_tensor_norm.set_xticks(indices)
    ax_tensor_norm.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax_tensor_norm.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=6)
    
    # Add statistics
    mean_val = np.mean(values)
    max_val = np.max(values)
    min_val = np.min(values)
    std_val = np.std(values)
    
    stats_text = f"Mean: {mean_val:.3f}\nMax: {max_val:.3f}\nMin: {min_val:.3f}\nStd: {std_val:.3f}"
    ax_tensor_norm.text(0.02, 0.98, stats_text, transform=ax_tensor_norm.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    fig_tensor_norm.canvas.draw()
    plt.pause(0.001)

def animate_motion_terminate_ratio(motion_terminate_memory: torch.Tensor, motion_names: dict | None = None):
    values = motion_terminate_memory.detach().cpu().float().numpy()
    total = values.sum()

    if total <= 0 or np.isnan(total) or np.isinf(total):
        return
    else:
        ratios = values / total
        ratios[np.isnan(ratios)] = 0
        ratios[np.isinf(ratios)] = 0

    labels = [motion_names.get(i, f"ID {i}") for i in range(len(values))]

    ax_motion_pie.clear()
    ax_motion_pie.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=90)
    ax_motion_pie.set_title("Motion Termination Ratio")
    fig_motion_pie.canvas.draw()
    plt.pause(0.001)



# Initialize two interactive figures once
def _init_bar_plots():
    global fig1, ax1, fig2, ax2
    # Create first figure if not exists or closed
    if 'fig1' not in globals() or not plt.fignum_exists(fig1.number):
        fig1, ax1 = plt.subplots()
        plt.ion()
    # Create second figure if not exists or closed
    if 'fig2' not in globals() or not plt.fignum_exists(fig2.number):
        fig2, ax2 = plt.subplots()
        plt.ion()
    return (fig1, ax1), (fig2, ax2)
def animate_two_bars(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    motion_names: dict[int, str] | None = None,
    labels: tuple[str, str] = ("Tensor 1", "Tensor 2")
):
    """
    Plot two tensors in separate bar charts in real time, using two figures.

    Args:
        tensor1 (torch.Tensor): First tensor of shape (num_motions,).
        tensor2 (torch.Tensor): Second tensor of shape (num_motions,).
        motion_names (dict[int, str], optional): Mapping from index to label.
        labels (tuple[str, str], optional): Titles for the two figures.
    """
    # Ensure two figures
    (fig1, ax1), (fig2, ax2) = _init_bar_plots()

    # Convert to numpy
    v1 = tensor1.detach().cpu().float().numpy()
    v2 = tensor2.detach().cpu().float().numpy()

    # Validate dimensions and values
    if v1.size != v2.size:
        raise ValueError("Both tensors must have the same length")
    if np.any(np.isnan(v1)) or np.any(np.isinf(v1)) or np.any(np.isnan(v2)) or np.any(np.isinf(v2)):
        return

    n = v1.size
    indices = np.arange(n)
    width = 0.5  # full width for single bars in each figure

    # Generate labels
    x_labels = [motion_names.get(i, f"ID {i}") for i in indices] if motion_names else [f"ID {i}" for i in indices]

    # Plot tensor1 in figure 1
    ax1.clear()
    ax1.bar(indices, v1, width)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylabel("Value")
    ax1.set_title(labels[0])

    # Plot tensor2 in figure 2
    ax2.clear()
    ax2.bar(indices, v2, width)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_ylabel("Value")
    ax2.set_title(labels[1])

    # Render updates
    fig1.canvas.draw()
    fig2.canvas.draw()
    plt.pause(0.001)



def visualize_motion_3d(motion_tensor, bones=SMPL_BONES, joint_names=SMPL_JOINT_NAMES, 
                       joint_dict=SMPL_JOINT_IDX, fps=30.0, title="3D Motion Visualization"):
    """
    Visualize skeletal motion in 3D space with animated bones and joints.
    
    Args:
        motion_tensor: Tensor of shape [frames, joints, 3] containing joint positions over time
        bones: List of (parent, child) tuples defining bone connections
        joint_names: List of joint names corresponding to tensor indices
        joint_dict: Dictionary mapping joint names to indices
        fps: Frames per second for animation playback
        title: Title for the plot window
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import torch
    
    # Convert to numpy if needed
    if isinstance(motion_tensor, torch.Tensor):
        coords = motion_tensor.detach().cpu().numpy()
    else:
        coords = np.array(motion_tensor)
    
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"Expected tensor shape [frames, joints, 3], got {coords.shape}")
    
    frames, num_joints, _ = coords.shape
    print(f"Visualizing motion: {frames} frames, {num_joints} joints")
    
    # Set up the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate global bounds for consistent scaling
    all_coords = coords.reshape(-1, 3)
    x_range = [all_coords[:, 0].min(), all_coords[:, 0].max()]
    y_range = [all_coords[:, 1].min(), all_coords[:, 1].max()]
    z_range = [all_coords[:, 2].min(), all_coords[:, 2].max()]
    
    # Add some margin
    margin = 0.1
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0] 
    z_span = z_range[1] - z_range[0]
    max_span = max(x_span, y_span, z_span)
    
    center_x = (x_range[0] + x_range[1]) / 2
    center_y = (y_range[0] + y_range[1]) / 2
    center_z = (z_range[0] + z_range[1]) / 2
    
    half_range = max_span / 2 + margin
    
    plt.ion()  # Turn on interactive mode
    
    try:
        frame = 0
        while True:
            # Check if figure window is still open
            if not plt.fignum_exists(fig.number):
                print("\nAnimation window closed by user")
                break
            
            # Get current frame data
            current_coords = coords[frame % frames]
            
            # Clear the plot
            ax.clear()
            
            # Plot joints as points
            ax.scatter(current_coords[:, 0], current_coords[:, 1], current_coords[:, 2], 
                      c='red', s=50, alpha=0.8, label='Joints')
            
            # Plot bones as lines with different colors
            bone_colors = plt.cm.tab10(np.linspace(0, 1, len(bones)))
            
            for i, (parent, child) in enumerate(bones):
                if parent in joint_dict and child in joint_dict:
                    pid = joint_dict[parent]
                    cid = joint_dict[child]
                    
                    if pid < num_joints and cid < num_joints:
                        p1 = current_coords[pid]
                        p2 = current_coords[cid]
                        
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=bone_colors[i % len(bone_colors)], linewidth=3, alpha=0.8)
            
            # Add joint labels (optional, can be toggled)
            show_labels = num_joints <= 24  # Only show labels for reasonable number of joints
            if show_labels:
                for i, name in enumerate(joint_names):
                    if i < num_joints:
                        ax.text(current_coords[i, 0], current_coords[i, 1], current_coords[i, 2], 
                               name, fontsize=6, alpha=0.7)
            
            # Set consistent axis limits and labels
            ax.set_xlim(center_x - half_range, center_x + half_range)
            ax.set_ylim(center_y - half_range, center_y + half_range)
            ax.set_zlim(center_z - half_range, center_z + half_range)
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            ax.set_title(f'{title} - Frame {frame + 1}/{frames}')
            
            # Add frame info and controls
            info_text = f"Frame: {frame + 1}/{frames}\nFPS: {fps}\nPress Ctrl+C to stop or close window"
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Update display
            plt.draw()
            plt.pause(1.0 / fps)
            
            frame += 1
            
    except KeyboardInterrupt:
        print("\nAnimation stopped by user")
    finally:
        plt.ioff()
        try:
            plt.close(fig)
        except Exception:
            pass


def visualize_motion_3d_comparison(motion1, motion2, bones=SMPL_BONES, 
                                       joint_names=SMPL_JOINT_NAMES, joint_dict=SMPL_JOINT_IDX,
                                       fps=30.0, labels=("Motion 1", "Motion 2")):
    """
    Compare two motion sequences side by side in 3D.
    
    Args:
        motion1: First motion tensor [frames, joints, 3]
        motion2: Second motion tensor [frames, joints, 3]  
        bones: List of (parent, child) tuples
        joint_names: List of joint names
        joint_dict: Dictionary mapping joint names to indices
        fps: Frames per second for animation
        labels: Tuple of labels for the two motions
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import torch
    
    # Convert to numpy
    if isinstance(motion1, torch.Tensor):
        coords1 = motion1.detach().cpu().numpy()
    else:
        coords1 = np.array(motion1)
        
    if isinstance(motion2, torch.Tensor):
        coords2 = motion2.detach().cpu().numpy()
    else:
        coords2 = np.array(motion2)
    
    frames1, frames2 = coords1.shape[0], coords2.shape[0]
    max_frames = max(frames1, frames2)
    
    # Set up dual plot
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Calculate shared bounds
    all_coords = np.vstack([coords1.reshape(-1, 3), coords2.reshape(-1, 3)])
    center = all_coords.mean(axis=0)
    ranges = all_coords.max(axis=0) - all_coords.min(axis=0)
    max_range = ranges.max() / 2 + 0.1
    
    plt.ion()
    
    try:
        frame = 0
        while True:
            # Check if figure window is still open
            if not plt.fignum_exists(fig.number):
                print("\nComparison animation window closed by user")
                break
                
            # Get current frames (loop if needed)
            frame1_idx = frame % frames1
            frame2_idx = frame % frames2
            
            current_coords1 = coords1[frame1_idx]
            current_coords2 = coords2[frame2_idx]
            
            # Clear plots
            ax1.clear()
            ax2.clear()
            
            # Plot first motion
            ax1.scatter(current_coords1[:, 0], current_coords1[:, 1], current_coords1[:, 2], 
                       c='red', s=50, alpha=0.8)
            
            for parent, child in bones:
                if parent in joint_dict and child in joint_dict:
                    pid, cid = joint_dict[parent], joint_dict[child]
                    if pid < len(current_coords1) and cid < len(current_coords1):
                        p1, p2 = current_coords1[pid], current_coords1[cid]
                        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                'b-', linewidth=2, alpha=0.8)
            
            # Plot second motion  
            ax2.scatter(current_coords2[:, 0], current_coords2[:, 1], current_coords2[:, 2], 
                       c='red', s=50, alpha=0.8)
            
            for parent, child in bones:
                if parent in joint_dict and child in joint_dict:
                    pid, cid = joint_dict[parent], joint_dict[child]
                    if pid < len(current_coords2) and cid < len(current_coords2):
                        p1, p2 = current_coords2[pid], current_coords2[cid]
                        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                'g-', linewidth=2, alpha=0.8)
            
            # Set consistent limits and labels
            for ax, label in [(ax1, labels[0]), (ax2, labels[1])]:
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(center[2] - max_range, center[2] + max_range)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'{label} - Frame {frame + 1}')
            
            plt.draw()
            plt.pause(1.0 / fps)
            frame += 1
            
    except KeyboardInterrupt:
        print("\nComparison animation stopped by user")
    finally:
        plt.ioff()
        try:
            plt.close(fig)
        except Exception:
            pass