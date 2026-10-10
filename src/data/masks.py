"""
Body part occlusion masks and joint groupings for Pose Completion.

Covers the 4 benchmark conditions:
- left_leg
- legs
- arms
- torso
Joint indices are 0-indexed relative to the 21 body joints (excluding pelvis/root).
"""

from typing import Dict, List, Optional, Set, Tuple
import torch

# Body joint indexing (21 body joints, excluding pelvis):
# 0: left_hip
# 1: right_hip
# 2: spine1
# 3: left_knee
# 4: right_knee
# 5: spine2
# 6: left_ankle
# 7: right_ankle
# 8: spine3
# 9: left_foot
# 10: right_foot
# 11: neck
# 12: left_collar
# 13: right_collar
# 14: head
# 15: left_shoulder
# 16: right_shoulder
# 17: left_elbow
# 18: right_elbow
# 19: left_wrist
# 20: right_wrist

OCCLUSION_JOINT_INDICES: Dict[str, List[int]] = {
    # Left leg: left hip (0), left knee (3), left ankle (6), left foot (9)
    "left_leg": [0, 3, 6, 9],

    # Both legs: left/right hip (0, 1), knee (3, 4), ankle (6, 7), foot (9, 10)
    "legs": [0, 1, 3, 4, 6, 7, 9, 10],

    # Both arms: left/right collar (12, 13), shoulder (15, 16), elbow (17, 18), wrist (19, 20)
    "arms": [12, 13, 15, 16, 17, 18, 19, 20],

    # Torso: spine1 (2), spine2 (5), spine3 (8), neck (11)
    "torso": [2, 5, 8, 11],
}


def get_occlusion_mask(mask_name: str, num_joints: int = 21, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Get a boolean mask of shape (num_joints,) where True indicates OCCLUDED (missing),
    and False indicates VISIBLE (observed).
    """
    if mask_name not in OCCLUSION_JOINT_INDICES:
        raise ValueError(f"Unknown occlusion mask '{mask_name}'. Supported: {list(OCCLUSION_JOINT_INDICES.keys())}")

    occluded_indices = OCCLUSION_JOINT_INDICES[mask_name]
    mask = torch.zeros(num_joints, dtype=torch.bool, device=device)
    mask[occluded_indices] = True
    return mask


def apply_occlusion(
    full_poses: torch.Tensor,
    mask_name: str,
    fill_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply synthetic occlusion to a batch of full poses.
    Args:
        full_poses: Tensor of shape (B, 21, 3, 3) in SO(3)
        mask_name: One of 'left_leg', 'legs', 'arms', 'torso'
        fill_value: Tensor of shape (21, 3, 3) or (B, 21, 3, 3) to fill occluded joints (e.g., mean pose or Identity).
                    Default is Identity matrix.
    Returns:
        obs_poses: Poses with occluded joints replaced by fill_value
        occ_mask: Boolean tensor of shape (21,) where True = occluded
    """
    b, k, _, _ = full_poses.shape
    device = full_poses.device
    dtype = full_poses.dtype

    occ_mask = get_occlusion_mask(mask_name, num_joints=k, device=device)

    if fill_value is None:
        fill_value = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(k, 3, 3)

    if fill_value.dim() == 3:
        fill_value = fill_value.unsqueeze(0).expand(b, k, 3, 3)

    obs_poses = full_poses.clone()
    obs_poses[:, occ_mask, :, :] = fill_value[:, occ_mask, :, :]

    return obs_poses, occ_mask
