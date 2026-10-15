"""
SMPL kinematics and body model utilities.

Provides:
- Standard 21 body joint names and kinematic parent tree
- Differentiable Forward Kinematics (FK) from local rotation matrices
- Interface to smplx.SMPL body model
- Joint extraction and vertex computation
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# Standard SMPL 24 joints:
# 0: Pelvis (Root)
# 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 6: Spine2,
# 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot, 12: Neck,
# 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder,
# 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand

SMPL_JOINT_NAMES = [
    "pelvis",       # 0 (root)
    "left_hip",     # 1
    "right_hip",    # 2
    "spine1",       # 3
    "left_knee",    # 4
    "right_knee",   # 5
    "spine2",       # 6
    "left_ankle",   # 7
    "right_ankle",  # 8
    "spine3",       # 9
    "left_foot",    # 10
    "right_foot",   # 11
    "neck",         # 12
    "left_collar",  # 13
    "right_collar", # 14
    "head",         # 15
    "left_shoulder",# 16
    "right_shoulder",# 17
    "left_elbow",   # 18
    "right_elbow",  # 19
    "left_wrist",   # 20
    "right_wrist"   # 21
]

# The 21 body joints (excluding pelvis/root 0)
BODY_JOINT_NAMES = SMPL_JOINT_NAMES[1:]

# Standard SMPL kinematic tree parents for 22 joints (0 to 21)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

# Neutral template joint offsets in meters (canonical T-pose)
DEFAULT_OFFSETS = [
    [0.0, 0.0, 0.0],           # 0: Pelvis
    [0.08, -0.07, -0.02],      # 1: L_Hip
    [-0.08, -0.07, -0.02],     # 2: R_Hip
    [0.0, 0.10, -0.02],        # 3: Spine1
    [0.0, -0.38, 0.01],        # 4: L_Knee
    [0.0, -0.38, 0.01],        # 5: R_Knee
    [0.0, 0.12, 0.01],         # 6: Spine2
    [0.0, -0.40, -0.03],       # 7: L_Ankle
    [0.0, -0.40, -0.03],       # 8: R_Ankle
    [0.0, 0.11, 0.03],         # 9: Spine3
    [0.04, -0.06, 0.12],       # 10: L_Foot
    [-0.04, -0.06, 0.12],      # 11: R_Foot
    [0.0, 0.18, -0.02],        # 12: Neck
    [0.08, 0.11, -0.03],       # 13: L_Collar
    [-0.08, 0.11, -0.03],      # 14: R_Collar
    [0.0, 0.10, 0.05],         # 15: Head
    [0.10, -0.01, -0.01],      # 16: L_Shoulder
    [-0.10, -0.01, -0.01],     # 17: R_Shoulder
    [0.25, 0.0, 0.0],          # 18: L_Elbow
    [-0.25, 0.0, 0.0],         # 19: R_Elbow
    [0.24, 0.0, 0.0],          # 20: L_Wrist
    [-0.24, 0.0, 0.0],         # 21: R_Wrist
]


class DifferentiableKinematics(nn.Module):
    """
    Differentiable Forward Kinematics module computing 3D joint positions
    directly from 21 body rotation matrices (and optional root rotation/offsets).
    """

    def __init__(self, offsets: Optional[torch.Tensor] = None):
        super().__init__()
        if offsets is None:
            offsets = torch.tensor(DEFAULT_OFFSETS, dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.parents = SMPL_PARENTS

    def forward(
        self,
        body_rotations: torch.Tensor,
        root_rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute 3D joint positions from local joint rotations.
        Args:
            body_rotations: Tensor of shape (..., 21, 3, 3)
            root_rotation: Optional Tensor of shape (..., 3, 3). Default is Identity.
            translation: Optional Tensor of shape (..., 3). Default is zeros.
        Returns:
            Joint positions of shape (..., 22, 3) in meters.
        """
        batch_shape = body_rotations.shape[:-3]
        device = body_rotations.device
        dtype = body_rotations.dtype

        if root_rotation is None:
            root_rotation = torch.eye(3, dtype=dtype, device=device).expand(*batch_shape, 1, 3, 3)
        else:
            root_rotation = root_rotation.unsqueeze(-3)

        # Full 22 rotations: root + 21 body joints
        all_rotations = torch.cat([root_rotation, body_rotations], dim=-3)  # (..., 22, 3, 3)

        # Forward kinematics along the kinematic chain
        num_joints = len(self.parents)
        global_rotations = []
        global_positions = []

        offsets = self.offsets.expand(*batch_shape, num_joints, 3)

        for i in range(num_joints):
            parent = self.parents[i]
            local_rot = all_rotations[..., i, :, :]
            local_pos = offsets[..., i, :]

            if parent == -1:
                # Root joint
                glob_rot = local_rot
                glob_pos = local_pos
                if translation is not None:
                    glob_pos = glob_pos + translation
            else:
                parent_rot = global_rotations[parent]
                parent_pos = global_positions[parent]

                # R_glob = R_parent * R_local
                glob_rot = torch.matmul(parent_rot, local_rot)
                # p_glob = p_parent + R_parent * offset_local
                offset_rotated = torch.matmul(parent_rot, local_pos.unsqueeze(-1)).squeeze(-1)
                glob_pos = parent_pos + offset_rotated

            global_rotations.append(glob_rot)
            global_positions.append(glob_pos)

        joints = torch.stack(global_positions, dim=-2)  # (..., 22, 3)
        return joints
