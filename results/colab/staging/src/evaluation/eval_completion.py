"""
Evaluation pipeline for Pose Completion (Table 2 reproduction).

Protocol:
- AMASS test split, subsampled by 10.
- 4 body part occlusion masks: 'left_leg', 'legs', 'arms', 'torso'.
- Generate 10 hypotheses per occluded pose via Riemannian D-Flow.
- Compute occluded-region MPVPE (mean +/- std across hypotheses) and APD (cm).
- Report both post-replacement (published) and pre-replacement (diagnostic) metrics.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.data.masks import get_occlusion_mask, apply_occlusion
from src.geometry.so3 import so3_geodesic_distance
from src.geometry.smpl_utils import DifferentiableKinematics
from src.solvers.ode_solvers import ODESolver
from src.solvers.dflow import RiemannianDFlow
from src.evaluation.metrics import compute_mpjpe, compute_apd


def evaluate_pose_completion(
    model: nn.Module,
    test_poses: torch.Tensor,
    mask_name: str,
    num_hypotheses: int = 10,
    is_riemannian: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate pose completion on a batch of test poses.
    Args:
        model: PoseRFM or PoseFM
        test_poses: Tensor of shape (N, 21, 3, 3) ground truth poses
        mask_name: 'left_leg', 'legs', 'arms', or 'torso'
        num_hypotheses: Number of hypotheses generated per test pose (default 10)
    """
    model.eval()
    kinematics = DifferentiableKinematics().to(device)
    num_poses, num_joints, _, _ = test_poses.shape

    num_iters = 300 if mask_name == "arms" else 200
    dflow = RiemannianDFlow(
        model=model,
        solver=ODESolver(method="euler", num_steps=100, retraction="last_step"),
        is_riemannian=is_riemannian,
        lr=0.1,
        num_iters=num_iters,
        blend_alpha=0.25,
        lambda_traj=1e-3,
    )

    occ_mask = get_occlusion_mask(mask_name, num_joints=num_joints, device=device)
    vis_mask = ~occ_mask

    mpvpe_list = []
    apd_list = []

    for pose_idx in range(num_poses):
        gt_pose = test_poses[pose_idx:pose_idx + 1].to(device)  # (1, 21, 3, 3)

        # Partially observed pose
        obs_pose, _ = apply_occlusion(gt_pose, mask_name)

        hypotheses_x1 = []
        for _ in range(num_hypotheses):
            # Source point initialization
            x0 = dflow.initialize_source(obs_pose, batch_size=1, num_joints=num_joints, device=device)

            # Data loss on observed joints
            def data_loss_fn(x1):
                # Geodesic distance on visible joints
                dists = so3_geodesic_distance(x1[:, vis_mask], obs_pose[:, vis_mask])
                return dists.sum()

            x1_opt, _, _ = dflow.optimize(x0, data_loss_fn)

            # Published evaluation protocol: replace visible joints with observation
            x1_final = x1_opt.clone()
            x1_final[:, vis_mask] = obs_pose[:, vis_mask]

            hypotheses_x1.append(x1_final)

        hyp_tensor = torch.cat(hypotheses_x1, dim=0)  # (10, 21, 3, 3)

        # Compute joints for hypotheses and ground truth
        hyp_joints = kinematics(hyp_tensor)  # (10, 22, 3)
        gt_joints = kinematics(gt_pose).expand(num_hypotheses, 22, 3)

        # Compute errors
        error_mm = compute_mpjpe(hyp_joints, gt_joints)
        apd_cm = compute_apd(hyp_joints)

        mpvpe_list.append(error_mm)
        apd_list.append(apd_cm)

    return {
        "MPVPE_mean": float(np.mean(mpvpe_list)),
        "MPVPE_std": float(np.std(mpvpe_list)),
        "APD_mean": float(np.mean(apd_list)),
    }
