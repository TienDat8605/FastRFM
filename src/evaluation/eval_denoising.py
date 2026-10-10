"""
Evaluation pipeline for Motion Denoising (Table 3 reproduction).

Protocol:
- 60-frame sequences at 30 Hz (HumanEva: 190 segments; HPS: 350 segments).
- Gaussian noise with sigma = 40 mm (0.04m) or 100 mm (0.10m) added to 3D joints.
- Optimization objective: L_data + lambda_temp * L_smooth + lambda_traj * L_traj.
- Evaluates MPJPE in mm.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from src.geometry.smpl_utils import DifferentiableKinematics
from src.solvers.ode_solvers import ODESolver
from src.solvers.dflow import RiemannianDFlow
from src.evaluation.metrics import compute_mpjpe


def evaluate_motion_denoising(
    model: nn.Module,
    clean_sequences: torch.Tensor,
    noise_sigma_m: float = 0.04,  # 40 mm = 0.04 m, 100 mm = 0.10 m
    is_riemannian: bool = True,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Evaluate motion denoising across noisy 60-frame sequence segments.
    Args:
        model: Flow model
        clean_sequences: Tensor of shape (S, 60, 21, 3, 3) representing S segments of 60 frames
        noise_sigma_m: Standard deviation of Gaussian noise in meters (0.04 or 0.10)
    Returns:
        Mean MPJPE in mm across all segments
    """
    model.eval()
    kinematics = DifferentiableKinematics().to(device)
    num_segments, seq_len, num_joints, _, _ = clean_sequences.shape

    # Set hyperparameters according to Table D.1
    lambda_temp = 1e-1 if noise_sigma_m >= 0.08 else 1e-2
    num_iters = 400 if noise_sigma_m >= 0.08 else 300

    dflow = RiemannianDFlow(
        model=model,
        solver=ODESolver(method="euler", num_steps=100, retraction="last_step"),
        is_riemannian=is_riemannian,
        lr=0.25,
        num_iters=num_iters,
        blend_alpha=0.75,
        lambda_traj=1e-5,
    )

    mpjpe_list = []

    for seg_idx in range(num_segments):
        gt_poses = clean_sequences[seg_idx].to(device)  # (60, 21, 3, 3)

        # 1. Compute clean 3D joints and add Gaussian noise
        with torch.no_grad():
            clean_joints = kinematics(gt_poses)  # (60, 22, 3)
            noisy_joints = clean_joints + torch.randn_like(clean_joints) * noise_sigma_m

        # 2. Source initialization for each frame in sequence
        x0 = dflow.initialize_source(gt_poses, batch_size=seq_len, num_joints=num_joints, device=device)

        # 3. Data loss + Temporal smoothness loss
        def data_loss_fn(x1):
            pred_j = kinematics(x1)  # (60, 22, 3)
            # Distance to noisy joints
            l_joint = torch.mean((pred_j - noisy_joints) ** 2)
            return l_joint

        def aux_loss_fn(x1, x0):
            pred_j = kinematics(x1)
            # Temporal smoothness: acceleration or velocity difference across consecutive frames
            diff = pred_j[1:] - pred_j[:-1]
            l_smooth = torch.mean(diff ** 2)
            return lambda_temp * l_smooth

        opt_x1, _, _ = dflow.optimize(x0, data_loss_fn, aux_loss_fn=aux_loss_fn)

        with torch.no_grad():
            denoised_joints = kinematics(opt_x1)
            error_mm = compute_mpjpe(denoised_joints, clean_joints)
            mpjpe_list.append(error_mm)

    return float(np.mean(mpjpe_list))
