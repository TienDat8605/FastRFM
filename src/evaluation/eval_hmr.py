"""
Evaluation pipeline for Human Mesh Recovery / Inverse Kinematics (Tables 4 & 5).

Protocol:
- EHF dataset (100 images).
- ViTPose-H 2D keypoints; from scratch vs. CLIFF initialization.
- Geman-McClure robust reprojection loss with confidence weights.
- Metrics: PA-MPJPE (mm), PA-MPVPE (mm), PCK@50 (%).
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.geometry.smpl_utils import DifferentiableKinematics
from src.solvers.ode_solvers import ODESolver
from src.solvers.dflow import RiemannianDFlow
from src.evaluation.metrics import compute_pa_mpjpe, compute_pck_at_threshold


def geman_mcclure_loss(diff: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    """
    Robust Geman-McClure error function rho(e) = (e^2) / (e^2 + sigma^2).
    """
    e_sq = torch.sum(diff ** 2, dim=-1)
    return e_sq / (e_sq + sigma ** 2)


def project_2d(joints_3d: torch.Tensor, camera_scale: torch.Tensor, camera_trans: torch.Tensor) -> torch.Tensor:
    """
    Orthographic / weak perspective projection of 3D joints into 2D image coordinates.
    """
    xy = joints_3d[..., :2]
    proj_2d = camera_scale.unsqueeze(-1) * xy + camera_trans.unsqueeze(-2)
    return proj_2d


def evaluate_hmr(
    model: nn.Module,
    gt_joints: torch.Tensor,
    init_poses: torch.Tensor,
    keypoints_2d: torch.Tensor,
    confidences: torch.Tensor,
    is_riemannian: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Run HMR optimization on images/keypoints and report PA-MPJPE and PCK@50.
    """
    model.eval()
    kinematics = DifferentiableKinematics().to(device)
    num_samples = gt_joints.shape[0]

    dflow = RiemannianDFlow(
        model=model,
        solver=ODESolver(method="euler", num_steps=100, retraction="last_step"),
        is_riemannian=is_riemannian,
        lr=0.1,
        num_iters=400,
        blend_alpha=0.75,
        lambda_traj=50.0,
    )

    pred_joints_list = []

    for i in range(num_samples):
        init_pose = init_poses[i:i + 1].to(device)
        kp_2d = keypoints_2d[i:i + 1].to(device)
        conf = confidences[i:i + 1].to(device)

        # Learnable camera scale and translation for SMPLify
        cam_scale = torch.tensor([1.0], device=device, requires_grad=True)
        cam_trans = torch.zeros(1, 2, device=device, requires_grad=True)

        x0 = dflow.initialize_source(init_pose, batch_size=1, num_joints=21, device=device)

        def data_loss_fn(x1):
            j3d = kinematics(x1)  # (1, 22, 3)
            # Take relevant 2D joints
            j2d = project_2d(j3d, cam_scale, cam_trans)
            diff = j2d - kp_2d
            robust_err = geman_mcclure_loss(diff)
            loss_2d = torch.mean(conf * robust_err)
            return loss_2d

        opt_x1, _, _ = dflow.optimize(x0, data_loss_fn)

        with torch.no_grad():
            opt_j3d = kinematics(opt_x1)
            pred_joints_list.append(opt_j3d)

    all_pred = torch.cat(pred_joints_list, dim=0)
    all_gt = gt_joints.to(device)

    pa_mpjpe = compute_pa_mpjpe(all_pred, all_gt)
    pck_50 = compute_pck_at_threshold(all_pred, all_gt, threshold_mm=50.0)

    return {
        "PA-MPJPE": pa_mpjpe,
        "PCK@50": pck_50,
    }
