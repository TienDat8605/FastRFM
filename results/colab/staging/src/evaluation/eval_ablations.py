"""
Evaluation pipeline for Geometric Ablations (Table 6 reproduction).

Ablates:
1. Model: PoseFM, PoseFM + geometry, PoseRFM
2. Geodesic loss: On vs. Off (MSE)
3. Trajectory regularizer: On vs. Off
Tasks: Left leg and legs completion on AMASS test split.
"""

from typing import Dict, List
import pandas as pd
import torch
import torch.nn as nn

from src.data.masks import get_occlusion_mask, apply_occlusion
from src.geometry.so3 import so3_geodesic_distance
from src.geometry.smpl_utils import DifferentiableKinematics
from src.solvers.ode_solvers import ODESolver
from src.solvers.dflow import RiemannianDFlow
from src.evaluation.metrics import compute_mpjpe, compute_apd

ABLATION_CONFIGS = [
    # (Model Type, Geo Loss, Traj Loss)
    ("PoseFM", False, False),
    ("PoseFM", True, False),
    ("PoseFM", False, True),
    ("PoseFM", True, True),
    ("PoseFM+geom", False, False),
    ("PoseFM+geom", True, False),
    ("PoseFM+geom", False, True),
    ("PoseFM+geom", True, True),
    ("PoseRFM", False, False),
    ("PoseRFM", True, False),
    ("PoseRFM", False, True),
    ("PoseRFM", True, True),
]


def run_ablation_matrix(
    models: Dict[str, nn.Module],
    test_poses: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """
    Execute all 12 ablation combinations on left leg and legs occlusion.
    """
    records = []
    kinematics = DifferentiableKinematics().to(device)

    for model_name, use_geo_loss, use_traj_loss in ABLATION_CONFIGS:
        model = models[model_name]
        model.eval()

        is_riemannian = (model_name == "PoseRFM")
        lambda_traj = 1e-3 if use_traj_loss else 0.0

        dflow = RiemannianDFlow(
            model=model,
            solver=ODESolver(method="euler", num_steps=100, retraction="last_step"),
            is_riemannian=is_riemannian,
            lr=0.1,
            num_iters=200,
            blend_alpha=0.25,
            lambda_traj=lambda_traj,
        )

        row = {
            "Model": model_name,
            "Geo Loss": "Yes" if use_geo_loss else "No",
            "Traj Loss": "Yes" if use_traj_loss else "No",
        }

        for mask_name in ["left_leg", "legs"]:
            occ_mask = get_occlusion_mask(mask_name, num_joints=21, device=device)
            vis_mask = ~occ_mask

            mpvpe_vals = []
            apd_vals = []

            for i in range(len(test_poses)):
                gt_pose = test_poses[i:i + 1].to(device)
                obs_pose, _ = apply_occlusion(gt_pose, mask_name)

                x0 = dflow.initialize_source(obs_pose, batch_size=1, num_joints=21, device=device)

                def data_loss_fn(x1):
                    if use_geo_loss:
                        return so3_geodesic_distance(x1[:, vis_mask], obs_pose[:, vis_mask]).sum()
                    else:
                        # MSE in flattened matrix space
                        return torch.mean((x1[:, vis_mask] - obs_pose[:, vis_mask]) ** 2)

                x1_opt, _, _ = dflow.optimize(x0, data_loss_fn)
                x1_final = x1_opt.clone()
                x1_final[:, vis_mask] = obs_pose[:, vis_mask]

                j_hyp = kinematics(x1_final)
                j_gt = kinematics(gt_pose)

                mpvpe_vals.append(compute_mpjpe(j_hyp, j_gt))
                apd_vals.append(compute_apd(j_hyp))

            row[f"{mask_name}_MPVPE"] = float(pd.Series(mpvpe_vals).mean())
            row[f"{mask_name}_APD"] = float(pd.Series(apd_vals).mean())

        records.append(row)

    return pd.DataFrame(records)
