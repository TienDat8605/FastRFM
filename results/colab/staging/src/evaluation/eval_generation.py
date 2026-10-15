"""
Evaluation pipeline for unconditional pose generation (Table 1 reproduction).

Protocol:
- Sample 20 independent sets of 500 poses using Midpoint ODE solver (N=100 or N=1000).
- Compute FID, APD (cm), and d_NN (rad) against reference AMASS dataset.
- Aggregate mean and 95% confidence intervals.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.geometry.so3 import sample_uniform_so3
from src.geometry.smpl_utils import DifferentiableKinematics
from src.solvers.ode_solvers import ODESolver
from src.evaluation.metrics import compute_pose_fid, compute_apd, compute_d_nn


def run_generation_evaluation(
    model: nn.Module,
    ref_joints: torch.Tensor,
    ref_rotations: torch.Tensor,
    num_runs: int = 20,
    poses_per_run: int = 500,
    solver_steps: int = 100,
    solver_method: str = "midpoint",
    is_riemannian: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate unconditional generation across num_runs independent batches.
    Returns:
        dict with mean and 95% confidence interval half-width for FID, APD, and d_NN.
    """
    model.eval()
    kinematics = DifferentiableKinematics().to(device)
    solver = ODESolver(method=solver_method, num_steps=solver_steps, retraction="last_step")

    fid_list = []
    apd_list = []
    dnn_list = []

    with torch.no_grad():
        for run_idx in range(num_runs):
            # Sample initial noise on SO(3)^21
            x0 = sample_uniform_so3((poses_per_run, 21), device=device)

            # Integrate forward
            x1, _ = solver.integrate(model, x0, is_riemannian=is_riemannian)

            # Compute 3D joint positions
            gen_joints = kinematics(x1)  # (500, 22, 3)

            # Metrics
            fid = compute_pose_fid(gen_joints, ref_joints.to(device))
            apd = compute_apd(gen_joints)
            dnn = compute_d_nn(x1, ref_rotations.to(device))

            fid_list.append(fid)
            apd_list.append(apd)
            dnn_list.append(dnn)

    def get_stats(vals):
        arr = np.array(vals)
        mean = float(np.mean(arr))
        # 95% CI half-width
        sem = float(stats.sem(arr)) if len(arr) > 1 else 0.0
        ci95 = 1.96 * sem
        return mean, ci95

    mean_fid, ci_fid = get_stats(fid_list)
    mean_apd, ci_apd = get_stats(apd_list)
    mean_dnn, ci_dnn = get_stats(dnn_list)

    return {
        "FID": (mean_fid, ci_fid),
        "APD": (mean_apd, ci_apd),
        "d_NN": (mean_dnn, ci_dnn),
    }
