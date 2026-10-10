"""Evaluation subpackage for metrics and benchmark tasks."""

from src.evaluation.metrics import (
    compute_pose_fid,
    compute_apd,
    compute_d_nn,
    compute_mpjpe,
    compute_mpvpe,
    compute_pa_mpjpe,
    compute_pa_mpvpe,
    compute_pck_at_threshold,
)
from src.evaluation.eval_generation import run_generation_evaluation
from src.evaluation.eval_completion import evaluate_pose_completion
from src.evaluation.eval_denoising import evaluate_motion_denoising
from src.evaluation.eval_hmr import evaluate_hmr
from src.evaluation.eval_ablations import run_ablation_matrix

__all__ = [
    "compute_pose_fid",
    "compute_apd",
    "compute_d_nn",
    "compute_mpjpe",
    "compute_mpvpe",
    "compute_pa_mpjpe",
    "compute_pa_mpvpe",
    "compute_pck_at_threshold",
    "run_generation_evaluation",
    "evaluate_pose_completion",
    "evaluate_motion_denoising",
    "evaluate_hmr",
    "run_ablation_matrix",
]
