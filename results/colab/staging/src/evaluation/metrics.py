"""
Evaluation metrics for 3D human pose generation, completion, denoising, and mesh recovery.

Implements:
- FID on 3D SMPL joint positions (Fréchet distance on Gaussian kinematics)
- APD (Average Pairwise Distance in cm)
- d_NN (Nearest-neighbor distance in radians on SO(3)^K)
- MPJPE and MPVPE (Mean Per Joint / Vertex Position Error in mm)
- PA-MPJPE and PA-MPVPE (Procrustes Aligned errors in mm)
- PCK@50 (Percentage of Correct Keypoints within 50 mm)
"""

import math
from typing import Optional, Tuple
import numpy as np
from scipy import linalg
import torch

from src.geometry.so3 import so3_geodesic_distance


def compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet distance between two multivariate Gaussians:
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary part cleanup
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_pose_fid(gen_joints: torch.Tensor, ref_joints: torch.Tensor) -> float:
    """
    Compute pose FID using 3D joint positions following NRDF protocol.
    Args:
        gen_joints: Generated 3D joints of shape (N, J, 3) in meters
        ref_joints: Reference real 3D joints of shape (M, J, 3) in meters
    Returns:
        Scalar FID value
    """
    n, j, _ = gen_joints.shape
    m = ref_joints.shape[0]

    gen_flat = gen_joints.reshape(n, j * 3).detach().cpu().numpy()
    ref_flat = ref_joints.reshape(m, j * 3).detach().cpu().numpy()

    mu_gen = np.mean(gen_flat, axis=0)
    sigma_gen = np.cov(gen_flat, rowvar=False)

    mu_ref = np.mean(ref_flat, axis=0)
    sigma_ref = np.cov(ref_flat, rowvar=False)

    return compute_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)


def compute_apd(joints: torch.Tensor) -> float:
    """
    Average Pairwise Distance (APD) in centimetres between 3D joint configurations.
    Args:
        joints: Tensor of shape (N, J, 3) in meters
    Returns:
        APD in cm (float)
    """
    n = joints.shape[0]
    if n <= 1:
        return 0.0

    # Flatten joints: (N, J*3)
    flat = joints.reshape(n, -1)
    diff = flat.unsqueeze(1) - flat.unsqueeze(0)  # (N, N, J*3)
    pdist = torch.norm(diff, p=2, dim=-1)         # (N, N)

    # Upper triangular mask (excluding diagonal)
    triu_idx = torch.triu_indices(n, n, offset=1)
    dists = pdist[triu_idx[0], triu_idx[1]]

    # Convert meters to cm
    apd_cm = dists.mean().item() * 100.0
    return apd_cm


def compute_d_nn(gen_rotations: torch.Tensor, ref_rotations: torch.Tensor, chunk_size: int = 500) -> float:
    """
    Mean nearest-neighbor distance (d_NN) in radians between generated and reference poses on SO(3)^K.
    Args:
        gen_rotations: (N, K, 3, 3)
        ref_rotations: (M, K, 3, 3)
    Returns:
        Mean nearest-neighbor distance in radians
    """
    n, k, _, _ = gen_rotations.shape
    m = ref_rotations.shape[0]

    min_dists = []
    # Process in chunks to prevent memory blowup
    for i in range(0, n, chunk_size):
        chunk_gen = gen_rotations[i:i + chunk_size]  # (B, K, 3, 3)
        b = chunk_gen.shape[0]

        # Expand for pairwise geodesic computation
        # (B, 1, K, 3, 3) vs (1, M, K, 3, 3)
        chunk_min = []
        for j in range(0, m, chunk_size):
            chunk_ref = ref_rotations[j:j + chunk_size]
            r1 = chunk_gen.unsqueeze(1)  # (B, 1, K, 3, 3)
            r2 = chunk_ref.unsqueeze(0)  # (1, B_ref, K, 3, 3)

            # Sum of geodesic angles across K joints divided by K (average joint angle in radians)
            dists = so3_geodesic_distance(r1, r2).mean(dim=-1)  # (B, B_ref)
            chunk_min.append(dists.min(dim=1).values)

        chunk_min_all = torch.stack(chunk_min, dim=1).min(dim=1).values
        min_dists.append(chunk_min_all)

    all_min = torch.cat(min_dists, dim=0)
    return all_min.mean().item()


def compute_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """
    Mean Per Joint Position Error in millimetres (mm).
    Args:
        pred_joints: (..., J, 3) in meters
        gt_joints: (..., J, 3) in meters
    Returns:
        MPJPE in mm
    """
    diff = pred_joints - gt_joints  # (..., J, 3)
    error = torch.norm(diff, p=2, dim=-1)  # (..., J)
    return (error.mean() * 1000.0).item()


def compute_mpvpe(pred_vertices: torch.Tensor, gt_vertices: torch.Tensor) -> float:
    """
    Mean Per Vertex Position Error in millimetres (mm).
    Args:
        pred_vertices: (..., V, 3) in meters
        gt_vertices: (..., V, 3) in meters
    Returns:
        MPVPE in mm
    """
    diff = pred_vertices - gt_vertices
    error = torch.norm(diff, p=2, dim=-1)
    return (error.mean() * 1000.0).item()


def batch_rigid_procrustes(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Rigid Procrustes Alignment (orthogonal transformation and translation) of pred onto gt.
    Args:
        pred: (B, N, 3)
        gt: (B, N, 3)
    Returns:
        Aligned pred of shape (B, N, 3)
    """
    b, n, _ = pred.shape
    # Center points
    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)

    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    # Correlation matrix: H = pred_centered^T * gt_centered
    h = torch.matmul(pred_centered.transpose(1, 2), gt_centered)  # (B, 3, 3)

    u, _, vt = torch.linalg.svd(h)
    v = vt.transpose(1, 2)
    ut = u.transpose(1, 2)

    # Det correction
    det = torch.linalg.det(torch.matmul(v, ut))
    d = torch.eye(3, device=pred.device, dtype=pred.dtype).repeat(b, 1, 1)
    d[:, 2, 2] = det

    r = torch.matmul(torch.matmul(v, d), ut)  # (B, 3, 3)
    aligned_pred = torch.matmul(pred_centered, r.transpose(1, 2)) + mu_gt
    return aligned_pred


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """
    Procrustes-Aligned MPJPE in mm.
    """
    b = pred_joints.shape[0] if pred_joints.dim() == 3 else 1
    p = pred_joints.view(b, -1, 3)
    g = gt_joints.view(b, -1, 3)

    aligned = batch_rigid_procrustes(p, g)
    return compute_mpjpe(aligned, g)


def compute_pa_mpvpe(pred_vertices: torch.Tensor, gt_vertices: torch.Tensor) -> float:
    """
    Procrustes-Aligned MPVPE in mm.
    """
    b = pred_vertices.shape[0] if pred_vertices.dim() == 3 else 1
    p = pred_vertices.view(b, -1, 3)
    g = gt_vertices.view(b, -1, 3)

    aligned = batch_rigid_procrustes(p, g)
    return compute_mpvpe(aligned, g)


def compute_pck_at_threshold(pred_joints: torch.Tensor, gt_joints: torch.Tensor, threshold_mm: float = 50.0, align: bool = True) -> float:
    """
    Percentage of Correct Keypoints within threshold_mm (default 50mm).
    Returns percentage in [0, 100].
    """
    b = pred_joints.shape[0] if pred_joints.dim() == 3 else 1
    p = pred_joints.view(b, -1, 3)
    g = gt_joints.view(b, -1, 3)

    if align:
        p = batch_rigid_procrustes(p, g)

    diff = torch.norm(p - g, p=2, dim=-1) * 1000.0  # (B, J) in mm
    correct = (diff <= threshold_mm).float()
    return (correct.mean() * 100.0).item()
