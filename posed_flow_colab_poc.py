#!/usr/bin/env python3
"""
PoseD-Flow & Few-Step Riemannian D-Flow: Colab Research Scaffold
=============================================================================
Research scaffold using shared, tested SO(3) geometry. Includes full-graph Euler,
few-step midpoint, and an explicitly heuristic surrogate gradient benchmark.
Midpoint is a solver baseline, not a trained consistency model. No speedup or
adjoint correctness claim is implied by this scaffold.
"""

import argparse
import copy
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from src.study import geometry as geometry


# =============================================================================
# 1. Differential Geometry Engine on Product Manifold M = SO(3)^K
# =============================================================================

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Maps a vector v in R^(..., 3) to a skew-symmetric matrix in so(3) subset R^(..., 3, 3).
    """
    zeros = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zeros, -vz, vy], dim=-1)
    row1 = torch.stack([vz, zeros, -vx], dim=-1)
    row2 = torch.stack([-vy, vx, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee(W: torch.Tensor) -> torch.Tensor:
    """
    Maps a skew-symmetric matrix W in so(3) subset R^(..., 3, 3) to a vector in R^(..., 3).
    """
    vx = 0.5 * (W[..., 2, 1] - W[..., 1, 2])
    vy = 0.5 * (W[..., 0, 2] - W[..., 2, 0])
    vz = 0.5 * (W[..., 1, 0] - W[..., 0, 1])
    return torch.stack([vx, vy, vz], dim=-1)


def so3_exp_map(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Lie group exponential map exp: so(3) -> SO(3) via Rodrigues' formula with
    Taylor expansion near theta = 0 for smooth, NaN-free backpropagation.
    Input W: skew-symmetric matrix in R^(..., 3, 3).
    """
    return geometry.exp(vee(W))


def so3_log_map(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Lie group logarithmic map log: SO(3) -> so(3) returning a skew-symmetric matrix.
    Input R: rotation matrix in SO(3) subset R^(..., 3, 3).
    """
    return hat(geometry.log(R))


def proj_tangent(R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection Pi_{T_R SO(3)^K}(G) of an ambient matrix G in R^(..., K, 3, 3)
    onto the tangent space T_R SO(3)^K = { R * W : W^T = -W }.
    """
    skew = 0.5 * (torch.matmul(R.transpose(-1, -2), G) - torch.matmul(G.transpose(-1, -2), R))
    return torch.matmul(R, skew)


def exp_x(R: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Riemannian exponential map Exp_R(V) on SO(3)^K (Eq. 3).
    R: base point in SO(3)^K, shape (..., K, 3, 3)
    V: tangent vector in T_R SO(3)^K, shape (..., K, 3, 3)
    """
    W = torch.matmul(R.transpose(-1, -2), V)
    W = 0.5 * (W - W.transpose(-1, -2))
    return torch.matmul(R, so3_exp_map(W))


def log_x(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Riemannian logarithmic map Log_{R1}(R2) in T_{R1} SO(3)^K (Eq. 2).
    Returns tangent vector at R1 pointing along the geodesic toward R2.
    """
    R_rel = torch.matmul(R1.transpose(-1, -2), R2)
    W = so3_log_map(R_rel)
    return torch.matmul(R1, W)


def parallel_transport(R_old: torch.Tensor, R_new: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Levi-Civita parallel transport PT_{R_old -> R_new}(V) on SO(3)^K
    used in Riemannian Adam (Algorithm 3, Line 3).
    """
    body = vee(R_old.transpose(-1, -2) @ V)
    displacement = geometry.log(R_old.transpose(-1, -2) @ R_new)
    transported = (geometry.exp(-displacement / 2) @ body[..., None]).squeeze(-1)
    return R_new @ hat(transported)


def riemannian_inner(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Canonical bi-invariant Riemannian metric <U, V>_R = 0.5 * tr(U^T V) summed over K joints.
    Returns shape (B,).
    """
    return 0.5 * torch.sum(U * V, dim=(-3, -2, -1))


def joint_geodesic_angles(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes per-joint geodesic angles d_{SO(3)}(R1_k, R2_k) in radians, shape (..., K).
    Matches the term inside Eq. 2 and Eq. 20: arccos((tr(R1^T R2) - 1) / 2).
    """
    return geometry.log(R1.transpose(-1, -2) @ R2).norm(dim=-1)


def product_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Product manifold distance d_M(x, x') = || (d(R1_1, R2_1), ..., d(R1_K, R2_K)) ||_2 (Eq. 2).
    Returns shape (...,).
    """
    angles = joint_geodesic_angles(R1, R2)
    return torch.linalg.norm(angles, dim=-1)


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """
    Projects arbitrary 3x3 matrices onto SO(3) via Special Procrustes / SVD
    (used for the Euclidean PoseFM baseline).
    """
    return geometry.project(M)


def sample_uniform_so3(batch_size: int, num_joints: int, device: torch.device) -> torch.Tensor:
    """
    Samples uniform random rotations x_0 ~ p(x_0) on SO(3)^K using QR decomposition
    of standard Gaussian matrices (Haar measure on SO(3)).
    """
    z = torch.randn(batch_size, num_joints, 3, 3, device=device)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    ph = d / torch.abs(d)
    q = q * ph.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[..., :, 0] = q[..., :, 0] * torch.where(det < 0, -1.0, 1.0).unsqueeze(-1)
    return q


# =============================================================================
# 2. Synthetic & Real Articulated Pose Dataset (Zero-Download or .npz)
# =============================================================================

def generate_correlated_pose_dataset(
    num_samples: int,
    num_joints: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generates a multimodal, strongly correlated articulated pose distribution on SO(3)^K
    that mimics anatomical joint couplings (e.g., knee-hip flexion, bilateral symmetry,
    or finger curl synergies in MANO).
    Allows immediate proof-of-concept testing on Colab without downloading AMASS.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    num_modes = 6
    latent_dim = 4  # Low-dimensional anatomical synergy subspace

    # Create mode centers and coupling projection matrices
    mode_centers = 0.6 * torch.randn(num_modes, num_joints, 3, generator=gen, device=device)
    synergy_basis = 0.45 * torch.randn(num_modes, latent_dim, num_joints, 3, generator=gen, device=device)

    mode_ids = torch.randint(0, num_modes, (num_samples,), generator=gen, device=device)
    z = torch.randn(num_samples, latent_dim, generator=gen, device=device)
    noise = 0.06 * torch.randn(num_samples, num_joints, 3, generator=gen, device=device)

    # Combine mode center + coupled synergy + small anatomical noise in axis-angle space
    centers = mode_centers[mode_ids]  # (N, K, 3)
    basis = synergy_basis[mode_ids]    # (N, latent_dim, K, 3)
    coupled = torch.einsum("nl,nlkc->nkc", z, basis)
    axis_angles = centers + coupled + noise

    # Map axis-angle vectors to SO(3)^K via exponential map at identity
    W = hat(axis_angles)
    return so3_exp_map(W)


def load_or_create_dataset(
    data_path: Optional[str],
    num_joints: int,
    num_train: int,
    num_test: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads real 3D joint rotations from a .npz file (supporting shape (N, K, 3, 3)
    or axis-angle (N, K*3) / (N, K, 3) from AMASS, HumanEva, or FreiHAND)
    or falls back to the correlated synthetic articulated manifold.
    """
    if data_path is not None:
        raw = np.load(data_path)
        if "poses" not in raw:
            raise ValueError(f"Expected explicit poses array in {data_path}")
        key = "poses"
        arr = torch.tensor(raw[key], dtype=torch.float32, device=device)
        if arr.ndim == 2 and arr.shape[-1] == num_joints * 3:
            aa = arr.reshape(-1, num_joints, 3)
            poses = so3_exp_map(hat(aa))
        elif arr.ndim == 2 and num_joints == 21 and arr.shape[-1] in (72, 156):
            aa = arr[:, 3:66].reshape(-1, 21, 3)
            poses = so3_exp_map(hat(aa))
        elif arr.ndim == 3 and arr.shape[-2:] == (num_joints, 3):
            poses = so3_exp_map(hat(arr[:, :num_joints, :]))
        elif arr.ndim == 4 and arr.shape[-3:] == (num_joints, 3, 3):
            poses = arr[:, :num_joints]
        else:
            raise ValueError(f"Unsupported pose array shape {arr.shape} in {data_path}")
        if len(poses) < num_train + num_test:
            raise ValueError(f"Need {num_train + num_test} disjoint train/test poses; got {len(poses)}")
        perm = torch.randperm(poses.shape[0], device=device)
        poses = poses[perm]
        return poses[:num_train], poses[num_train : num_train + num_test]

    all_poses = generate_correlated_pose_dataset(num_train + num_test, num_joints, device=device)
    return all_poses[:num_train], all_poses[num_train:]


# =============================================================================
# 3. PoseRFM & Euclidean PoseFM Network Architecture (Section 4, p. 4)
# =============================================================================

class PoseFlowMLP(nn.Module):
    """
    Research MLP architecture (checkpoint compatibility depends on configuration):
      - Input: pose x_t in SO(3)^K flattened to 9*K concatenated with time t in [0, 1]
      - 4 hidden layers with hidden dimension 512 and Swish (SiLU) activation
      - Output projected onto the tangent space T_{x_t} SO(3)^K for PoseRFM
        (or kept in ambient R^(K x 3 x 3) for the Euclidean PoseFM baseline).
    """

    def __init__(self, num_joints: int = 21, hidden_dim: int = 512, riemannian: bool = True):
        super().__init__()
        self.num_joints = num_joints
        self.riemannian = riemannian
        in_dim = num_joints * 9 + 1
        out_dim = num_joints * (3 if riemannian else 9)

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(3):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, K, 3, 3)
        t:   (B, 1) or scalar tensor in [0, 1]
        Returns velocity field v_w(x_t, t) of shape (B, K, 3, 3).
        """
        B, K, _, _ = x_t.shape
        if t.ndim == 0:
            t = t.expand(B, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)

        flat_x = x_t.reshape(B, K * 9)
        inp = torch.cat([flat_x, t], dim=-1)
        raw_out = self.net(inp)
        if self.riemannian:
            return x_t @ hat(raw_out.reshape(B, K, 3))
        return raw_out.reshape(B, K, 3, 3)


# =============================================================================
# 4. Algorithm 1: PoseRFM & PoseFM Training
# =============================================================================

def train_flow_model(
    train_poses: torch.Tensor,
    num_joints: int,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    ema_decay: float = 0.99,
    riemannian: bool = True,
    verbose: bool = True,
) -> PoseFlowMLP:
    """
    Implements Algorithm 1 (PoseRFM training, p. 3-4) with AdamW, EMA (0.99),
    and ReduceLROnPlateau scheduler.
    """
    device = train_poses.device
    model = PoseFlowMLP(num_joints=num_joints, hidden_dim=hidden_dim, riemannian=riemannian).to(device)
    ema_model = copy.deepcopy(model).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    N = train_poses.shape[0]
    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x1 = train_poses[idx]
        x0 = sample_uniform_so3(batch_size, num_joints, device=device)

        # Sample t ~ U(0, 1 - 1e-3) to avoid division by zero at t = 1 in Eq. (9)
        t = torch.rand(batch_size, 1, device=device) * 0.995
        t_view = t.view(batch_size, 1, 1, 1)

        if riemannian:
            # Algorithm 1, Line 6: x_t <- Exp_{x_0}(t * Log_{x_0}(x_1))
            v_0to1 = log_x(x0, x1)
            xt = exp_x(x0, t_view * v_0to1)
            # Equation (9): u_t(x_t | x_1) = Log_{x_t}(x_1) / (1 - t)
            ut_target = log_x(xt, x1) / (1.0 - t_view)
            vt_pred = model(xt, t)
            # Algorithm 1, Line 7: Riemannian metric norm ||v_w(x_t, t) - u_t(x_t | x_1)||_g^2
            diff = vt_pred - ut_target
            loss = torch.mean(riemannian_inner(diff, diff))
        else:
            # Euclidean Flow Matching baseline (PoseFM)
            xt = (1.0 - t_view) * x0 + t_view * x1
            ut_target = x1 - x0
            vt_pred = model(xt, t)
            loss = torch.mean(torch.sum((vt_pred - ut_target) ** 2, dim=(-3, -2, -1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Update Exponential Moving Average (EMA = 0.99, Section 4, p. 4)
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        running_loss += loss.item()
        if step % max(1, steps // 10) == 0:
            avg_loss = running_loss / max(1, steps // 10)
            scheduler.step(avg_loss)
            if verbose:
                tag = "PoseRFM (Riemannian)" if riemannian else "PoseFM (Euclidean)"
                print(
                    f"[{tag}] Step {step:5d}/{steps} | Loss: {avg_loss:8.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | Elapsed: {time.time() - t0:.1f}s"
                )
            running_loss = 0.0

    return ema_model


# =============================================================================
# 5. Unconditional ODE Sampling (Euler & 2nd-Order Midpoint on SO(3)^K)
# =============================================================================

@torch.no_grad()
def sample_unconditional(
    model: PoseFlowMLP,
    num_samples: int,
    ode_steps: int = 50,
    solver: str = "midpoint",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Integrates the flow ODE from t = 0 to t = 1 on SO(3)^K (Section 4 & 5.1).
    Supports both 'euler' and 'midpoint' (used in Section 5.1, p. 6).
    """
    K = model.num_joints
    xt = sample_uniform_so3(num_samples, K, device=device)
    dt = 1.0 / ode_steps

    for i in range(ode_steps):
        t_cur = torch.full((num_samples, 1), i * dt, device=device)
        if model.riemannian:
            v1 = model(xt, t_cur)
            if solver == "midpoint":
                x_mid = exp_x(xt, (0.5 * dt) * v1)
                t_mid = torch.full((num_samples, 1), (i + 0.5) * dt, device=device)
                v_mid = model(x_mid, t_mid)
                xt = exp_x(xt, dt * (xt @ (x_mid.transpose(-1, -2) @ v_mid)))
            else:
                xt = exp_x(xt, dt * v1)
        else:
            v1 = model(xt, t_cur)
            if solver == "midpoint":
                x_mid = xt + (0.5 * dt) * v1
                t_mid = torch.full((num_samples, 1), (i + 0.5) * dt, device=device)
                v_mid = model(x_mid, t_mid)
                xt = xt + dt * v_mid
            else:
                xt = xt + dt * v1

    if not model.riemannian:
        xt = project_to_so3(xt)
    return xt


# =============================================================================
# 6. Algorithm 2 & 3: Riemannian D-Flow & Fast Inference Extensions
# =============================================================================

def riemannian_adam_step(
    xk: torch.Tensor,
    mk: torch.Tensor,
    vk: torch.Tensor,
    ambient_grad: torch.Tensor,
    step_idx: int,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementation inspired by Algorithm 3 (RAdam_step on SO(3)^K, p. 4):
      Line 2: g_k <- grad_{x_k} L := Pi_{T_{x_k} M}(nabla_{x_k} L)
      Line 3: m_k <- PT_{x_{k-1} -> x_k}(m_k)  (handled upon retraction)
      Line 4: m_{k+1} <- beta_1 m_k + (1 - beta_1) g_k
      Line 5: v_{k+1} <- beta_2 v_k + (1 - beta_2) <g_k, g_k>_{x_k}
      Line 6-8: bias-corrected step Delta x_k
      Line 9: x_{k+1} <- Exp_{x_k}(Delta x_k)
    """
    # Line 2: Project ambient gradient onto tangent space T_{x_k} M
    gk = proj_tangent(xk, ambient_grad)

    # Line 4: First moment in tangent space
    mk_next = beta1 * mk + (1.0 - beta1) * gk

    # Line 5: Second moment using Riemannian metric norm per sample
    g_norm2 = riemannian_inner(gk, gk).view(-1, 1, 1, 1)
    vk_next = beta2 * vk + (1.0 - beta2) * g_norm2

    # Lines 6-7: Bias correction
    m_hat = mk_next / (1.0 - beta1 ** step_idx)
    v_hat = vk_next / (1.0 - beta2 ** step_idx)

    # Line 8-9: Retraction via Exponential map and parallel transport of momentum
    delta_x = -lr * m_hat / (torch.sqrt(v_hat) + eps)
    xk_next = exp_x(xk, delta_x)
    mk_transported = parallel_transport(xk, xk_next, mk_next)

    return xk_next.detach(), mk_transported.detach(), vk_next.detach()


def pose_completion_loss(
    x1: torch.Tensor,
    x_obs: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Equation (20) (Pose completion geodesic loss, p. 6):
      L_data(x) = sum_{k in Omega} arccos((tr(x_k^T x_k^obs) - 1) / 2)
    visible_mask: boolean/float mask of shape (K,) where 1 = observed, 0 = occluded.
    """
    angles = joint_geodesic_angles(x1, x_obs)  # (B, K)
    masked_angles = angles * visible_mask.unsqueeze(0)
    return torch.sum(masked_angles, dim=-1).mean()


def trajectory_regularizer(trajectory_steps: list) -> torch.Tensor:
    """
    Implements Equation (11) (Trajectory regularizer, p. 4):
      R := L_traj = sum_{i=1}^K sum_{k=1}^N (3 - tr(x_{ik}))
    where x_{ik} is the post-step state rotation at integration step k.
    The sum is deliberately not normalized by the number of steps.
    """
    reg = 0.0
    for rel_rot in trajectory_steps:
        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        reg = reg + torch.mean(torch.sum(3.0 - trace, dim=-1))
    return reg


def guided_pose_completion(
    model: PoseFlowMLP,
    x_obs: torch.Tensor,
    visible_mask: torch.Tensor,
    mode: str = "vanilla_dflow",
    ode_steps: int = 40,
    opt_iters: int = 50,
    lr: float = 0.08,
    traj_weight: float = 0.02,
    initial_source: Optional[torch.Tensor] = None,
    warm_start: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Source optimization; returns batch-average milliseconds per pose and peak MB.

    surrogate_pullback uses a heuristic transported terminal gradient with one
    local Jacobian correction. It is not a Riemannian adjoint implementation and
    does not differentiate trajectory regularization.
    """
    if mode not in ("euclidean_dflow", "vanilla_dflow", "fast_midpoint", "surrogate_pullback"):
        raise ValueError(f"Unknown guidance mode: {mode}")
    if ode_steps < 1 or opt_iters < 0:
        raise ValueError("ode_steps must be positive and opt_iters nonnegative")
    model.eval().requires_grad_(False)
    device = x_obs.device
    B, K, _, _ = x_obs.shape

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Initialize source point x_0 in SO(3)^K
    x0 = (sample_uniform_so3(B, K, device=device) if initial_source is None
          else initial_source.detach().clone().to(x_obs))
    if x0.shape != x_obs.shape:
        raise ValueError("initial_source shape must match observations")

    # Warm-start source initialization for fast modes: slerp visible joints halfway toward x_obs
    if warm_start:
        vis_view = visible_mask.view(1, K, 1, 1)
        v_init = log_x(x0, x_obs)
        x0 = exp_x(x0, 0.35 * vis_view * v_init).detach()

    mk = torch.zeros_like(x0)
    vk = torch.zeros((B, 1, 1, 1), device=device)

    if mode == "euclidean_dflow":
        x0_param = x0.clone().detach().requires_grad_(True)
        euc_opt = torch.optim.Adam([x0_param], lr=lr)

    dt = 1.0 / ode_steps

    for it in range(1, opt_iters + 1):
        if mode == "euclidean_dflow":
            euc_opt.zero_grad()
            xt = x0_param
            for s in range(ode_steps):
                t_cur = torch.full((B, 1), s * dt, device=device)
                xt = xt + dt * model(xt, t_cur)
            x1_proj = project_to_so3(xt)
            loss = pose_completion_loss(x1_proj, x_obs, visible_mask)
            loss.backward()
            euc_opt.step()

        elif mode in ("vanilla_dflow", "fast_midpoint"):
            x0_req = x0.clone().detach().requires_grad_(True)
            xt = x0_req
            rel_rotations = []

            for s in range(ode_steps):
                t_cur = torch.full((B, 1), s * dt, device=device)
                v1 = model(xt, t_cur)
                if mode == "fast_midpoint":
                    x_mid = exp_x(xt, (0.5 * dt) * v1)
                    t_mid = torch.full((B, 1), (s + 0.5) * dt, device=device)
                    step_v = dt * (xt @ (x_mid.transpose(-1, -2) @ model(x_mid, t_mid)))
                else:
                    step_v = dt * v1

                # Retraction followed by recording the post-step state for Eq. (11)
                W_step = torch.matmul(xt.transpose(-1, -2), step_v)
                W_step = 0.5 * (W_step - W_step.transpose(-1, -2))
                rel_rot = so3_exp_map(W_step)
                xt = torch.matmul(xt, rel_rot)
                rel_rotations.append(xt)

            loss_data = pose_completion_loss(xt, x_obs, visible_mask)
            loss_traj = trajectory_regularizer(rel_rotations)
            total_loss = loss_data + traj_weight * loss_traj

            ambient_grad = torch.autograd.grad(total_loss, x0_req)[0]
            x0, mk, vk = riemannian_adam_step(x0, mk, vk, ambient_grad, step_idx=it, lr=lr)

        elif mode == "surrogate_pullback":
            # Heuristic surrogate pullback (not an adjoint):
            # 1) Integrate ODE forward without storing intermediate autograd graph (O(1) memory)
            with torch.no_grad():
                xt = x0.clone()
                for s in range(ode_steps):
                    t_cur = torch.full((B, 1), s * dt, device=device)
                    v1 = model(xt, t_cur)
                    x_mid = exp_x(xt, (0.5 * dt) * v1)
                    t_mid = torch.full((B, 1), (s + 0.5) * dt, device=device)
                    xt = exp_x(xt, dt * (xt @ (x_mid.transpose(-1, -2) @ model(x_mid, t_mid))))

            # 2) Compute terminal Riemannian gradient grad_{x_1} L(x_1)
            x1_leaf = xt.detach().requires_grad_(True)
            loss_data = pose_completion_loss(x1_leaf, x_obs, visible_mask)
            grad_x1_ambient = torch.autograd.grad(loss_data, x1_leaf)[0]
            grad_x1_tan = proj_tangent(x1_leaf, grad_x1_ambient)

            # 3) Pull back terminal gradient to T_{x_0} M via parallel transport +
            #    one-step heuristic Jacobian correction; no adjoint guarantee
            with torch.no_grad():
                pulled_grad = parallel_transport(x1_leaf, x0, grad_x1_tan)

            x0_req = x0.clone().detach().requires_grad_(True)
            t_zero = torch.zeros((B, 1), device=device)
            v0 = model(x0_req, t_zero)
            x1_one_step = exp_x(x0_req, v0)
            surrogate_loss = torch.sum(x1_one_step * pulled_grad.detach()) / B
            jac_grad = torch.autograd.grad(surrogate_loss, x0_req)[0]

            combined_grad = pulled_grad + 0.5 * jac_grad
            x0, mk, vk = riemannian_adam_step(x0, mk, vk, combined_grad, step_idx=it, lr=lr)

    # Final forward ODE pass to produce completed poses x_1
    with torch.no_grad():
        xt = x0_param if mode == "euclidean_dflow" else x0
        for s in range(ode_steps):
            t_cur = torch.full((B, 1), s * dt, device=device)
            if model.riemannian:
                if mode in ("fast_midpoint", "surrogate_pullback"):
                    x_mid = exp_x(xt, 0.5 * dt * model(xt, t_cur))
                    t_mid = torch.full((B, 1), (s + 0.5) * dt, device=device)
                    velocity = xt @ (x_mid.transpose(-1, -2) @ model(x_mid, t_mid))
                else:
                    velocity = model(xt, t_cur)
                xt = exp_x(xt, dt * velocity)
            else:
                xt = xt + dt * model(xt, t_cur)
        if not model.riemannian:
            xt = project_to_so3(xt)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_mem_mb = 0.0

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0 / B
    return xt, elapsed_ms, peak_mem_mb


# =============================================================================
# 7. Evaluation Metrics (Section 5: d_NN, APD, Masked Geodesic Error)
# =============================================================================

@torch.no_grad()
def evaluate_unconditional_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes nearest-neighbor geodesic distance d_NN (rad, lower is better)
    and Average Pairwise Distance APD (rad, higher means more diverse) from Table 1.
    """
    N_gen = generated.shape[0]
    # Pairwise geodesic distances between generated and reference sets
    gen_exp = generated.unsqueeze(1)   # (N_gen, 1, K, 3, 3)
    ref_exp = reference.unsqueeze(0)   # (1, N_ref, K, 3, 3)
    dists = product_geodesic_distance(gen_exp, ref_exp)  # (N_gen, N_ref)
    d_nn = torch.min(dists, dim=1).values.mean().item()

    # Average Pairwise Distance (APD) within generated samples
    sub = generated[: min(100, N_gen)]
    p_dists = product_geodesic_distance(sub.unsqueeze(1), sub.unsqueeze(0))
    mask = ~torch.eye(sub.shape[0], dtype=torch.bool, device=generated.device)
    apd = p_dists[mask].mean().item()

    return {"d_NN_rad": d_nn, "APD_rad": apd}


@torch.no_grad()
def evaluate_completion_accuracy(
    completed: torch.Tensor,
    ground_truth: torch.Tensor,
    visible_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes mean per-joint rotational geodesic error (in radians and degrees)
    specifically on the OCCLUDED joints (1 - visible_mask), plus full-body error.
    """
    per_joint_err = joint_geodesic_angles(completed, ground_truth)  # (B, K)
    occ_mask = 1.0 - visible_mask
    occ_err_rad = (per_joint_err * occ_mask.unsqueeze(0)).sum(dim=-1) / occ_mask.sum().clamp(min=1.0)
    full_err_rad = per_joint_err.mean(dim=-1)

    return {
        "occluded_err_deg": math.degrees(occ_err_rad.mean().item()),
        "full_body_err_deg": math.degrees(full_err_rad.mean().item()),
    }


# =============================================================================
# 8. Main Experiment Runner (Generates Your Advisor Pitch Table)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PoseD-Flow & Fast Riemannian D-Flow PoC")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full", "inference", "pullback_screen"],
                        help="demo/full train controls; inference benchmarks few-step students and a discrete adjoint.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional path to .npz pose file (e.g., FreiHAND or HumanEva).")
    parser.add_argument("--num_joints", type=int, default=15,
                        help="Number of articulated SO(3) joints (15 for MANO hand, 21 for SMPL body).")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training iterations.")
    parser.add_argument("--validation_path", type=str, default=None,
                        help="Separate validation NPZ; required for inference mode.")
    parser.add_argument("--output_dir", type=str, default="results/colab_inference",
                        help="Checkpoint, prediction, timing, and report directory.")
    parser.add_argument("--model_root", type=str, default=None,
                        help="Root containing models/smpl/SMPL_NEUTRAL.pkl; otherwise discovered above data_path.")
    parser.add_argument("--distill_steps", type=int, default=500)
    parser.add_argument("--opt_iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse_training_dir", type=str, default=None,
                        help="Reuse this PoC's compatible trained prior/students after an audit-only fix.")
    parser.add_argument("--teacher_run", type=str)
    parser.add_argument("--budget_hours", type=float, default=4)
    args = parser.parse_args()
    if args.mode == "pullback_screen":
        from src.acceleration.pullback_screen import run
        run(args)
        return

    if args.mode == "inference":
        from src.acceleration.colab_benchmark import run
        run(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"PoseD-Flow Research PoC | Device: {device} | Manifold: SO(3)^{args.num_joints}")
    print("=" * 78)

    if args.mode == "demo":
        train_steps = args.steps or 1200
        batch_size = 256
        hidden_dim = 256
        num_train, num_test = 4096, 128
    else:
        train_steps = args.steps or 50000
        batch_size = 4096
        hidden_dim = 512
        num_train, num_test = 32768, 500

    # 1. Load or synthesize articulated pose dataset
    train_poses, test_poses = load_or_create_dataset(
        args.data_path, args.num_joints, num_train=num_train, num_test=num_test, device=device
    )
    print(f"Loaded dataset: train={tuple(train_poses.shape)}, test={tuple(test_poses.shape)}\n")

    # 2. Train PoseRFM (Riemannian Flow Matching) and PoseFM (Euclidean Baseline)
    print(">>> Training PoseRFM (Riemannian Flow Matching on SO(3)^K)...")
    poserfm = train_flow_model(
        train_poses,
        num_joints=args.num_joints,
        steps=train_steps,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        riemannian=True,
    )

    print("\n>>> Training PoseFM (Euclidean Flow Matching Baseline)...")
    posefm = train_flow_model(
        train_poses,
        num_joints=args.num_joints,
        steps=train_steps,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        riemannian=False,
    )

    # 3. Evaluate Table 1: Unconditional Pose Generation
    print("\n>>> Evaluating Unconditional Pose Generation (Table 1 metrics)...")
    gen_rfm = sample_unconditional(poserfm, num_samples=128, ode_steps=50, solver="midpoint", device=device)
    gen_fm = sample_unconditional(posefm, num_samples=128, ode_steps=50, solver="midpoint", device=device)
    m_rfm = evaluate_unconditional_metrics(gen_rfm, test_poses)
    m_fm = evaluate_unconditional_metrics(gen_fm, test_poses)

    print(f"  PoseFM  (Euclidean)  | Fidelity d_NN: {m_fm['d_NN_rad']:.4f} rad | Diversity APD: {m_fm['APD_rad']:.4f} rad")
    print(f"  PoseRFM (Riemannian) | Fidelity d_NN: {m_rfm['d_NN_rad']:.4f} rad | Diversity APD: {m_rfm['APD_rad']:.4f} rad")

    # 4. Evaluate Table 2 + Speedup Benchmark: Guided Pose Completion (40% joints occluded)
    print("\n>>> Evaluating Guided Pose Completion & Fast Inference Extensions...")
    eval_batch = test_poses[:32]
    visible_mask = torch.ones(args.num_joints, device=device)
    num_occluded = max(2, int(0.4 * args.num_joints))
    visible_mask[-num_occluded:] = 0.0  # Occlude the last 40% of joints (e.g., legs or fingers)

    configs = [
        ("1. PoseFM + Euclidean D-Flow (N=40)", posefm, "euclidean_dflow", 40, 40),
        ("2. Vanilla PoseD-Flow (Paper Alg. 2, N=40)", poserfm, "vanilla_dflow", 40, 40),
        ("3. Midpoint baseline (N=2)", poserfm, "fast_midpoint", 2, 40),
        ("4. Midpoint baseline (N=4)", poserfm, "fast_midpoint", 4, 40),
        ("5. Heuristic surrogate (N=40)", poserfm, "surrogate_pullback", 40, 40),
    ]

    print("\n" + "=" * 92)
    print(f"{'Method':<45} | {'Occ Err (deg)':<13} | {'Full Err (deg)':<14} | {'Batch ms/pose':<14}")
    print("-" * 92)

    shared_source = sample_uniform_so3(len(eval_batch), args.num_joints, device=device)
    for name, model, mode, ode_n, opt_iters in configs:
        comp_poses, ms_per_pose, _ = guided_pose_completion(
            model=model,
            x_obs=eval_batch,
            visible_mask=visible_mask,
            mode=mode,
            ode_steps=ode_n,
            opt_iters=opt_iters,
            lr=0.08,
            initial_source=shared_source,
        )
        acc = evaluate_completion_accuracy(comp_poses, eval_batch, visible_mask)
        print(
            f"{name:<45} | {acc['occluded_err_deg']:>11.2f}° | "
            f"{acc['full_body_err_deg']:>12.2f}° | {ms_per_pose:>11.2f} ms"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
