"""
SO(3) and product manifold SO(3)^K geometric operations.

Implements:
- Hat and Vee operators
- Exp and Log maps on SO(3) with robust Taylor expansions for small angles
- Geodesic distance (L1 product metric on SO(3)^K)
- Orthogonal projection of matrices onto SO(3) via SVD
- Tangent space orthogonal projection (egrad2rgrad)
- Haar-uniform random rotation sampling
- Trajectory regularizer: sum_{k} (3 - tr(R_k))
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Map vector in R^3 to skew-symmetric matrix in so(3).
    Args:
        v: Tensor of shape (..., 3)
    Returns:
        Tensor of shape (..., 3, 3)
    """
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(v1)
    row0 = torch.stack([zeros, -v3, v2], dim=-1)
    row1 = torch.stack([v3, zeros, -v1], dim=-1)
    row2 = torch.stack([-v2, v1, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee(omega: torch.Tensor) -> torch.Tensor:
    """
    Map skew-symmetric matrix in so(3) to vector in R^3.
    Args:
        omega: Tensor of shape (..., 3, 3)
    Returns:
        Tensor of shape (..., 3)
    """
    v1 = omega[..., 2, 1]
    v2 = omega[..., 0, 2]
    v3 = omega[..., 1, 0]
    return torch.stack([v1, v2, v3], dim=-1)


def so3_exp_map(v: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Rodrigues formula: Exp: so(3) -> SO(3)
    Args:
        v: Tangent vector (axis-angle) of shape (..., 3)
        eps: Threshold for small angles
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    theta_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # (..., 1)
    theta = torch.sqrt(torch.clamp(theta_sq, min=eps))   # (..., 1)

    theta_sq_expanded = theta_sq.unsqueeze(-1)
    theta_expanded = theta.unsqueeze(-1)

    k = hat(v)  # (..., 3, 3)
    k2 = torch.matmul(k, k)

    # Taylor expansion for small angles:
    # sin(theta)/theta = 1 - theta^2 / 6 + theta^4 / 120
    # (1 - cos(theta))/theta^2 = 1/2 - theta^2 / 24 + theta^4 / 720
    small_angle = (theta_sq < 1e-4).unsqueeze(-1)

    alpha_small = 1.0 - theta_sq_expanded / 6.0
    beta_small = 0.5 - theta_sq_expanded / 24.0

    alpha_large = torch.sin(theta_expanded) / theta_expanded
    beta_large = (1.0 - torch.cos(theta_expanded)) / theta_sq_expanded

    alpha = torch.where(small_angle, alpha_small, alpha_large)
    beta = torch.where(small_angle, beta_small, beta_large)

    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(k)
    return eye + alpha * k + beta * k2


def so3_log_map(r: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Inverse Rodrigues formula: Log: SO(3) -> so(3) (returns axis-angle vector in R^3).
    Args:
        r: Rotation matrix of shape (..., 3, 3)
    Returns:
        Axis-angle vector of shape (..., 3)
    """
    # tr(R) = 1 + 2*cos(theta)
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...)

    sin_theta = torch.sin(theta)

    # Skew-symmetric part: (R - R^T) / 2
    skew = (r - r.transpose(-1, -2)) / 2.0
    v = vee(skew)  # (..., 3)

    small_angle = (theta < 1e-4).unsqueeze(-1)

    # For small theta: theta / sin(theta) ~ 1 + theta^2 / 6
    scale_small = 1.0 + (theta ** 2).unsqueeze(-1) / 6.0
    scale_large = (theta / torch.clamp(sin_theta, min=eps)).unsqueeze(-1)

    scale = torch.where(small_angle, scale_small, scale_large)
    return v * scale


def exp_map_at_point(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exp_R(v) = R * exp(v) where v in R^3 is axis-angle representation of body tangent vector.
    Args:
        r: Base rotation matrix of shape (..., 3, 3)
        v: Tangent coordinates of shape (..., 3)
    Returns:
        Updated rotation matrix of shape (..., 3, 3)
    """
    delta_r = so3_exp_map(v)
    return torch.matmul(r, delta_r)


def log_map_at_point(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    Log_{R1}(R2): Returns the body tangent vector v in R^3 such that Exp_{R1}(v) = R2.
    Specifically: v = so3_log_map(R1^T * R2).
    """
    r_rel = torch.matmul(r1.transpose(-1, -2), r2)
    return so3_log_map(r_rel)


class _ProjectToSO3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m: torch.Tensor) -> torch.Tensor:
        # Use detached tensor for SVD to prevent NaN in autograd from degenerate singular values
        m_det = m.detach()
        u, _, vt = torch.linalg.svd(m_det)
        det = torch.linalg.det(torch.matmul(u, vt))
        d = torch.ones_like(m[..., :3, 0])
        d[..., 2] = det
        d_mat = torch.diag_embed(d)
        r = torch.matmul(u, torch.matmul(d_mat, vt))
        ctx.save_for_backward(r)
        return r

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (r,) = ctx.saved_tensors
        # The differential of the projection map onto SO(3) is the tangent projection:
        # dPi_R(H) = R * skew(R^T * H)
        rt_grad = torch.matmul(r.transpose(-1, -2), grad_output)
        skew = (rt_grad - rt_grad.transpose(-1, -2)) / 2.0
        grad_m = torch.matmul(r, skew)
        return grad_m


def project_to_so3(m: torch.Tensor) -> torch.Tensor:
    """
    Orthogonally project an arbitrary 3x3 matrix onto SO(3) via SVD (Polar decomposition).
    Uses custom autograd backward to avoid SVD gradient explosion on identical singular values.
    Args:
        m: Tensor of shape (..., 3, 3)
    Returns:
        Valid rotation matrix of shape (..., 3, 3) in SO(3)
    """
    return _ProjectToSO3.apply(m)


def project_to_tangent_space(r: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Project ambient gradient grad onto tangent space T_R SO(3).
    Returns skew-symmetric tangent matrix: R * skew(R^T * grad).
    Or tangent coordinate vector v in R^3.
    """
    rt_grad = torch.matmul(r.transpose(-1, -2), grad)
    skew = (rt_grad - rt_grad.transpose(-1, -2)) / 2.0
    return vee(skew)


def so3_geodesic_distance(r1: torch.Tensor, r2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute geodesic distance (rotation angle) on SO(3) in radians.
    d(R1, R2) = arccos( clamp((tr(R1^T * R2) - 1) / 2) )
    Args:
        r1: Tensor of shape (..., 3, 3)
        r2: Tensor of shape (..., 3, 3)
    Returns:
        Angles in radians of shape (...)
    """
    r_rel = torch.matmul(r1.transpose(-1, -2), r2)
    trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)


def so3_product_distance(x1: torch.Tensor, x2: torch.Tensor, p: int = 1) -> torch.Tensor:
    """
    Lp product geodesic distance on SO(3)^K.
    For p=1 (used in PoseD-Flow): sum_{k=1}^K d_{SO(3)}(R1_k, R2_k).
    Args:
        x1: Tensor of shape (..., K, 3, 3)
        x2: Tensor of shape (..., K, 3, 3)
        p: Norm degree (default 1)
    Returns:
        Distance tensor of shape (...)
    """
    dists = so3_geodesic_distance(x1, x2)  # (..., K)
    if p == 1:
        return torch.sum(dists, dim=-1)
    else:
        return torch.norm(dists, p=p, dim=-1)


def sample_uniform_so3(size: Tuple[int, ...], device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Sample rotations uniformly distributed under Haar measure on SO(3).
    Uses normalized Gaussian quaternion sampling.
    """
    q = torch.randn(*size, 4, device=device, dtype=dtype)
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1.0 - 2.0 * (y ** 2 + z ** 2)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x ** 2 + z ** 2)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x ** 2 + y ** 2)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def trajectory_regularizer(rotations: torch.Tensor) -> torch.Tensor:
    """
    Compute trajectory penalty L_traj = sum_{i, k} (3 - tr(R_k^{(i)})).
    Args:
        rotations: Intermediate rotations of shape (num_steps, batch_size, K, 3, 3) or (batch_size, num_steps, K, 3, 3)
    Returns:
        Scalar penalty per batch item or total mean scalar.
    """
    # trace across diagonal
    trace = rotations[..., 0, 0] + rotations[..., 1, 1] + rotations[..., 2, 2]  # (..., K)
    penalty = 3.0 - trace  # >= 0
    return torch.sum(penalty, dim=(-2, -1))
