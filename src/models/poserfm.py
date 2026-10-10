"""
PoseRFM: Riemannian Flow Matching model of human pose on SO(3)^K.

Input: Flattened rotation matrices (K * 9) concatenated with scalar time t in [0, 1] (Total dim: 190).
Output: Tangent vector in T_x SO(3)^K (Dimension: K * 3 = 63).
Architecture: 4 hidden layers MLP (512 hidden units), Swish (SiLU) activation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.geometry.so3 import (
    exp_map_at_point,
    log_map_at_point,
    project_to_so3,
    sample_uniform_so3,
)


class PoseRFM(nn.Module):
    """
    Time-conditional Riemannian Flow Matching network for 21 articulated body joints.
    """

    def __init__(
        self,
        num_joints: int = 21,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = num_joints * 9 + 1  # 21 * 9 + 1 = 190
        self.output_dim = num_joints * 3     # 21 * 3 = 63

        layers = []
        in_features = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())  # Swish
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict tangent velocity field u_t(x).
        Args:
            x: Rotation state of shape (B, K, 3, 3) or (B, K*9)
            t: Scalar time tensor of shape (B,) or (B, 1) in [0, 1]
        Returns:
            Tangent vector coordinates of shape (B, K, 3)
        """
        batch_size = x.shape[0]
        if x.dim() == 4:
            x_flat = x.reshape(batch_size, -1)  # (B, K*9)
        else:
            x_flat = x

        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        feat = torch.cat([x_flat, t], dim=-1)  # (B, 190)
        out = self.net(feat)                   # (B, 63)
        return out.view(batch_size, self.num_joints, 3)

    def compute_loss(
        self,
        x1: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Riemannian Conditional Flow Matching (RCFM) loss.
        x_t = Exp_{x_0}(t * Log_{x_0}(x_1))
        u_t(x_t | x_1) = -Log_{x_t}(x_1) / (1 - t)
        Loss = ||v_w(x_t, t) - u_t(x_t | x_1)||^2
        Args:
            x1: Ground truth target poses of shape (B, K, 3, 3) in SO(3)
            x0: Optional source points from Haar measure. If None, sampled uniformly.
            t: Optional random time in [0, 1]. If None, sampled U(0, 1).
        """
        b, k, _, _ = x1.shape
        device = x1.device
        dtype = x1.dtype

        if x0 is None:
            x0 = sample_uniform_so3((b, k), device=device, dtype=dtype)

        if t is None:
            t = torch.rand(b, 1, device=device, dtype=dtype)

        t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        # Geodesic interpolation along SO(3)^K
        # xi_0 = Log_{x_0}(x_1) in R^3
        xi_0 = log_map_at_point(x0, x1)  # (B, K, 3)
        xt = exp_map_at_point(x0, t.unsqueeze(-1) * xi_0)  # (B, K, 3, 3)
        xt = project_to_so3(xt)

        # Target conditional vector field: -Log_{x_t}(x_1) / (1 - t)
        # Note: Along geodesic, Log_{x_t}(x_1) = (1 - t) * (parallel_transported xi_0)
        # Hence target u_t in body coordinates is xi_0!
        # Numerically: compute -Log_{x_t}(x_1) / max(1 - t, eps)
        target_ut = -log_map_at_point(xt, x1) / torch.clamp(1.0 - t.unsqueeze(-1), min=eps)

        pred_ut = self.forward(xt, t)  # (B, K, 3)

        loss = torch.mean((pred_ut - target_ut) ** 2)

        return loss, {
            "rcfm_loss": loss.item(),
        }
