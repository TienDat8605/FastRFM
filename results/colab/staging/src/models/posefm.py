"""
PoseFM: Euclidean Flow Matching baseline model of human pose.

Input: Flattened rotation matrices (K * 9 = 189) + scalar time t (Total: 190).
Output: Flattened matrix velocity in R^(K * 9) (Total: 189).
Architecture: 4 hidden layers MLP (512 hidden units), Swish (SiLU) activation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.geometry.so3 import project_to_so3, sample_uniform_so3


class PoseFM(nn.Module):
    """
    Architecture-matched Euclidean Flow Matching model for 21 body joints.
    """

    def __init__(
        self,
        num_joints: int = 21,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = num_joints * 9 + 1
        self.output_dim = num_joints * 9  # 189 Euclidean velocity

        layers = []
        in_features = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict Euclidean velocity field.
        Args:
            x: Matrix tensor of shape (B, K, 3, 3) or (B, K*9)
            t: Scalar time tensor of shape (B,) or (B, 1) in [0, 1]
        Returns:
            Velocity of shape (B, K, 3, 3)
        """
        b = x.shape[0]
        if x.dim() == 4:
            x_flat = x.reshape(b, -1)
        else:
            x_flat = x

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        feat = torch.cat([x_flat, t], dim=-1)
        out = self.net(feat)
        return out.view(b, self.num_joints, 3, 3)

    def compute_loss(
        self,
        x1: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Euclidean Conditional Flow Matching loss:
        x_t = (1 - t) * x_0 + t * x_1
        u_t(x_t | x_1) = x_1 - x_0
        Loss = ||v_w(x_t, t) - (x_1 - x_0)||^2
        """
        b, k, _, _ = x1.shape
        device = x1.device
        dtype = x1.dtype

        if x0 is None:
            # Sample source rotation matrices (or standard normal matrices)
            x0 = sample_uniform_so3((b, k), device=device, dtype=dtype)

        if t is None:
            t = torch.rand(b, 1, 1, 1, device=device, dtype=dtype)
        elif t.dim() == 1 or t.dim() == 2:
            t = t.view(b, 1, 1, 1)

        # Linear Euclidean interpolation
        xt = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0

        pred_velocity = self.forward(xt, t.view(b, 1))

        loss = torch.mean((pred_velocity - target_velocity) ** 2)

        return loss, {
            "euclidean_fm_loss": loss.item(),
        }
