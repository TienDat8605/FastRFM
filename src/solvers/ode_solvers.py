"""
Riemannian and Euclidean ODE solvers for Flow Matching models.

Implements:
- Riemannian Euler solver (with configurable retraction: 'last_step', 'every_step', 'exp_map')
- Midpoint ODE solver (2-eval Runge-Kutta for high quality unconditional sampling)
- Trajectory recording for computing L_traj penalty
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from src.geometry.so3 import exp_map_at_point, project_to_so3, hat


class ODESolver:
    """
    Differentiable ODE integrator supporting Riemannian and Euclidean flows.
    """

    def __init__(
        self,
        method: str = "euler",
        retraction: str = "last_step",
        num_steps: int = 100,
    ):
        """
        Args:
            method: 'euler' or 'midpoint'
            retraction: 'last_step', 'every_step', or 'exp_map'
            num_steps: Number of integration steps between t=0 and t=1
        """
        self.method = method.lower()
        self.retraction = retraction.lower()
        self.num_steps = num_steps

    def step_euler(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        is_riemannian: bool = True,
    ) -> torch.Tensor:
        """Single Euler integration step."""
        vel = model(x, t)  # (B, K, 3) for PoseRFM or (B, K, 3, 3) for PoseFM

        if is_riemannian:
            if self.retraction == "exp_map":
                # Intrinsic geodesic step: x_{t+dt} = x_t * exp(dt * v)
                x_next = exp_map_at_point(x, dt * vel)
            else:
                # Extrinsic tangent step: x_dot = x * hat(v)
                omega_skew = hat(vel)  # (B, K, 3, 3)
                x_dot = torch.matmul(x, omega_skew)
                x_next = x + dt * x_dot
                if self.retraction == "every_step":
                    x_next = project_to_so3(x_next)
        else:
            # Pure Euclidean step
            x_next = x + dt * vel
            if self.retraction == "every_step":
                x_next = project_to_so3(x_next)

        return x_next

    def step_midpoint(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        is_riemannian: bool = True,
    ) -> torch.Tensor:
        """Single Midpoint integration step."""
        half_dt = 0.5 * dt
        half_t = t + half_dt

        # Stage 1: Half-step
        v1 = model(x, t)
        if is_riemannian:
            if self.retraction == "exp_map":
                x_mid = exp_map_at_point(x, half_dt * v1)
            else:
                omega1 = hat(v1)
                x_mid = x + half_dt * torch.matmul(x, omega1)
                if self.retraction == "every_step":
                    x_mid = project_to_so3(x_mid)
        else:
            x_mid = x + half_dt * v1
            if self.retraction == "every_step":
                x_mid = project_to_so3(x_mid)

        # Stage 2: Full step using slope at midpoint
        v2 = model(x_mid, half_t)
        if is_riemannian:
            if self.retraction == "exp_map":
                x_next = exp_map_at_point(x, dt * v2)
            else:
                omega2 = hat(v2)
                x_next = x + dt * torch.matmul(x, omega2)
                if self.retraction == "every_step":
                    x_next = project_to_so3(x_next)
        else:
            x_next = x + dt * v2
            if self.retraction == "every_step":
                x_next = project_to_so3(x_next)

        return x_next

    def integrate(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        is_riemannian: bool = True,
        record_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Integrate the ODE from t=0 to t=1.
        Args:
            model: Flow model (PoseRFM or PoseFM)
            x0: Initial condition at t=0 of shape (B, K, 3, 3)
            is_riemannian: Whether model predicts tangent velocities
            record_trajectory: If True, returns tensor of intermediate states for L_traj
        Returns:
            x1: Terminal state at t=1 of shape (B, K, 3, 3) in SO(3)
            trajectory: Intermediate states of shape (num_steps, B, K, 3, 3) or None
        """
        device = x0.device
        dtype = x0.dtype
        b = x0.shape[0]

        dt = 1.0 / self.num_steps
        x = x0

        trajectory_steps: List[torch.Tensor] = []
        if record_trajectory:
            trajectory_steps.append(x)

        for step in range(self.num_steps):
            t_val = step * dt
            t_tensor = torch.full((b, 1), t_val, device=device, dtype=dtype)

            if self.method == "midpoint":
                x = self.step_midpoint(model, x, t_tensor, dt, is_riemannian)
            else:
                x = self.step_euler(model, x, t_tensor, dt, is_riemannian)

            if record_trajectory:
                trajectory_steps.append(x)

        # Final projection onto SO(3)
        x1 = project_to_so3(x)

        traj_tensor = None
        if record_trajectory:
            traj_tensor = torch.stack(trajectory_steps, dim=0)

        return x1, traj_tensor

    def integrate_backward(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        is_riemannian: bool = True,
    ) -> torch.Tensor:
        """
        Integrate ODE backward in time from t=1 to t=0 to find source point x_0^obs.
        """
        device = x1.device
        dtype = x1.dtype
        b = x1.shape[0]

        dt = -1.0 / self.num_steps
        x = x1

        for step in range(self.num_steps):
            t_val = 1.0 + step * dt  # 1.0 down to 0.0
            t_tensor = torch.full((b, 1), t_val, device=device, dtype=dtype)

            if is_riemannian:
                vel = model(x, t_tensor)
                omega_skew = hat(vel)
                x_dot = torch.matmul(x, omega_skew)
                x = x + dt * x_dot
            else:
                vel = model(x, t_tensor)
                x = x + dt * vel

        x0 = project_to_so3(x)
        return x0
