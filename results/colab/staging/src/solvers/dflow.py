"""
Riemannian D-Flow: Training-free guided generation by differentiating through Riemannian flows.

Implements Algorithm 2 from PoseD-Flow:
- Initializes source point x_0 with linear blend strategy (Supplement Eq. D.1).
- Solves ODE forward in time via Euler integration.
- Evaluates task objective L_data + lambda_traj * L_traj (+ auxiliary regularizers).
- Differentiates terminal loss back to source point x_0.
- Updates x_0 using Riemannian Adam on SO(3)^K.
"""

import math
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.geometry.so3 import (
    project_to_so3,
    sample_uniform_so3,
    trajectory_regularizer,
)
from src.geometry.radam import RiemannianAdam
from src.solvers.ode_solvers import ODESolver


class RiemannianDFlow:
    """
    Guidance framework that optimizes the source state x_0 of a pre-trained flow prior.
    """

    def __init__(
        self,
        model: nn.Module,
        solver: Optional[ODESolver] = None,
        is_riemannian: bool = True,
        lr: float = 0.1,
        num_iters: int = 200,
        blend_alpha: float = 0.25,
        lambda_traj: float = 1e-3,
    ):
        self.model = model
        self.solver = solver or ODESolver(method="euler", num_steps=100, retraction="last_step")
        self.is_riemannian = is_riemannian
        self.lr = lr
        self.num_iters = num_iters
        self.blend_alpha = blend_alpha
        self.lambda_traj = lambda_traj

    def initialize_source(
        self,
        x_obs: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_joints: int = 21,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize source point x_0 using the square-root weighted blend:
        x_0 = sqrt(alpha) * x_0^rand + sqrt(1 - alpha) * x_0^obs
        followed by projection onto SO(3).
        """
        x_rand = sample_uniform_so3((batch_size, num_joints), device=device, dtype=dtype)

        if x_obs is None or self.blend_alpha >= 1.0:
            return x_rand

        # Integrate backward from observation
        with torch.no_grad():
            x_obs_source = self.solver.integrate_backward(
                self.model, x_obs, is_riemannian=self.is_riemannian
            )

        w_rand = math.sqrt(self.blend_alpha)
        w_obs = math.sqrt(1.0 - self.blend_alpha)

        blended = w_rand * x_rand + w_obs * x_obs_source
        x0 = project_to_so3(blended)
        return x0

    def optimize(
        self,
        x0: torch.Tensor,
        data_loss_fn: Callable[[torch.Tensor], torch.Tensor],
        aux_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, list]]:
        """
        Execute source-point Riemannian optimization.
        Args:
            x0: Initial source state of shape (B, K, 3, 3)
            data_loss_fn: Callable returning scalar data error L_data(x_1)
            aux_loss_fn: Optional auxiliary loss callable(x_1, x0)
        Returns:
            best_x1: Optimized pose at t=1 of shape (B, K, 3, 3)
            best_x0: Optimized source state at t=0 of shape (B, K, 3, 3)
            history: Dictionary of loss values across iterations
        """
        optimizer = RiemannianAdam(lr=self.lr)
        curr_x0 = x0.clone().detach()

        history = {"loss": [], "data_loss": [], "traj_loss": []}
        best_loss = float("inf")
        best_x1 = None
        best_x0 = curr_x0.clone()

        for it in range(self.num_iters):
            curr_x0 = curr_x0.detach().requires_grad_(True)

            # 1. Forward ODE integration through frozen prior
            x1, traj = self.solver.integrate(
                self.model,
                curr_x0,
                is_riemannian=self.is_riemannian,
                record_trajectory=(self.lambda_traj > 0),
            )

            # 2. Evaluate data loss
            l_data = data_loss_fn(x1)
            total_loss = l_data

            # 3. Trajectory regularizer: sum_{i, k} (3 - tr(R_k^{(i)}))
            l_traj = torch.tensor(0.0, device=x1.device)
            if traj is not None and self.lambda_traj > 0:
                l_traj = torch.mean(trajectory_regularizer(traj))
                total_loss = total_loss + self.lambda_traj * l_traj

            # 4. Auxiliary loss if specified (e.g., temporal smoothness or SMPLify camera/shape)
            if aux_loss_fn is not None:
                total_loss = total_loss + aux_loss_fn(x1, curr_x0)

            # 5. Differentiate through entire ODE chain
            total_loss.backward()

            # Record history
            curr_loss_val = total_loss.item()
            history["loss"].append(curr_loss_val)
            history["data_loss"].append(l_data.item())
            history["traj_loss"].append(l_traj.item())

            if curr_loss_val < best_loss or best_x1 is None:
                best_loss = curr_loss_val
                best_x1 = x1.detach().clone()
                best_x0 = curr_x0.detach().clone()

            # 6. Update source point on SO(3)^K
            grad_x0 = curr_x0.grad
            curr_x0 = optimizer.step(curr_x0, grad_x0)

        if best_x1 is None:
            best_x1, _ = self.solver.integrate(self.model, curr_x0, is_riemannian=self.is_riemannian)

        return best_x1, best_x0, history
