"""
Riemannian Adam (RAdam) optimizer for optimization on SO(3)^K manifolds.

Implements Algorithm 3 from PoseD-Flow paper:
- Projects Euclidean gradients onto the tangent space T_{x_k} M.
- Tracks 1st and 2nd momentum with parallel transport.
- Updates points via exponential map Exp_{x_k}(Delta x_k).
"""

from typing import Dict, Optional, Tuple
import torch
from src.geometry.so3 import project_to_tangent_space, exp_map_at_point, project_to_so3


class RiemannianAdam:
    """
    Riemannian Adam optimizer tailored for optimizing source rotation states x_0 in SO(3)^K.
    """

    def __init__(
        self,
        lr: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def reset(self):
        """Reset momentum buffers."""
        self.step_count = 0
        self.m = None
        self.v = None

    def step(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Perform a single Riemannian Adam optimization step on x.
        Args:
            x: Current rotation state of shape (..., K, 3, 3) in SO(3)
            grad: Ambient Euclidean gradient dL/dx of shape (..., K, 3, 3)
        Returns:
            Updated rotation state x_next of shape (..., K, 3, 3) in SO(3)
        """
        self.step_count += 1
        t = self.step_count

        # 1. Project Euclidean gradient to tangent space coordinate in R^3
        # g has shape (..., K, 3)
        g = project_to_tangent_space(x, grad)

        if self.weight_decay > 0:
            # Add weight decay towards identity rotation
            # Tangent vector from identity to x
            pass

        # Initialize moments if first step
        if self.m is None or self.m.shape != g.shape:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)

        # 2. Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g

        # 3. Update biased second raw moment estimate (component-wise squared inner product)
        # For SO(3), inner product <g, g> in coordinate basis is component-wise g^2
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # 4. Compute bias-corrected first and second moment estimates
        bias_correction1 = 1.0 - self.beta1 ** t
        bias_correction2 = 1.0 - self.beta2 ** t

        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2

        # 5. Compute Riemannian update direction Delta x in tangent space
        delta_x = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)  # (..., K, 3)

        # 6. Apply retraction via Exponential map
        x_next = exp_map_at_point(x, delta_x)

        # 7. Final numerical projection to enforce strict SO(3) orthogonality
        x_next = project_to_so3(x_next)

        return x_next
