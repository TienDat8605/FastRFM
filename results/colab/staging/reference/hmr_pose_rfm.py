"""
PoseRFM and Riemannian D-Flow Engine.
Implementation of Riemannian Flow Matching on SO(3)^21 and Riemannian D-Flow guidance
following Nadar et al., "PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose", CVPR 2026.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. SO(3) Lie Group and Riemannian Manifold Operations
# ==============================================================================

def so3_hat(w: torch.Tensor) -> torch.Tensor:
    """
    Hat map: converts tangent vectors in R^3 to skew-symmetric matrices in so(3).
    Args:
        w: Tensor of shape (..., 3)
    Returns:
        W: Tensor of shape (..., 3, 3)
    """
    zeros = torch.zeros_like(w[..., 0])
    w0, w1, w2 = w[..., 0], w[..., 1], w[..., 2]
    
    W = torch.stack([
        zeros, -w2,  w1,
        w2,   zeros, -w0,
        -w1,  w0,   zeros
    ], dim=-1).view(w.shape[:-1] + (3, 3))
    return W


def so3_vee(W: torch.Tensor) -> torch.Tensor:
    """
    Vee map: extracts vector from skew-symmetric matrix in so(3).
    Args:
        W: Tensor of shape (..., 3, 3)
    Returns:
        w: Tensor of shape (..., 3)
    """
    w0 = W[..., 2, 1]
    w1 = W[..., 0, 2]
    w2 = W[..., 1, 0]
    return torch.stack([w0, w1, w2], dim=-1)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """
    Matrix exponential map from so(3) ~ R^3 to SO(3) via Rodrigues formula.
    Numerically stable for small rotation angles.
    Args:
        w: Tangent vector (..., 3)
    Returns:
        R: Rotation matrix (..., 3, 3)
    """
    theta_sq = torch.sum(w ** 2, dim=-1, keepdim=True)  # (..., 1)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-14))
    
    # Taylor expansions for small theta:
    # sin(theta)/theta = 1 - theta^2 / 6 + theta^4 / 120
    # (1 - cos(theta))/theta^2 = 1/2 - theta^2 / 24 + theta^4 / 720
    small_angle = theta_sq < 1e-6
    
    sinc = torch.where(
        small_angle,
        1.0 - theta_sq / 6.0 + (theta_sq ** 2) / 120.0,
        torch.sin(theta) / theta
    )
    inv_cos = torch.where(
        small_angle,
        0.5 - theta_sq / 24.0 + (theta_sq ** 2) / 720.0,
        (1.0 - torch.cos(theta)) / (theta_sq + 1e-14)
    )
    
    W = so3_hat(w)
    W2 = torch.matmul(W, W)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
    
    R = eye + sinc.unsqueeze(-1) * W + inv_cos.unsqueeze(-1) * W2
    return R


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Robust matrix to unit quaternion [w, x, y, z] conversion via Shepperd's algorithm.
    Numerically stable everywhere including antipodal / pi rotations.
    """
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    
    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22
    
    q0 = torch.stack([
        t0,
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1]
    ], dim=-1) / (2.0 * torch.sqrt(torch.clamp(t0, min=1e-8)).unsqueeze(-1))
    
    q1 = torch.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        t1,
        matrix[..., 0, 1] + matrix[..., 1, 0],
        matrix[..., 0, 2] + matrix[..., 2, 0]
    ], dim=-1) / (2.0 * torch.sqrt(torch.clamp(t1, min=1e-8)).unsqueeze(-1))
    
    q2 = torch.stack([
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 0, 1] + matrix[..., 1, 0],
        t2,
        matrix[..., 1, 2] + matrix[..., 2, 1]
    ], dim=-1) / (2.0 * torch.sqrt(torch.clamp(t2, min=1e-8)).unsqueeze(-1))
    
    q3 = torch.stack([
        matrix[..., 1, 0] - matrix[..., 0, 1],
        matrix[..., 0, 2] + matrix[..., 2, 0],
        matrix[..., 1, 2] + matrix[..., 2, 1],
        t3
    ], dim=-1) / (2.0 * torch.sqrt(torch.clamp(t3, min=1e-8)).unsqueeze(-1))
    
    diag = torch.stack([t0, t1, t2, t3], dim=-1)
    max_idx = torch.argmax(diag, dim=-1, keepdim=True).unsqueeze(-1)
    q_candidates = torch.stack([q0, q1, q2, q3], dim=-2)
    q = torch.gather(q_candidates, -2, max_idx.expand(*max_idx.shape[:-2], 1, 4)).squeeze(-2)
    return q / torch.norm(q, dim=-1, keepdim=True)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    Matrix logarithm from SO(3) to so(3) ~ R^3.
    Numerically exact across the entire manifold including theta -> 0 and theta -> pi.
    Args:
        R: Rotation matrix (..., 3, 3)
    Returns:
        w: Tangent vector (..., 3)
    """
    q = matrix_to_quaternion(R)
    # Ensure positive scalar component w >= 0
    q = torch.where(q[..., :1] < 0, -q, q)
    qw = q[..., 0:1]
    qv = q[..., 1:]
    norm_v = torch.norm(qv, dim=-1, keepdim=True)
    
    # 2 * atan2(norm_v, qw) has full relative precision across all angles
    theta = 2.0 * torch.atan2(norm_v, qw)
    
    # Taylor expansion: theta / norm_v = 2 + (norm_v^2) / 3 + 7 * (norm_v^4) / 45
    small_angle = norm_v < 1e-6
    scale = torch.where(
        small_angle,
        2.0 + (norm_v ** 2) / 3.0 + 7.0 * (norm_v ** 4) / 45.0,
        theta / (norm_v + 1e-12)
    )
    return (scale * qv).view(R.shape[:-2] + (3,))


def so3_geodesic_dist(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on SO(3): d_g(R1, R2) = ||Log(R1^T R2)||_2.
    Args:
        R1, R2: (..., 3, 3)
    Returns:
        dist: (...)
    """
    R_rel = torch.matmul(R1.transpose(-1, -2), R2)
    return torch.norm(so3_log(R_rel), dim=-1)


def so3_geodesic_interp(R0: torch.Tensor, R1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Geodesic interpolation between R0 and R1 on SO(3) at time t in [0, 1].
    gamma(t) = R0 * Exp(t * Log(R0^T * R1))
    Also returns the target tangent velocity along the geodesic: v0 = Log(R0^T * R1).
    Args:
        R0, R1: (..., 3, 3)
        t: (..., 1) or scalar float/tensor
    Returns:
        R_t: (..., 3, 3)
        v0: (..., 3)
    """
    R_rel = torch.matmul(R0.transpose(-1, -2), R1)
    v0 = so3_log(R_rel)  # (..., 3)
    
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=R0.device, dtype=R0.dtype)
    while t.dim() < v0.dim():
        t = t.unsqueeze(-1)
        
    delta_w = t * v0
    R_t = torch.matmul(R0, so3_exp(delta_w))
    return R_t, v0


def sample_uniform_so3(shape: Tuple[int, ...], device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Sample uniform random rotation matrices on SO(3) according to the Haar measure.
    Generated via uniformly distributed random quaternions on S^3.
    Args:
        shape: Prefix batch shape, e.g. (B, K)
    Returns:
        R: Tensor of shape (*shape, 3, 3)
    """
    # Sample 4D Gaussian vectors and normalize to unit sphere S^3
    u = torch.randn(*shape, 4, device=device, dtype=dtype)
    q = F.normalize(u, p=2, dim=-1)  # (..., 4) [w, x, y, z]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Quaternion to rotation matrix
    r00 = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx ** 2 + qz ** 2)
    r12 = 2.0 * (qy * qz - qx * qw)
    
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    
    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).view(*shape, 3, 3)
    return R


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection of arbitrary 3x3 matrices onto SO(3) via SVD.
    R = U * diag(1, 1, det(U * V^T)) * V^T
    """
    U, _, Vh = torch.linalg.svd(M)
    V = Vh.transpose(-1, -2)
    d = torch.det(torch.matmul(U, Vh))
    diag = torch.ones_like(M[..., 0])
    diag[..., 2] = d
    S = torch.diag_embed(diag)
    return torch.matmul(torch.matmul(U, S), Vh)


# ==============================================================================
# 2. Sinusoidal Time Embedding and PoseRFM Architecture
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for continuous time t in [0, 1].
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) continuous time
        Returns:
            emb: (B, embed_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * self.freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class PoseRFM(nn.Module):
    """
    Riemannian Conditional Flow Matching (RCFM) Network on SO(3)^K.
    Nadar et al. (CVPR 2026), Section 4 & Appendix B.
    
    Predicts the tangent vector field u_t in T_{x_t} M ~ R^{K x 3}
    given pose x_t in SO(3)^K and time t in [0, 1].
    """
    def __init__(
        self,
        num_joints: int = 21,
        hidden_dim: int = 512,
        time_embed_dim: int = 64,
        num_layers: int = 4,
        use_6d_rot: bool = False
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_6d_rot = use_6d_rot
        self.rot_dim = 6 if use_6d_rot else 9
        self.input_dim = num_joints * self.rot_dim
        self.output_dim = num_joints * 3
        
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # 4-layer MLP with Swish (SiLU) and residual connections
        in_features = self.input_dim + time_embed_dim
        layers = []
        
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            )
            for _ in range(num_layers - 2)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, self.output_dim)

    def _rot_to_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, K, 3, 3) rotation matrices to input features.
        """
        B, K = x.shape[:2]
        if self.use_6d_rot:
            # First two columns of rotation matrix
            col01 = x[..., :, :2].contiguous()
            return col01.view(B, K * 6)
        else:
            return x.contiguous().view(B, K * 9)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict tangent velocity vector field v_t(x_t, t).
        Args:
            x_t: (B, K, 3, 3) rotation matrices on SO(3)^K
            t: (B,) or (B, 1) time in [0, 1]
        Returns:
            v: (B, K, 3) tangent velocity in Lie algebra / local frame
        """
        B = x_t.shape[0]
        x_feat = self._rot_to_features(x_t)  # (B, K * rot_dim)
        t_feat = self.time_embed(t)          # (B, time_embed_dim)
        
        h = torch.cat([x_feat, t_feat], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks:
            h = h + block(h)  # Residual connection
        out = self.out_proj(h)  # (B, K * 3)
        return out.view(B, self.num_joints, 3)


# ==============================================================================
# 3. Riemannian ODE Solvers
# ==============================================================================

class RiemannianODESolver:
    """
    Deterministic Riemannian ODE solver on SO(3)^K.
    Integrates dx_t / dt = x_t * [v_t]_x from t=0 to t=1 along geodesics.
    Fully differentiable with autograd.
    """
    @staticmethod
    def step_euler(
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single Riemannian Euler step on SO(3)^K:
        x_{t + dt, k} = x_{t, k} * Exp(dt * v_{t, k})
        """
        v = model(x, t)  # (B, K, 3)
        delta = dt * v
        R_delta = so3_exp(delta)
        return torch.matmul(x, R_delta)

    @staticmethod
    def step_midpoint(
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single Riemannian Midpoint (2nd order) step on SO(3)^K:
        x_mid = x * Exp(0.5 * dt * v(x, t))
        v_mid = v(x_mid, t + 0.5 * dt)
        x_{t + dt} = x * Exp(dt * v_mid)
        """
        v1 = model(x, t)
        x_mid = torch.matmul(x, so3_exp(0.5 * dt * v1))
        
        t_mid = t + 0.5 * dt
        v_mid = model(x_mid, t_mid)
        return torch.matmul(x, so3_exp(dt * v_mid))

    @classmethod
    def solve(
        cls,
        model: nn.Module,
        x0: torch.Tensor,
        num_steps: int = 10,
        method: str = "midpoint",
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Integrates the flow from t=0 to t=1.
        Args:
            model: PoseRFM neural network
            x0: Initial pose on SO(3)^K (B, K, 3, 3)
            num_steps: Number of integration steps (default 10)
            method: 'euler' or 'midpoint'
            return_trajectory: whether to return full path [x_0, ..., x_1]
        Returns:
            x1: (B, K, 3, 3) generated pose at t=1
        """
        step_fn = cls.step_midpoint if method == "midpoint" else cls.step_euler
        dt = 1.0 / num_steps
        x = x0
        trajectory = [x] if return_trajectory else None
        
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype
        
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            x = step_fn(model, x, t, dt)
            if return_trajectory:
                trajectory.append(x)
                
        if return_trajectory:
            return x, trajectory
        return x


# ==============================================================================
# 4. Riemannian D-Flow Optimizer (Test-Time Inversion & Guidance)
# ==============================================================================

class RiemannianDFlowOptimizer:
    """
    Test-time guidance on SO(3)^K via Riemannian D-Flow.
    Solves inverse problems:
        min_{x_0 in SO(3)^K} L( Phi(x_0), y_obs )
    where Phi(x_0) is the deterministic ODE solution at t=1.
    
    Uses Riemannian Adam (RAdam) optimization on SO(3)^K by reparameterizing
    tangent updates: x_0 <- x_0 * Exp(-lr * step).
    """
    def __init__(
        self,
        model: PoseRFM,
        num_ode_steps: int = 10,
        ode_method: str = "midpoint",
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.model = model
        self.num_ode_steps = num_ode_steps
        self.ode_method = ode_method
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        init_x0: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_joints: int = 21,
        num_iterations: int = 50,
        device: torch.device = torch.device("cpu"),
        early_stop_tol: float = 1e-5,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Runs Riemannian D-Flow test-time guidance.
        Args:
            loss_fn: Function mapping posed x_1 (B, K, 3, 3) to scalar loss
            init_x0: Optional initial source point on SO(3)^K. If None, samples random SO(3).
            num_iterations: Number of guidance iterations
        Returns:
            optimal_x1: Generated pose matching observation (B, K, 3, 3)
            optimal_x0: Optimized source state (B, K, 3, 3)
            loss_history: List of loss values per iteration
        """
        if init_x0 is None:
            x0 = sample_uniform_so3((batch_size, num_joints), device=device)
        else:
            x0 = init_x0.clone().detach().to(device)
            
        B, K = x0.shape[:2]
        
        # Initialize Adam moment buffers in tangent space R^{B x K x 3}
        m = torch.zeros(B, K, 3, device=device, dtype=x0.dtype)
        v = torch.zeros(B, K, 3, device=device, dtype=x0.dtype)
        loss_history = []
        
        for iteration in range(1, num_iterations + 1):
            # Parameterize perturbation delta in tangent space around current x0
            delta = torch.zeros(B, K, 3, device=device, dtype=x0.dtype, requires_grad=True)
            
            # Current source point perturbed: x0(delta) = x0 * Exp(delta)
            x0_pert = torch.matmul(x0, so3_exp(delta))
            
            # Forward solve ODE to endpoint x1
            x1 = RiemannianODESolver.solve(
                self.model,
                x0_pert,
                num_steps=self.num_ode_steps,
                method=self.ode_method
            )
            
            # Compute guidance loss
            loss = loss_fn(x1)
            loss_history.append(loss.item())
            
            if verbose and (iteration % 10 == 0 or iteration == 1):
                print(f"[D-Flow] Iteration {iteration:3d}/{num_iterations} | Loss: {loss.item():.6f}")
                
            # Riemannian gradient is exactly d(Loss) / d(delta) at delta=0
            loss.backward()
            grad = delta.grad.detach()  # (B, K, 3)
            
            # Check for early stopping
            if grad.norm().item() < early_stop_tol:
                if verbose:
                    print(f"[D-Flow] Early stopping at iteration {iteration}")
                break
                
            # Riemannian Adam update in tangent space
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1.0 - (self.beta1 ** iteration))
            v_hat = v / (1.0 - (self.beta2 ** iteration))
            
            step = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)  # (B, K, 3)
            
            # Lie group update: x0 <- x0 * Exp(-step)
            with torch.no_grad():
                x0 = torch.matmul(x0, so3_exp(-step))
                # Optional numerical projection back to exact SO(3)
                x0 = project_to_so3(x0)
                
        # Final forward evaluation from optimized x0
        with torch.no_grad():
            final_x1 = RiemannianODESolver.solve(
                self.model,
                x0,
                num_steps=self.num_ode_steps,
                method=self.ode_method
            )
            
        return final_x1, x0, loss_history
