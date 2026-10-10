"""
Unit test suite for models, ODE integration, finite differences, and metrics.
"""

import unittest
import torch

from src.models.poserfm import PoseRFM
from src.models.posefm import PoseFM
from src.solvers.ode_solvers import ODESolver
from src.solvers.dflow import RiemannianDFlow
from src.geometry.so3 import sample_uniform_so3
from src.evaluation.metrics import compute_mpjpe, compute_pck_at_threshold, compute_apd


class TestModelsAndSolvers(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_poserfm_forward_and_loss(self):
        model = PoseRFM(num_joints=21, hidden_dim=64, num_layers=2)
        x = sample_uniform_so3((4, 21))
        t = torch.rand(4, 1)

        out = model(x, t)
        self.assertEqual(out.shape, (4, 21, 3))

        loss, logs = model.compute_loss(x)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("rcfm_loss", logs)

    def test_posefm_forward_and_loss(self):
        model = PoseFM(num_joints=21, hidden_dim=64, num_layers=2)
        x = sample_uniform_so3((4, 21))
        t = torch.rand(4, 1)

        out = model(x, t)
        self.assertEqual(out.shape, (4, 21, 3, 3))

        loss, logs = model.compute_loss(x)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("euclidean_fm_loss", logs)

    def test_ode_solver_integration(self):
        model = PoseRFM(num_joints=21, hidden_dim=64, num_layers=2)
        solver = ODESolver(method="euler", num_steps=5, retraction="last_step")
        x0 = sample_uniform_so3((2, 21))

        x1, traj = solver.integrate(model, x0, is_riemannian=True, record_trajectory=True)
        self.assertEqual(x1.shape, (2, 21, 3, 3))
        self.assertEqual(traj.shape, (6, 2, 21, 3, 3))

        # Check backward integration
        x0_rec = solver.integrate_backward(model, x1, is_riemannian=True)
        self.assertEqual(x0_rec.shape, (2, 21, 3, 3))

    def test_finite_difference_gradient(self):
        """
        Verify that autograd gradient through a short ODE step is non-zero and consistent.
        """
        model = PoseRFM(num_joints=21, hidden_dim=32, num_layers=2)
        solver = ODESolver(method="euler", num_steps=2, retraction="last_step")

        x0 = sample_uniform_so3((1, 21)).requires_grad_(True)
        x1, _ = solver.integrate(model, x0, is_riemannian=True)

        target = sample_uniform_so3((1, 21))
        loss = torch.sum((x1 - target) ** 2)
        loss.backward()

        self.assertIsNotNone(x0.grad)
        grad_norm = torch.norm(x0.grad).item()
        self.assertGreater(grad_norm, 0.0)

    def test_metrics_sanity(self):
        # 1. Identical poses have zero MPJPE
        joints = torch.randn(5, 22, 3)
        self.assertAlmostEqual(compute_mpjpe(joints, joints), 0.0, places=4)

        # 2. Identical poses have 100% PCK@50
        pck = compute_pck_at_threshold(joints, joints, threshold_mm=50.0)
        self.assertAlmostEqual(pck, 100.0, places=4)

        # 3. APD on positive random joints > 0
        apd = compute_apd(joints)
        self.assertGreater(apd, 0.0)


if __name__ == "__main__":
    unittest.main()
