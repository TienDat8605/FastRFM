"""
Numerical validation suite for SO(3) geometry, Exp/Log maps, and gradients.
"""

import math
import unittest
import torch

from src.geometry.so3 import (
    hat,
    vee,
    so3_exp_map,
    so3_log_map,
    exp_map_at_point,
    log_map_at_point,
    project_to_so3,
    project_to_tangent_space,
    so3_geodesic_distance,
    sample_uniform_so3,
    trajectory_regularizer,
)
from src.geometry.radam import RiemannianAdam
from src.geometry.smpl_utils import DifferentiableKinematics


class TestSO3Geometry(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_hat_vee_roundtrip(self):
        v = torch.randn(10, 3)
        omega = hat(v)
        v_recovered = vee(omega)
        self.assertTrue(torch.allclose(v, v_recovered, atol=1e-6))

    def test_exp_log_roundtrip_general(self):
        # Sample random rotation vectors with angle in [0.01, pi - 0.05]
        v = torch.randn(20, 3)
        v = v / torch.norm(v, dim=-1, keepdim=True) * torch.rand(20, 1) * 3.0
        r = so3_exp_map(v)

        # Check orthogonality and determinant
        eye = torch.eye(3).expand_as(r)
        self.assertTrue(torch.allclose(torch.matmul(r, r.transpose(-1, -2)), eye, atol=1e-5))
        self.assertTrue(torch.allclose(torch.linalg.det(r), torch.ones(20), atol=1e-5))

        # Check Log map recovery
        v_rec = so3_log_map(r)
        self.assertTrue(torch.allclose(v, v_rec, atol=1e-4))

    def test_exp_log_small_angles(self):
        # Small angles < 1e-4
        v = torch.randn(15, 3) * 1e-5
        r = so3_exp_map(v)
        v_rec = so3_log_map(r)
        self.assertTrue(torch.allclose(v, v_rec, atol=1e-7))

    def test_point_exp_log_roundtrip(self):
        r1 = sample_uniform_so3((10,))
        v = torch.randn(10, 3) * 0.5
        r2 = exp_map_at_point(r1, v)
        v_rec = log_map_at_point(r1, r2)
        self.assertTrue(torch.allclose(v, v_rec, atol=1e-4))

    def test_project_to_so3(self):
        # Corrupted matrices
        m = torch.randn(10, 3, 3)
        r = project_to_so3(m)
        eye = torch.eye(3).expand_as(r)
        self.assertTrue(torch.allclose(torch.matmul(r, r.transpose(-1, -2)), eye, atol=1e-5))
        self.assertTrue(torch.allclose(torch.linalg.det(r), torch.ones(10), atol=1e-5))

    def test_radam_step(self):
        radam = RiemannianAdam(lr=0.01)
        x = sample_uniform_so3((5, 21))
        # Arbitrary ambient gradient
        grad = torch.randn_like(x)
        x_next = radam.step(x, grad)

        # Check result is strictly in SO(3)
        eye = torch.eye(3).expand_as(x_next)
        self.assertTrue(torch.allclose(torch.matmul(x_next, x_next.transpose(-1, -2)), eye, atol=1e-5))
        self.assertTrue(torch.allclose(torch.linalg.det(x_next), torch.ones(5, 21), atol=1e-5))

    def test_kinematics_identity(self):
        fk = DifferentiableKinematics()
        # Identity rotations for all 21 joints
        eye_rotations = torch.eye(3).expand(2, 21, 3, 3)
        joints = fk(eye_rotations)
        self.assertEqual(joints.shape, (2, 22, 3))


if __name__ == "__main__":
    unittest.main()
