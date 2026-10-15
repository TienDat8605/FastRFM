import unittest
import torch
from src.study.geometry import exp, log, project, sample, TransportAdam
from src.study.inference import rollout, optimize


class GeometryStudyTests(unittest.TestCase):
    def test_roundtrip_small_and_near_pi(self):
        v = torch.tensor([[0., 0., 0.], [1e-8, 0., 0.], [3.141592-1e-5, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(log(exp(v)), v, atol=1e-7, rtol=1e-7)

    def test_polar_derivative_off_manifold(self):
        torch.manual_seed(3)
        r = exp(torch.randn(2, 3, dtype=torch.float64)*.3)
        m = (r@torch.diag(torch.tensor([1.1, 1.2, 1.3], dtype=torch.float64))).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(project, (m,), atol=1e-5))

    def test_log_gradient(self):
        v = torch.tensor([[.02, -.12, .25]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: log(exp(x)), (v,)))

    def test_positive_geodesic_velocity(self):
        a = exp(torch.tensor([[.3, .2, -.1]], dtype=torch.float64))
        b = exp(torch.tensor([[-.1, .5, .2]], dtype=torch.float64))
        w = log(a.transpose(-1, -2)@b)
        torch.testing.assert_close(a@exp(w), b)
        tangent = torch.randn_like(w)
        moved = exp(-tangent/2)@w[..., None]
        torch.testing.assert_close(moved.squeeze(-1).norm(dim=-1), w.norm(dim=-1))

    def test_adam_transport(self):
        x = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
        g = torch.tensor([[[.4, .2, -.3]]], dtype=torch.float64)
        opt = TransportAdam(.1)
        y = opt.step(x, g)
        delta = log(x.transpose(-1, -2)@y)
        torch.testing.assert_close(opt.m, (exp(-delta/2)@(.1*g)[..., None]).squeeze(-1))
        torch.testing.assert_close(y.transpose(-1, -2)@y, x)

    def test_short_ode_gradient(self):
        class Field(torch.nn.Module):
            def forward(self, x, t):
                return .05*log(x)+.1
        x = exp(torch.tensor([[[.2, .3, -.1]]], dtype=torch.float64))
        d = torch.zeros(1, 1, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: rollout(Field(), x@exp(z), 3)[0], (d,), atol=1e-4))

    def test_identical_hypotheses_zero_apd(self):
        p = torch.randn(4, 5, 3)
        repeats = p[:, None].expand(-1, 10, -1, -1)
        self.assertEqual(float((repeats[:, 0]-repeats[:, 1]).norm()), 0.)

    def test_frozen_model_and_zero_rotation_loss(self):
        class Field(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.velocity = torch.nn.Parameter(torch.tensor([.1, .2, .3], dtype=torch.float64), requires_grad=False)
            def forward(self, x, t):
                return self.velocity.expand(*x.shape[:2], 3)
        field = Field()
        saved = field.velocity.clone()
        initial = sample((2, 1), 'cpu', torch.float64)
        target = rollout(field, initial, 3, 'intrinsic')[0].detach()
        self.assertLess(float(log(target.transpose(-1, -2)@target).norm()), 1e-10)
        optimize(field, initial, lambda x: (x-target).square().mean(), 3, 'intrinsic', 2, .1)
        torch.testing.assert_close(saved, field.velocity)

    def test_source_optimizer_is_batch_invariant(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros(*x.shape[:2], 3, dtype=x.dtype)
        x = exp(torch.tensor([[[.2, -.1, .3]], [[-.3, .1, .2]]], dtype=torch.float64))
        objective = lambda r: log(r).square().sum((-1, -2)).mean()
        together = optimize(Zero(), x, objective, 3, 'intrinsic', 4, .1, direct=True)[0]
        separate = torch.cat([optimize(Zero(), row[None], objective, 3, 'intrinsic', 4, .1, direct=True)[0]
                              for row in x])
        torch.testing.assert_close(together, separate)


if __name__ == '__main__':
    unittest.main()
