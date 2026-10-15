import unittest
import torch
from src.study.geometry import exp
from src.acceleration.adjoint import integrate, adjoint_integrate


class AdjointTests(unittest.TestCase):
    def test_matches_full_discrete_gradient_with_running_cost(self):
        class Field(torch.nn.Module):
            def forward(self, x, t):
                return .08*x[..., 0]+.1*t[:, None, None]
        torch.manual_seed(4)
        source = exp(torch.randn(2, 3, 3, dtype=torch.float64)*.2)
        for solver in ('euler', 'midpoint'):
            values = []
            for adjoint in (False, True):
                delta = torch.zeros(2, 3, 3, dtype=torch.float64, requires_grad=True)
                initial = source@exp(delta)
                output, penalty = (adjoint_integrate(Field(), initial, 9, solver, tolerance=1e-12)
                                   if adjoint else integrate(Field(), initial, 9, solver))
                loss = output[..., 0, 1].sum()+.03*penalty.mean()
                gradient, = torch.autograd.grad(loss, delta)
                values.append((output, penalty, gradient))
            for left, right in zip(*values):
                torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-9)

    def test_prefix_fallback_and_terminal_only_cost(self):
        class Field(torch.nn.Module):
            def forward(self, x, t):
                return .3*x[..., 0]+.2
        source = exp(torch.tensor([[[.2, -.3, .1]]], dtype=torch.float64))
        gradients = []
        stats = {}
        for adjoint in (False, True):
            initial = source.clone().requires_grad_()
            output = (adjoint_integrate(Field(), initial, 4, trajectory=False,
                                       tolerance=1e-15, inverse_iterations=1, stats=stats)[0]
                      if adjoint else integrate(Field(), initial, 4, trajectory=False)[0])
            gradients.append(torch.autograd.grad(output[..., 0, 1].sum(), initial)[0])
        self.assertGreater(stats['prefix_fallbacks'], 0)
        torch.testing.assert_close(*gradients, atol=1e-10, rtol=1e-10)
