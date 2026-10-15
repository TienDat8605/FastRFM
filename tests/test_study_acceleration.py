import unittest
import torch
from src.study.geometry import exp
from src.study.inference import rollout
from src.study.acceleration import checkpoint_rollout, local_guidance


class AccelerationTests(unittest.TestCase):
    def test_checkpoint_preserves_endpoint_penalty_and_gradient(self):
        class Field(torch.nn.Module):
            def forward(self, x, t):
                return .1*x[..., 0]+t[:, None, None]*.03
        torch.manual_seed(7)
        x = exp(torch.randn(2, 3, 3, dtype=torch.float64)*.2)
        outputs = []
        for checked in (False, True):
            d = torch.zeros(2, 3, 3, dtype=torch.float64, requires_grad=True)
            state = x@exp(d)
            pred, penalty = (checkpoint_rollout(Field(), state, 7, 3) if checked else
                             rollout(Field(), state, 7, trajectory=True))
            loss = pred[..., 0, 1].sum()+.01*penalty.sum()
            gradient, = torch.autograd.grad(loss, d)
            outputs.append((pred, penalty, gradient))
        for left, right in zip(*outputs):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_local_guidance_descent_and_valid_rotations(self):
        class Zero(torch.nn.Module):
            def forward(self, x, t):
                return x.new_zeros(*x.shape[:2], 3)
        initial = exp(torch.tensor([[[.3, -.2, .1]]], dtype=torch.float64))
        target = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
        objective = lambda x: (x-target).square().sum()
        pred, history = local_guidance(Zero(), initial, objective, 10)
        self.assertLess(objective(pred), objective(initial))
        torch.testing.assert_close(pred.transpose(-1, -2)@pred, target)
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
