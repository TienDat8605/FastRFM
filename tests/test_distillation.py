import unittest
import torch
from src.study.geometry import exp, sample
from src.acceleration.distillation import directional_response, transported_response_error


class DistillationTests(unittest.TestCase):
    def test_identity_map_directional_response(self):
        torch.manual_seed(1)
        x = sample((3, 4), 'cpu', torch.float64)
        u = torch.randn(3, 4, 3, dtype=torch.float64)*.1
        response = directional_response(x, x@exp(.01*u), x@exp(-.01*u), .01)
        torch.testing.assert_close(response, u)

    def test_parallel_transport_roundtrip(self):
        torch.manual_seed(2)
        student = sample((2, 3), 'cpu', torch.float64)
        a = torch.randn(2, 3, 3, dtype=torch.float64)*.2
        teacher = student@exp(a)
        vector = torch.randn_like(a)
        at_teacher = (exp(-a/2)@vector[..., None]).squeeze(-1)
        loss = transported_response_error(student, vector, teacher, at_teacher)
        self.assertLess(loss.item(), 1e-20)

    def test_response_loss_has_parameter_gradients(self):
        x = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
        scale = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
        u = torch.tensor([[[.1, -.2, .3]]], dtype=torch.float64)
        response = directional_response(x, x@exp(.01*scale*u), x@exp(-.01*scale*u), .01)
        loss = transported_response_error(x, response, x, u)
        gradient, = torch.autograd.grad(loss, scale)
        self.assertLess(gradient.item(), 0)
