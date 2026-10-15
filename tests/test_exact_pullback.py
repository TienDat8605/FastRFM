import unittest
import torch
from src.study.geometry import exp, sample
from src.acceleration.distillation import exact_pullback

class ExactPullbackTests(unittest.TestCase):
    def test_identity_metric(self):
        x = sample((2,3), 'cpu', torch.float64)
        v = torch.randn(2,3,3,dtype=torch.float64)
        _, g = exact_pullback(lambda z:z, x, v)
        torch.testing.assert_close(g,v)

    def test_right_rotation(self):
        x = sample((2,3), 'cpu', torch.float64)
        r = exp(torch.randn(2,3,3,dtype=torch.float64)*.2)
        v = torch.randn(2,3,3,dtype=torch.float64)
        _, g = exact_pullback(lambda z:z@r,x,v)
        torch.testing.assert_close(g,(r@v[...,None]).squeeze(-1))

    def test_mixed_derivative(self):
        x = sample((2,3), 'cpu', torch.float64)
        v = torch.randn(2,3,3,dtype=torch.float64)
        axis = torch.randn_like(v)*.2
        p = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
        def value(p):
            _,g = exact_pullback(lambda z:z@exp(p*axis),x,v,create_graph=True)
            return (g*v).sum()
        derivative, = torch.autograd.grad(value(p),p)
        finite = (value(p+1e-5)-value(p-1e-5))/2e-5
        torch.testing.assert_close(derivative,finite,rtol=1e-5,atol=1e-7)

    def test_reference_transport_sign(self):
        x = sample((2,3), 'cpu', torch.float64)
        a = torch.randn(2,3,3,dtype=torch.float64)*.2
        reference=x@exp(a)
        v=torch.randn_like(a)
        _,g=exact_pullback(lambda z:z,x,v,reference)
        torch.testing.assert_close(g,(exp(a/2)@v[...,None]).squeeze(-1))

    def test_frozen_teacher_and_student_weight_gradient(self):
        from src.acceleration.adjoint import integrate
        class Field(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
            def forward(self,x,t):
                return self.weight*x[...,0,:]
        teacher=Field().requires_grad_(False)
        student=Field()
        x=sample((2,3),'cpu',torch.float64);v=torch.randn(2,3,3,dtype=torch.float64)
        before=teacher.weight.detach().clone()
        endpoint,target=exact_pullback(lambda z:integrate(teacher,z,4,'midpoint',False)[0],x,v)
        _,g=exact_pullback(lambda z:integrate(student,z,2,'midpoint',False)[0],x,v,endpoint.detach(),True)
        loss=(g-target).square().sum();loss.backward()
        self.assertTrue(torch.isfinite(student.weight.grad))
        self.assertIsNone(teacher.weight.grad)
        torch.testing.assert_close(teacher.weight,before)
