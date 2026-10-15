"""SO(3) geometry with differentiable polar projection and transported Adam.

Metric: body coordinates have Euclidean norm (half the ambient Frobenius norm).
Adam uses one scalar second moment per SO(3) factor, so it is frame invariant.
"""
import torch


def hat(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def exp(v):
    q = v.square().sum(-1, keepdim=True)
    angle = q.clamp_min(1e-16).sqrt()
    a = torch.where(q < 1e-6, 1-q/6+q.square()/120, torch.sin(angle)/angle)
    b = torch.where(q < 1e-6, .5-q/24+q.square()/720, (1-torch.cos(angle))/q.clamp_min(1e-16))
    w = hat(v)
    return torch.eye(3, device=v.device, dtype=v.dtype) + a[..., None]*w + b[..., None]*(w@w)


def log(r):
    # Choose the best-conditioned quaternion branch; principal angle in [0, pi].
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    terms = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                         1-m00+m11-m22, 1-m00-m11+m22), -1)
    xy = r[..., 0, 1]+r[..., 1, 0]
    xz = r[..., 0, 2]+r[..., 2, 0]
    yz = r[..., 1, 2]+r[..., 2, 1]
    dx = r[..., 2, 1]-r[..., 1, 2]
    dy = r[..., 0, 2]-r[..., 2, 0]
    dz = r[..., 1, 0]-r[..., 0, 1]
    rows = torch.stack((torch.stack((terms[..., 0], dx, dy, dz), -1),
                        torch.stack((dx, terms[..., 1], xy, xz), -1),
                        torch.stack((dy, xy, terms[..., 2], yz), -1),
                        torch.stack((dz, xz, yz, terms[..., 3]), -1)), -2)
    rows = rows/(2*terms.clamp_min(1e-12).sqrt()[..., None])
    idx = terms.argmax(-1)[..., None, None].expand(*r.shape[:-2], 1, 4)
    q = rows.gather(-2, idx).squeeze(-2)
    q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    norm = v.norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(norm, q[..., :1])
    scale = torch.where(norm < 1e-6, 2+norm.square()/3,
                        angle/norm.clamp_min(1e-12))
    return scale*v


class Polar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m):
        u, s, vt = torch.linalg.svd(m)
        sign = torch.ones_like(s)
        sign[..., -1] = torch.where(torch.linalg.det(u@vt) < 0, -1., 1.)
        r = (u*sign[..., None, :])@vt
        ctx.save_for_backward(r, vt, s*sign)
        return r

    @staticmethod
    def backward(ctx, grad):
        r, vt, s = ctx.saved_tensors
        b = vt@(r.transpose(-1, -2)@grad)@vt.transpose(-1, -2)
        denominator = s[..., :, None]+s[..., None, :]
        if (denominator.abs() < 1e-10).any():
            raise FloatingPointError('Polar differential is singular at this matrix')
        skew = (b-b.transpose(-1, -2))/denominator
        return r@vt.transpose(-1, -2)@skew@vt


def project(m):
    return Polar.apply(m)


def sample(shape, device, dtype=torch.float32, generator=None):
    q = torch.randn(*shape, 4, device=device, dtype=dtype, generator=generator)
    q = q/q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*shape, 3, 3)


class TransportAdam:
    def __init__(self, lr, betas=(.9, .999), eps=1e-8):
        self.lr, self.betas, self.eps = lr, betas, eps
        self.m = self.v = None
        self.n = 0

    @torch.no_grad()
    def step(self, x, gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite source gradient')
        if self.m is None:
            self.m = torch.zeros_like(gradient)
            self.v = torch.zeros_like(gradient[..., :1])
        a, b = self.betas
        self.n += 1
        self.m = a*self.m+(1-a)*gradient
        self.v = b*self.v+(1-b)*gradient.square().sum(-1, keepdim=True)
        delta = -self.lr*(self.m/(1-a**self.n))/((self.v/(1-b**self.n)).sqrt()+self.eps)
        result = x@exp(delta)
        # Parallel transport under the bi-invariant metric in body coordinates.
        self.m = (exp(-delta/2)@self.m[..., None]).squeeze(-1)
        return result.detach()
