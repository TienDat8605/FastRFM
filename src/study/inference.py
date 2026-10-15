"""Independent Riemannian D-Flow implementation; no model-weight updates."""
import torch
from .geometry import exp, hat, project, sample, TransportAdam


def rollout(model, x, steps=100, policy='final_projection', backward=False, trajectory=False):
    h = (-1 if backward else 1)/steps
    penalty = x.new_zeros(x.shape[0])
    for i in range(steps):
        t = (1-i/steps) if backward else i/steps
        velocity = model(x, torch.full((len(x),), t, device=x.device, dtype=x.dtype))
        x = x@exp(h*velocity) if policy == 'intrinsic' else x+h*(x@hat(velocity))
        if trajectory:
            penalty = penalty+(3-x.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1)
    # No differentiable SVD is needed for intrinsic integration.
    return (x if policy == 'intrinsic' else project(x)), penalty


@torch.no_grad()
def initialize(model, observation, steps, policy, alpha, generator=None, backward=True, noise=None):
    if noise is None:
        noise = sample(observation.shape[:2], observation.device, observation.dtype, generator)
    if not backward:
        return noise
    inverted, _ = rollout(model, observation, steps, policy, backward=True)
    return project(alpha**.5*noise+(1-alpha)**.5*inverted)


def optimize(model, initial, objective, steps, policy, iterations, lr, trajectory_weight=0., direct=False):
    x = initial.detach()
    optimizer = TransportAdam(lr)
    history = []
    for _ in range(iterations):
        delta = torch.zeros(*x.shape[:2], 3, device=x.device, dtype=x.dtype, requires_grad=True)
        perturbed = x@exp(delta)
        pred, penalty = (perturbed, x.new_zeros(len(x))) if direct else rollout(
            model, perturbed, steps, policy, trajectory=trajectory_weight > 0)
        loss = objective(pred)+trajectory_weight*penalty.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite guidance objective')
        gradient, = torch.autograd.grad(loss, delta)
        # Undo the example average for independent source variables. This keeps
        # Adam's epsilon scale invariant when evaluation batches change.
        gradient = gradient*len(x)
        x = optimizer.step(x, gradient)
        history.append(float(loss.detach()))
    with torch.no_grad():
        pred = x if direct else rollout(model, x, steps, policy)[0]
    return pred, history
