"""Experimental acceleration candidates; the frozen study solver is unchanged."""
import torch
from torch.utils.checkpoint import checkpoint
from .geometry import exp, hat, project, TransportAdam


def checkpoint_rollout(model, x, steps=100, block_size=10, trajectory=True):
    """Recompute blocks during backward, preserving the discrete Euler objective."""
    if steps < 1 or block_size < 1:
        raise ValueError('Positive steps and block_size required')
    penalty = x.new_zeros(len(x))
    for begin in range(0, steps, block_size):
        end = min(begin+block_size, steps)
        # Bind bounds now: checkpoint invokes this closure again during backward.
        def block(state, begin=begin, end=end):
            total = state.new_zeros(len(state))
            for i in range(begin, end):
                t = state.new_full((len(state),), i/steps)
                state = state+(state@hat(model(state, t)))/steps
                if trajectory:
                    total = total+(3-state.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1)
            return state, total
        if torch.is_grad_enabled() and x.requires_grad:
            x, increment = checkpoint(block, x, use_reentrant=False)
        else:
            x, increment = block(x)
        penalty = penalty+increment
    return project(x), penalty


def optimize_with_rollout(initial, objective, integrate, iterations, lr, trajectory_weight):
    x = initial.detach()
    optimizer = TransportAdam(lr)
    history = []
    for _ in range(iterations):
        delta = x.new_zeros(*x.shape[:2], 3).requires_grad_()
        pred, penalty = integrate(x@exp(delta))
        loss = objective(pred)+trajectory_weight*penalty.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite accelerated objective')
        gradient, = torch.autograd.grad(loss, delta)
        x = optimizer.step(x, gradient*len(x))
        history.append(float(loss.detach()))
    with torch.no_grad():
        pred = integrate(x)[0]
    return pred, history


def local_guidance(model, initial, objective, steps=20, strength=1., clip=1.):
    """One intrinsic rollout with endpoint-surrogate guidance.

    This is a heuristic, not OC-Flow or an exact conditional-mean estimator.
    Each local graph is discarded; gradients do not reach the original source.
    """
    x = initial.detach()
    history = []
    for i in range(steps):
        delta = x.new_zeros(*x.shape[:2], 3).requires_grad_()
        state = x@exp(delta)
        t = state.new_full((len(state),), i/steps)
        velocity = model(state, t)
        endpoint = state@exp((1-i/steps)*velocity)
        loss = objective(endpoint)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite local-guidance objective')
        gradient, = torch.autograd.grad(loss, delta)
        gradient = gradient*len(x)
        gradient = gradient*(clip/gradient.norm(dim=-1, keepdim=True).clamp_min(clip))
        with torch.no_grad():
            x = x@exp((velocity.detach()-strength*gradient)/steps)
        history.append(float(loss.detach()))
    return x, history
