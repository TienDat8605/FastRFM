"""Constant-trajectory-storage discrete adjoint for intrinsic SO(3) solvers.

Reconstruct each preceding state by fixed-point inversion, differentiate one
discrete step, and include the running cost's forcing term. If inversion fails
its residual check, replay that prefix from the original source. This bounds
stored state/activation memory independently of N, at increased computation.
A fixed four-interval anchor scheme limits long-horizon reverse drift.
It is a discrete-adjoint implementation, not the continuous OC-Flow algorithm.
"""
import torch
from src.study.geometry import exp


def step(model, state, index, steps, solver='euler'):
    t = state.new_full((len(state),), index/steps)
    velocity = model(state, t)
    if solver == 'midpoint':
        middle = state@exp(velocity/(2*steps))
        velocity = model(middle, t+1/(2*steps))
    elif solver != 'euler':
        raise ValueError(f'Unknown solver: {solver}')
    return state@exp(velocity/steps)


def integrate(model, source, steps=100, solver='euler', trajectory=True):
    """Full-graph reference, with post-step running costs summed per example."""
    if steps < 1:
        raise ValueError('Positive ODE steps required')
    state = source
    penalty = source.new_zeros(len(source))
    for i in range(steps):
        state = step(model, state, i, steps, solver)
        if trajectory:
            penalty = penalty+(3-state.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1)
    return state, penalty


class _DiscreteAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, model, steps, solver, trajectory, tolerance, inverse_iterations, stats):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Adjoint inference requires frozen model parameters')
        # A fixed number of anchors limits reverse drift without N-dependent storage.
        indices = sorted(set([0, steps]+[steps*i//4 for i in (1, 2, 3)]))
        anchors = {0: source}
        result, penalty = source, source.new_zeros(len(source))
        for i in range(steps):
            result = step(model, result, i, steps, solver)
            if trajectory:
                penalty = penalty+(3-result.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1)
            if i+1 in indices:
                anchors[i+1] = result
        ctx.save_for_backward(*(anchors[i] for i in indices))
        ctx.anchor_indices = indices
        ctx.model, ctx.steps, ctx.solver, ctx.trajectory = model, steps, solver, trajectory
        ctx.tolerance, ctx.inverse_iterations, ctx.stats = tolerance, inverse_iterations, stats
        return result, penalty

    @staticmethod
    def backward(ctx, terminal_gradient, penalty_gradient):
        anchors = dict(zip(ctx.anchor_indices, ctx.saved_tensors))
        source, terminal = anchors[0], anchors[ctx.steps]
        model, steps, solver = ctx.model, ctx.steps, ctx.solver
        adjoint = torch.zeros_like(terminal) if terminal_gradient is None else terminal_gradient
        forcing = source.new_zeros(len(source)) if penalty_gradient is None else penalty_gradient
        state = terminal.detach()
        stats = ctx.stats
        for i in reversed(range(steps)):
            with torch.no_grad():
                previous = state
                converged = False
                for _ in range(ctx.inverse_iterations):
                    t = previous.new_full((len(previous),), i/steps)
                    velocity = model(previous, t)
                    if solver == 'midpoint':
                        middle = previous@exp(velocity/(2*steps))
                        velocity = model(middle, t+1/(2*steps))
                    previous = state@exp(-velocity/steps)
                    residual = (step(model, previous, i, steps, solver)-state).abs().max().item()
                    stats['inverse_iterations'] = stats.get('inverse_iterations', 0)+1
                    if residual <= ctx.tolerance:
                        converged = True
                        break
                if not converged:
                    # Reference replay is deterministic and does not save a trajectory.
                    begin = max(index for index in anchors if index <= i)
                    previous = anchors[begin].detach()
                    for j in range(begin, i):
                        previous = step(model, previous, j, steps, solver)
                    stats['prefix_fallbacks'] = stats.get('prefix_fallbacks', 0)+1
                stats['max_local_inverse_residual'] = max(stats.get('max_local_inverse_residual', 0.), residual)
                if i in anchors:
                    stats['max_anchor_drift'] = max(stats.get('max_anchor_drift', 0.),
                                                   (previous-anchors[i]).abs().max().item())
                    previous = anchors[i].detach()
            with torch.enable_grad():
                leaf = previous.detach().requires_grad_()
                following = step(model, leaf, i, steps, solver)
                if ctx.trajectory:
                    cost = (3-following.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1)
                    adjoint, = torch.autograd.grad((following, cost), leaf,
                                                  (adjoint, forcing))
                else:
                    adjoint, = torch.autograd.grad(following, leaf, adjoint)
            state = previous.detach()
        stats['source_reconstruction_max_abs'] = max(stats.get('source_reconstruction_max_abs', 0.),
                                                     (state-source).abs().max().item())
        return adjoint, None, None, None, None, None, None, None


def adjoint_integrate(model, source, steps=100, solver='euler', trajectory=True,
                      tolerance=1e-6, inverse_iterations=12, stats=None):
    """O(1) stored trajectory/activation memory in N; inverse is tolerance-limited.

    The function stores at most five fixed anchors and creates a single-step
    graph during backward. Frozen network weights and batch size are additional
    memory costs. Peak allocator reservation is not equivalent to live storage.
    """
    if steps < 1 or inverse_iterations < 1 or tolerance <= 0:
        raise ValueError('Positive solver/inversion settings required')
    return _DiscreteAdjoint.apply(source, model, steps, solver, trajectory,
                                 tolerance, inverse_iterations, {} if stats is None else stats)
