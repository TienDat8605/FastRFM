"""Task-directed transported-response distillation on SO(3)^K.

This is deterministic endpoint-map compression inspired by RCM, not its
continuous-time training objective or stochastic multistep sampler.
"""
from src.study.geometry import exp, log


def endpoint_error(prediction, target):
    return log(prediction.transpose(-1, -2)@target).square().sum(-1).mean()


def directional_response(base, plus, minus, epsilon):
    """Central finite directional derivative in endpoint body coordinates."""
    if epsilon <= 0:
        raise ValueError('Positive epsilon required')
    return (log(base.transpose(-1, -2)@plus)-log(base.transpose(-1, -2)@minus))/(2*epsilon)


def transported_response_error(student_base, student_response, teacher_base, teacher_response):
    """Compare vectors after Levi-Civita transport under the bi-invariant metric.

    a=Log(YS^T YT); transporting teacher -> student gives Ad_Exp(a/2).
    For SO(3) vector coordinates, Ad_R v = R v.
    """
    a = log(student_base.transpose(-1, -2)@teacher_base)
    transported = (exp(a/2)@teacher_response[..., None]).squeeze(-1)
    return (student_response-transported).square().sum(-1).mean()


def exact_pullback(flow, source, covector, reference=None, create_graph=False):
    """Exact chart VJP; endpoint base has no dependency on source perturbation.

    Covectors use the half-Frobenius/body Euclidean metric. If reference is
    provided, transport its body covector to the independently evaluated base.
    Parameter derivatives through base and transport are retained.
    """
    import torch
    from src.study.geometry import hat
    base = flow(source)
    if reference is not None:
        a = log(base.transpose(-1, -2)@reference)
        covector = (exp(a/2)@covector[..., None]).squeeze(-1)
    delta = source.new_zeros(*source.shape[:2], 3).requires_grad_()
    perturbed = flow(source@exp(delta))
    functional = .5*((base@hat(covector))*perturbed).sum()
    result, = torch.autograd.grad(functional, delta, create_graph=create_graph)
    return base, result


def normalized_pullback_error(prediction, target):
    return (prediction-target).square().mean()/target.square().mean().clamp_min(1e-8)
