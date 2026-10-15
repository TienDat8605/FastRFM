"""Benchmark deterministic four-step manifold distillation for source guidance.

The twist uses transported central directional responses, with half of source
directions following the teacher's observed-joint task gradient. This is an
RCM-inspired endpoint-map experiment, not a reproduction of RCM training.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.acceleration.distillation import (directional_response, endpoint_error,
                                           transported_response_error)
from src.study.data import load, sha256
from src.study.geometry import exp, log, sample
from src.study.inference import initialize, rollout
from src.study.acceleration import optimize_with_rollout
from src.study.runtime import OfficialBody, completion_metrics, load_prior, weight_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/colab/staging'))
    p.add_argument('--output', type=Path, default=Path('results/distillation_pilot'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--updates', type=int, default=600)
    p.add_argument('--train-sources', type=int, default=512)
    p.add_argument('--heldout-sources', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--response-weight', type=float, default=.1)
    p.add_argument('--epsilon', type=float, default=.02)
    p.add_argument('--evaluation-iterations', type=int, default=200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Output already exists; select a new --output directory')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize()
    teacher_path = args.root/'reference/pose_rfm_3dpw.pt'
    teacher, saved = load_prior(args.root, teacher_path, device)
    original_hash = weight_hash(teacher)
    def teacher_map(x, trajectory=False):
        return rollout(teacher, x, 100, 'final_projection', trajectory=trajectory)
    def student_map(model, x, trajectory=False):
        return rollout(model, x, 4, 'intrinsic', trajectory=trajectory)
    train = load(args.root, 'train')
    real = np.flatnonzero([not str(s).startswith('mock_') for s in train['sequence']])
    count = args.train_sources+args.heldout_sources
    chosen = np.random.default_rng(42).choice(real, count, replace=False)
    mean = exp(torch.tensor(train['poses'][real].mean(0), device=device).reshape(21, 3))
    gt = exp(torch.tensor(train['poses'][chosen], device=device).reshape(-1, 21, 3))
    visible = torch.ones(count, 21, dtype=torch.bool, device=device)
    visible[:, [0, 1, 3, 4, 6, 7, 9, 10]] = False
    # Independently balance mask, source type, and direction type across eight rows.
    torso = torch.arange(count, device=device) % 4 >= 2
    visible[torso] = True
    visible[torso, :] = True
    torso_indices = torso.nonzero().flatten()
    visible[torso_indices[:, None], torch.tensor([2, 5, 8, 15, 16], device=device)] = False
    observations = torch.where(visible[..., None, None], gt, mean)
    manifest = {'date': '2026-10-14', 'stage': 'exploratory', 'device': str(device),
        'torch': torch.__version__, 'platform': platform.platform(), 'cpu_threads': 1,
        'teacher_sha256': sha256(teacher_path), 'teacher_epoch': saved.get('epoch'),
        'train_data_sha256': sha256(args.root/'data/3dpw/train.npz'),
        'teacher_solver': '100-step extrinsic Euler, final projection',
        'student_solver': '4-step intrinsic Euler', 'teacher_training_indices': chosen.tolist(),
        'train_sources': args.train_sources, 'heldout_sources': args.heldout_sources,
        'source_mixture': '50% Haar, 50% backward-initialized training-only partial poses',
        'direction_mixture': '50% random per-joint unit vectors, 50% teacher source task-gradient',
        'epsilon': args.epsilon, 'response_weight': args.response_weight,
        'updates': args.updates, 'batch_size': args.batch_size, 'lr': .0001,
        'validation_selection': 'Minimum heldout endpoint geodesic loss, identical across students',
        'evaluation_iterations': args.evaluation_iterations, 'evaluation_seeds': [0, 1],
        'codes': {str(f): sha256(f) for f in [Path(__file__), Path('src/acceleration/distillation.py')]},
        'caveats': ['Not RCM continuous-time training or stochastic sampler',
                    'Not a guarantee of covariance/Jacobian preservation',
                    'Existing HMR prior has prior test exposure; validation screening only',
                    'One training seed and small validation sample; novelty unestablished',
                    'Teacher and student have different trajectory discretizations']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    sync(); start = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(42)
    sources = sample((count, 21), device, generator=generator)
    # Every second row is an observation-conditioned source; others stay Haar.
    for begin in range(0, count, args.batch_size):
        end = min(count, begin+args.batch_size)
        with torch.no_grad():
            backward = initialize(teacher, observations[begin:end], 100, 'final_projection', .25,
                                  noise=sources[begin:end])
            use = (torch.arange(begin, end, device=device) % 2 == 1)
            sources[begin:end] = torch.where(use[:, None, None, None], backward, sources[begin:end])
    targets, responses, directions = [], [], []
    for begin in range(0, count, args.batch_size):
        end = min(count, begin+args.batch_size)
        source, obs, mask = sources[begin:end], observations[begin:end], visible[begin:end]
        u = torch.randn(len(source), 21, 3, device=device, generator=generator)
        u = u/u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = source.new_zeros(len(source), 21, 3).requires_grad_()
        pred = teacher_map(source@exp(delta))[0]
        observed_angles = log(pred.transpose(-1, -2)@obs).square().sum(-1)
        task_loss = (observed_angles*mask).sum(-1).mean()
        gradient, = torch.autograd.grad(task_loss, delta)
        task_direction = gradient/gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        use = (torch.arange(begin, end, device=device) % 8 >= 4)
        u = torch.where(use[:, None, None], task_direction, u).detach()
        with torch.no_grad():
            base = teacher_map(source)[0]
            plus = teacher_map(source@exp(args.epsilon*u))[0]
            minus = teacher_map(source@exp(-args.epsilon*u))[0]
            response = directional_response(base, plus, minus, args.epsilon)
        targets.append(base); responses.append(response); directions.append(u)
        print(f'teacher targets {end}/{count}', flush=True)
    targets, responses, directions = map(torch.cat, (targets, responses, directions))
    sync(); generation_seconds = time.perf_counter()-start
    tensors = {'sources': sources, 'targets': targets, 'responses': responses, 'directions': directions}
    torch.save({k: v.cpu() for k, v in tensors.items()}, args.output/'teacher_pairs.pt')
    def measure_model(model, indices):
        with torch.no_grad():
            source, u = sources[indices], directions[indices]
            pred = student_map(model, source)[0]
            plus = student_map(model, source@exp(args.epsilon*u))[0]
            minus = student_map(model, source@exp(-args.epsilon*u))[0]
            response = directional_response(pred, plus, minus, args.epsilon)
            return {'endpoint_loss': endpoint_error(pred, targets[indices]).item(),
                'endpoint_mean_rad': log(pred.transpose(-1, -2)@targets[indices]).norm(dim=-1).mean().item(),
                'response_loss': transported_response_error(pred, response,
                                     targets[indices], responses[indices]).item()}
    heldout = torch.arange(args.train_sources, count, device=device)
    models = {'raw4': deepcopy(teacher)}
    training = {'teacher_pair_generation_seconds': generation_seconds,
                'raw4': measure_model(models['raw4'], heldout)}
    for name, coefficient in [('endpoint4', 0.), ('transport4', args.response_weight)]:
        model = deepcopy(teacher).requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
        generator = torch.Generator(device=device).manual_seed(100)
        best = measure_model(model, heldout)
        best_weights = deepcopy(model.state_dict())
        best_step = 0
        history = []
        sync(); start = time.perf_counter()
        for step in range(1, args.updates+1):
            ids = torch.randint(args.train_sources, (args.batch_size,), device=device, generator=generator)
            source, target, u = sources[ids], targets[ids], directions[ids]
            pred = student_map(model, source)[0]
            endpoint = endpoint_error(pred, target)
            loss = endpoint
            if coefficient:
                plus = student_map(model, source@exp(args.epsilon*u))[0]
                minus = student_map(model, source@exp(-args.epsilon*u))[0]
                response = directional_response(pred, plus, minus, args.epsilon)
                sensitivity = transported_response_error(pred, response, target, responses[ids])
                warmup = min(1., step/100)
                loss = loss+coefficient*warmup*sensitivity
            if not torch.isfinite(loss): raise FloatingPointError(f'{name}: nonfinite distillation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0 or step == args.updates:
                validation = measure_model(model, heldout)
                history.append({'step': step, **validation})
                print(name, history[-1], flush=True)
                if validation['endpoint_loss'] < best['endpoint_loss']:
                    best, best_step = validation, step
                    best_weights = deepcopy(model.state_dict())
        sync(); elapsed = time.perf_counter()-start
        model.load_state_dict(best_weights)
        model.eval().requires_grad_(False)
        torch.save({'model_state_dict': best_weights, 'best_step': best_step,
                    'args': saved['args'], 'teacher_sha256': manifest['teacher_sha256']}, args.output/f'{name}.pt')
        models[name] = model
        training[name] = {'seconds': elapsed, 'selected_update': best_step,
                           'heldout': measure_model(model, heldout), 'history': history}
        (args.output/'training.json').write_text(json.dumps(training, indent=2))
    # Check central-difference stability for the frozen teacher on heldout sources.
    stability = {}
    with torch.no_grad():
        subset = heldout[:32]
        for epsilon in [args.epsilon/2, args.epsilon, args.epsilon*2]:
            plus = teacher_map(sources[subset]@exp(epsilon*directions[subset]))[0]
            minus = teacher_map(sources[subset]@exp(-epsilon*directions[subset]))[0]
            response = directional_response(targets[subset], plus, minus, epsilon)
            reference = responses[subset]
            stability[str(epsilon)] = {'relative_response_difference':
                ((response-reference).norm()/reference.norm().clamp_min(1e-8)).item()}
    training['finite_difference_stability'] = stability
    (args.output/'training.json').write_text(json.dumps(training, indent=2))
    data = load(args.root, 'validation')
    pilot_manifest = json.loads(Path('results/acceleration_pilot/manifest.json').read_text())
    ids = pilot_manifest['ids']
    manifest['evaluation_ids'] = ids
    manifest['evaluation_validation_sha256'] = sha256(args.root/'data/3dpw/validation.npz')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    body = OfficialBody(args.root, device)
    truth = exp(torch.tensor(data['poses'][ids], device=device).reshape(-1, 21, 3))
    observed = torch.ones(21, dtype=torch.bool, device=device)
    observed[[0, 1, 3, 4, 6, 7, 9, 10]] = False
    observations = truth.clone()
    observations[:, ~observed] = mean[~observed]
    observations = observations.repeat_interleave(2, 0)
    objective = lambda pred: log(pred.transpose(-1, -2)@observations)[:, observed].norm(dim=-1).sum(-1).mean()
    rows = []
    for seed in [0, 1]:
        noise = torch.cat([sample((2, 21), device,
            generator=torch.Generator(device=device).manual_seed(seed*1000003+case)) for case in ids])
        sync(); start = time.perf_counter()
        source = initialize(teacher, observations, 100, 'final_projection', .25, noise=noise)
        sync(); init_time = time.perf_counter()-start
        def task_gradient(integrate):
            delta = source.new_zeros(*source.shape[:2], 3).requires_grad_()
            prediction = integrate(source@exp(delta))[0]
            loss = objective(prediction)
            gradient, = torch.autograd.grad(loss, delta)
            return gradient.detach()
        teacher_gradient = task_gradient(teacher_map)
        candidates = [('teacher100', teacher), *models.items()]
        if seed % 2: candidates.reverse()
        for name, model in candidates:
            integrate = (lambda x: teacher_map(x, True)) if name == 'teacher100' else lambda x, model=model: student_map(model, x, True)
            gradient = task_gradient(integrate)
            cosine = F.cosine_similarity(gradient.flatten(1), teacher_gradient.flatten(1)).mean().item()
            sync()
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            pred, history = optimize_with_rollout(source, objective, integrate, args.evaluation_iterations, .1,
                                                  .001 if name == 'teacher100' else .025)
            sync(); solver_time = time.perf_counter()-start
            peak = {'allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                    'reserved_mib': torch.cuda.max_memory_reserved()/2**20} if device.type == 'cuda' else None
            start = time.perf_counter()
            metrics, replaced = completion_metrics(body, pred, truth, observed, 2)
            sync(); metric_time = time.perf_counter()-start
            row = {'method': name, 'seed': seed, 'status': 'complete', 'solver_seconds': solver_time,
                'initialization_seconds': init_time, 'end_to_end_seconds': solver_time+init_time+metric_time,
                'metrics': {k: float(np.mean(v)) for k, v in metrics.items()}, 'per_case': metrics,
                'source_task_gradient_cosine': cosine, 'peak_cuda_mib': peak, 'history': history}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('history', 'per_case')}), flush=True)
            torch.save({'predictions': replaced.cpu(), 'gt': truth.cpu()}, args.output/f'{seed}_{name}.pt')
            (args.output/'results.json').write_text(json.dumps(rows, indent=2))
    if weight_hash(teacher) != original_hash: raise RuntimeError('Teacher weights changed')
    lines = ['# Task-directed transported-response distillation pilot', '', '## Material Passport', '',
        'Mode: run; date: 2026-10-14; status: exploratory executed experiment; '
        'scope: four-step deterministic endpoint-map compression, validation legs completion.', '',
        f'Device {device}; one distillation seed; {args.train_sources} training and '
        f'{args.heldout_sources} heldout training-only sources; eight validation poses, two hypotheses, '
        'two inference seeds. CPU results do not establish T4 speed or VRAM savings.', '',
        '| Method | End-to-end s | Speedup | Hidden MPVPE mm | Visible rad | APD cm | Source gradient cosine |',
        '|---|---:|---:|---:|---:|---:|---:|']
    baseline = np.mean([r['end_to_end_seconds'] for r in rows if r['method'] == 'teacher100'])
    for name in ['teacher100', 'raw4', 'endpoint4', 'transport4']:
        values = [r for r in rows if r['method'] == name]
        elapsed = np.mean([r['end_to_end_seconds'] for r in values])
        metric = {k: np.mean([r['metrics'][k] for r in values]) for k in values[0]['metrics']}
        cosine = np.mean([r['source_task_gradient_cosine'] for r in values])
        lines.append(f"| {name} | {elapsed:.2f} | {baseline/elapsed:.2f}x | {metric['mpvpe_mm']:.2f} | "
                     f"{metric['visible_rad']:.3f} | {metric['apd_cm']:.2f} | {cosine:.3f} |")
    lines += ['', '## Teacher-map fidelity on heldout training sources', '',
        '| Student | Mean endpoint rad | Transported response MSE | Selected update | Training s |',
        '|---|---:|---:|---:|---:|']
    for name in models:
        cell = training[name]
        fidelity = cell if name == 'raw4' else cell['heldout']
        lines.append(f"| {name} | {fidelity['endpoint_mean_rad']:.4f} | {fidelity['response_loss']:.6f} | "
                     f"{cell.get('selected_update', 0)} | {cell.get('seconds', 0):.2f} |")
    lines += ['', f'Teacher-pair generation: {generation_seconds:.2f} seconds. '
        'Training and generation are excluded from inference timing and must be amortized over use.', '',
        'The transported-response candidate is an extension inspired by RCM, established Jacobian matching, '
        'and noise-sensitivity distillation. Novelty is unestablished. It does not inherit the teacher\'s '
        'covariance interpretation. Finite-difference stability is in training.json. '
        'Endpoint-only and response-aware students share data, initialization, sampling, updates, '
        'and endpoint-based checkpoint selection; the response-aware model has higher training compute. '
        'Four-step students use intrinsic Euler and a quadrature-rescaled trajectory weight; their '
        'trajectory objectives differ from the teacher. Tests evaluate transport and finite derivatives. '
        'No test cohort or AMASS poses are used here. Expand validation and evaluate denoising and '
        'GPU memory before promoting any candidate.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
