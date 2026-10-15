"""Paired validation-only completion pilot; no changes to the frozen full study.

Run: python scripts/run_acceleration_pilot.py --device cpu
CPU timings and saved-tensor payload are not GPU latency/VRAM measurements.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.study.acceleration import checkpoint_rollout, optimize_with_rollout, local_guidance
from src.study.data import load, balanced, sha256
from src.study.geometry import exp, log, sample
from src.study.inference import initialize, rollout
from src.study.runtime import OfficialBody, completion_metrics, load_prior, weight_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/colab/staging'))
    parser.add_argument('--output', type=Path, default=Path('results/acceleration_pilot'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--poses', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    dataset = load(args.root, 'validation')
    groups = {str(s): np.flatnonzero((dataset['sequence'] == s)&(dataset['frame'] % 10 == 0)).tolist()
              for s in np.unique(dataset['sequence']) if not str(s).startswith('mock_')}
    ids = balanced(groups, args.poses, np.random.default_rng(42))
    checkpoint_path = args.root/'reference/pose_rfm_3dpw.pt'
    model, saved = load_prior(args.root, checkpoint_path, device)
    frozen_hash = weight_hash(model)
    body = OfficialBody(args.root, device)
    train = load(args.root, 'train')
    clean_train = train['poses'][np.array([not str(s).startswith('mock_') for s in train['sequence']])]
    mean = exp(torch.tensor(clean_train.mean(0), device=device).reshape(21, 3))
    gt = exp(torch.tensor(dataset['poses'][ids], device=device).reshape(-1, 21, 3))
    visible = torch.ones(21, dtype=torch.bool, device=device)
    visible[[0, 1, 3, 4, 6, 7, 9, 10]] = False
    obs = gt.clone()
    obs[:, ~visible] = mean[~visible]
    hypotheses = 2
    obs = obs.repeat_interleave(hypotheses, 0)
    objective = lambda pred: log(pred.transpose(-1, -2)@obs)[:, visible].norm(dim=-1).sum(-1).mean()
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    with torch.no_grad():
        rollout(model, obs, 4)
    manifest = {'scope': 'Exploratory legs completion on 3DPW validation only; not full reproduction',
        'device': str(device), 'platform': platform.platform(), 'torch': torch.__version__,
        'cpu_threads': 1, 'checkpoint_sha256': sha256(checkpoint_path),
        'checkpoint_epoch': saved.get('epoch'), 'parameters': sum(p.numel() for p in model.parameters()),
        'validation_sha256': sha256(args.root/'data/3dpw/validation.npz'), 'body_sha256': body.model_hash,
        'ids': ids, 'sequences': [str(dataset['sequence'][i]) for i in ids],
        'hypotheses': hypotheses, 'seeds': args.seeds, 'iterations': args.iterations,
        'initialization_steps': 100, 'trajectory_weight': .001,
        'step_sweep_weight': 'lambda_N=.001*100/N; approximate quadrature parity',
        'caveats': ['Existing residual/time-embedding HMR prior, not paper MLP',
                    'Single measured batch per seed; no statistical speed guarantee',
                    'Shared 100-step backward initialization included in end-to-end times',
                    'APD based on only two hypotheses', 'No physical plausibility evaluator',
                    'Saved tensor payload counts saves, not unique storage or peak VRAM']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for seed in args.seeds:
        noise = torch.cat([sample((hypotheses, 21), device,
                     generator=torch.Generator(device=device).manual_seed(seed*1000003+case)) for case in ids])
        sync(); start = time.perf_counter()
        initial = initialize(model, obs, 100, 'final_projection', .25, noise=noise)
        sync(); init_seconds = time.perf_counter()-start
        candidates = [('baseline', 100, args.iterations, None),
                      ('checkpoint', 100, args.iterations, None),
                      ('steps20', 20, args.iterations, None),
                      ('steps4', 4, args.iterations, None),
                      ('iterations10', 100, 10, None)]
        candidates += [('local'+str(g), 20, 1, g) for g in (0., .1, 1., 10.)]
        # Alternate order to reduce first-run/order bias; baseline lookup uses names.
        if seed % 2:
            candidates.reverse()
        for name, steps, iterations, strength in candidates:
            print(f'seed={seed} {name} steps={steps} iterations={iterations}', flush=True)
            row = {'seed': seed, 'method': name, 'steps': steps, 'iterations': iterations,
                   'strength': strength, 'initialization_seconds': init_seconds,
                   'policy': 'intrinsic' if strength is not None else 'final_projection'}
            try:
                integrate = ((lambda x: checkpoint_rollout(model, x, steps, 10)) if name == 'checkpoint'
                    else (lambda x: rollout(model, x, steps, trajectory=True)))
                # Measure only the graph's saved payload in a separate single-iteration probe.
                if strength is None:
                    payload = [0]
                    def pack(tensor):
                        payload[0] += tensor.numel()*tensor.element_size()
                        return tensor
                    delta = initial.new_zeros(*initial.shape[:2], 3).requires_grad_()
                    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                        pred, penalty = integrate(initial@exp(delta))
                        loss = objective(pred)+(.001*100/steps)*penalty.mean()
                        torch.autograd.grad(loss, delta)
                    del pred, penalty, loss, delta
                    row['saved_tensor_payload_bytes'] = payload[0]
                sync()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                if strength is None:
                    pred, history = optimize_with_rollout(initial, objective, integrate,
                                      iterations, .1, .001*100/steps)
                else:
                    pred, history = local_guidance(model, initial, objective, steps, strength)
                sync(); row['solver_seconds'] = time.perf_counter()-start
                start = time.perf_counter()
                metrics, replaced = completion_metrics(body, pred, gt, visible, hypotheses)
                sync(); row['metrics_seconds'] = time.perf_counter()-start
                row['end_to_end_seconds'] = init_seconds+row['solver_seconds']+row['metrics_seconds']
                row['metrics'] = {key: float(np.mean(value)) for key, value in metrics.items()}
                row['per_case'] = metrics
                row['history'] = history
                row['status'] = 'complete'
                torch.save({'predictions': replaced.cpu(), 'gt': gt.cpu()}, args.output/f'{seed}_{name}.pt')
                print(json.dumps({k: v for k, v in row.items() if k not in ('history', 'per_case')}), flush=True)
            except (RuntimeError, FloatingPointError) as error:
                row.update(status='failed', error=repr(error))
                print('FAILED', repr(error), flush=True)
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(rows, indent=2))
            if weight_hash(model) != frozen_hash:
                raise RuntimeError('Frozen prior changed')
    report = ['# Acceleration pilot', '', '## Material Passport', '',
        'Mode: run; date: 2026-10-14; evidence: exploratory executed experiment; '
        'source: run_acceleration_pilot.py; scope: validation legs completion.', '',
        f'Device: {device}; {len(ids)} poses, two hypotheses, {len(args.seeds)} seeds. '
        'Times include shared 100-step initialization and body metrics. '
        'CPU measurements do not establish GPU speed or memory savings.', '',
        '| Method | End-to-end s | Speedup | MPVPE mm | Change mm | Visible rad | APD cm | Saved payload MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    baseline = [r for r in rows if r['method'] == 'baseline' and r['status'] == 'complete']
    if baseline:
        base_time = np.mean([r['end_to_end_seconds'] for r in baseline])
        base_error = np.mean([r['metrics']['mpvpe_mm'] for r in baseline])
        for name in dict.fromkeys(r['method'] for r in rows):
            cells = [r for r in rows if r['method'] == name and r['status'] == 'complete']
            if len(cells) != len(args.seeds):
                report.append(f'| {name} | incomplete | | | | | | |'); continue
            elapsed = np.mean([r['end_to_end_seconds'] for r in cells])
            metric = {k: np.mean([r['metrics'][k] for r in cells]) for k in cells[0]['metrics']}
            payload = np.mean([r.get('saved_tensor_payload_bytes', float('nan'))/2**20 for r in cells])
            report.append(f"| {name} | {elapsed:.2f} | {base_time/elapsed:.2f}x | {metric['mpvpe_mm']:.2f} | "
                          f"{metric['mpvpe_mm']-base_error:+.2f} | {metric['visible_rad']:.3f} | {metric['apd_cm']:.2f} | {payload:.1f} |")
    report += ['', 'Saved payload counts tensor saves (including repeated references); it is not peak memory. '
               'Checkpoint correctness is checked separately by endpoint, penalty, and gradient parity. '
               'Few-step penalty weights preserve approximate time-integrated strength, not an identical objective. '
               'Local guidance uses an endpoint surrogate and changes the solver/guidance algorithm. '
               'These examples support screening only; validate chosen settings on larger development cohorts '
               'and denoising before test evaluation. See manifest.json and results.json for provenance and failures.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n')


if __name__ == '__main__':
    main()
