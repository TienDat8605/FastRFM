"""Real-data GPU PoC: midpoint shortcut students and constant-storage adjoints."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from src.study.data import balanced, sha256
from src.study.geometry import exp, log, sample
from src.study.runtime import OfficialBody, completion_metrics, validate_rotations
from src.study.acceleration import optimize_with_rollout
from .adjoint import integrate, adjoint_integrate
from .distillation import endpoint_error


class BodyField(torch.nn.Module):
    """Adapt the PoC's ambient tangent output to body-vector coordinates."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, state, t):
        from posed_flow_colab_poc import vee
        return vee(state.transpose(-1, -2)@self.network(state, t))


def poses(path, device):
    with np.load(path, allow_pickle=False) as archive:
        if 'poses' not in archive: raise ValueError(f'Explicit poses array missing: {path}')
        data = {key: archive[key] for key in archive.files}
    array = data['poses']
    if array.ndim != 2 or array.shape[1] not in (63, 72, 156):
        raise ValueError('Expected body-63, SMPL-72, or SMPL-H-156 axis angles')
    aa = array if array.shape[1] == 63 else array[:, 3:66]
    if not np.isfinite(aa).all(): raise ValueError('Nonfinite pose parameters')
    keep = np.ones(len(aa), dtype=bool)
    if 'sequence' in data:
        keep = np.array([not str(s).startswith('mock_') for s in data['sequence']])
    aa = aa[keep]
    data = {k: v[keep] for k, v in data.items() if v.ndim > 0 and len(v) == len(keep)}
    data['poses'] = aa
    return exp(torch.tensor(aa, device=device).reshape(-1, 21, 3)), data


def run(args):
    from posed_flow_colab_poc import train_flow_model, PoseFlowMLP
    if args.num_joints != 21 or not args.data_path or not args.validation_path:
        raise ValueError('Inference mode requires 21 joints and separate train/validation NPZ paths')
    if not torch.cuda.is_available(): raise RuntimeError('This benchmark requires a CUDA GPU')
    device = torch.device('cuda')
    asset_candidates = ([Path(args.model_root)] if args.model_root else
                        list(Path(args.data_path).resolve().parents))
    asset_root = next((p for p in asset_candidates if (p/'models/smpl/SMPL_NEUTRAL.pkl').is_file()), None)
    if asset_root is None:
        raise ValueError('Official neutral SMPL missing; set --model_root')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Use a fresh output directory')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    train, train_data = poses(args.data_path, device)
    validation, validation_data = poses(args.validation_path, device)
    if 'sequence' not in train_data or 'sequence' not in validation_data:
        raise ValueError('Sequence provenance is required for disjoint-split checks')
    if set(train_data['sequence']) & set(validation_data['sequence']):
        raise ValueError('Train/validation sequences overlap')
    groups = {str(s): np.flatnonzero((validation_data['sequence'] == s)&
                         (validation_data['frame'] % 10 == 0)).tolist()
              for s in np.unique(validation_data['sequence'])}
    ids = balanced(groups, 8, np.random.default_rng(42))
    manifest = {'date': '2026-10-14', 'stage': 'exploratory GPU inference PoC',
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'seed': 42,
        'train_sha256': sha256(args.data_path), 'validation_sha256': sha256(args.validation_path),
        'train_count': len(train), 'validation_count': len(validation), 'validation_ids': ids,
        'validation_sequences': [str(validation_data['sequence'][i]) for i in ids],
        'prior_training_updates': args.steps or 1000, 'prior_batch_size': 256,
        'prior_architecture': 'Four 512-unit SiLU hidden layers; 190 inputs; 63 body tangent outputs',
        'distillation_updates': args.distill_steps, 'distillation_train_sources': 512,
        'distillation_selection_sources': 128, 'optimizer_iterations': args.opt_iters,
        'inference_seeds': [0, 1], 'repeats': args.repeats,
        'masks': {'legs': [0,1,3,4,6,7,9,10], 'torso': [2,5,8,15,16]},
        'trajectory_cost': 'Post-step states sum; lambda_N=.001*100/N',
        'adjoint': 'Discrete local VJPs with running-cost forcing; fixed-point reverse reconstruction; '
                   'prefix replay fallback; tolerance-limited, not OC-Flow',
        'source_initialization': 'Shared Haar sources, paired by example/hypothesis and seed',
        'caveats': ['1000-update small-data prior is not full-paper training',
                    'Midpoint endpoint distillation is not RCM continuous-time training',
                    'Student and teacher running costs differ with discretization',
                    'Eight validation poses and one training seed; no test claims',
                    'Batch time divided by batch size is throughput, not single-pose latency']}
    manifest['source_sha256'] = {str(path): sha256(path) for path in
        [Path('posed_flow_colab_poc.py'), Path(__file__), Path(__file__).with_name('adjoint.py')]}
    reuse = Path(args.reuse_training_dir) if args.reuse_training_dir else None
    old_training = None
    if reuse:
        old_manifest = json.loads((reuse/'manifest.json').read_text())
        for key in ('train_sha256', 'validation_sha256', 'prior_training_updates', 'prior_batch_size',
                    'prior_architecture', 'distillation_updates'):
            if manifest[key] != old_manifest[key]:
                raise ValueError(f'Incompatible training reuse: {key}')
        old_training = json.loads((reuse/'training.json').read_text())
        manifest['reused_training_dir'] = str(reuse)
        manifest['reused_teacher_sha256'] = sha256(reuse/'teacher.pt')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'state.json').write_text(json.dumps({'status': 'training'}))
    print('Training tiny PoseRFM on real 3DPW train poses', flush=True)
    torch.cuda.synchronize(); start = time.perf_counter()
    if reuse:
        network = PoseFlowMLP(21, hidden_dim=512, riemannian=True).to(device)
        checkpoint = torch.load(reuse/'teacher.pt', map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['state_dict'])
        network.eval().requires_grad_(False)
        print('Reused prior:', reuse/'teacher.pt', flush=True)
    else:
        network = train_flow_model(train, 21, steps=args.steps or 1000, batch_size=256,
                                   hidden_dim=512, riemannian=True).eval().requires_grad_(False)
    teacher = BodyField(network).eval().requires_grad_(False)
    torch.cuda.synchronize(); prior_seconds = old_training['prior_seconds'] if reuse else time.perf_counter()-start
    torch.save({'state_dict': network.state_dict(), 'manifest': manifest}, out/'teacher.pt')
    def teacher_map(x, trajectory=True): return integrate(teacher, x, 100, 'euler', trajectory)
    print('Generating paired teacher endpoints for midpoint shortcut distillation', flush=True)
    torch.cuda.synchronize(); start = time.perf_counter()
    mean = exp(torch.tensor(train_data['poses'].mean(0), device=device).reshape(21, 3))
    if reuse:
        pairs = torch.load(reuse/'distillation_pairs.pt', map_location=device, weights_only=False)
        sources, targets = pairs['sources'], pairs['targets']
    else:
        generator = torch.Generator(device=device).manual_seed(43)
        sources = sample((640, 21), device, generator=generator)
        # Student data use only the training cohort; no validation poses enter training.
        with torch.no_grad():
            clean = train[torch.randperm(len(train), device=device)[:640]]
            partial = clean.clone()
            partial[:, manifest['masks']['legs']] = mean[manifest['masks']['legs']]
            sources[1::2] = exp(.4*log(partial[1::2])+.6*log(sources[1::2]))
            targets = torch.cat([teacher_map(chunk, False)[0] for chunk in sources.split(64)])
    torch.cuda.synchronize(); pair_seconds = old_training['pair_seconds'] if reuse else time.perf_counter()-start
    torch.save({'sources': sources.cpu(), 'targets': targets.cpu()}, out/'distillation_pairs.pt')
    students, training = {}, {'prior_seconds': prior_seconds, 'pair_seconds': pair_seconds}
    if reuse: training['training_reused'] = True
    for n in (2, 4):
        if reuse:
            student = deepcopy(teacher)
            checkpoint = torch.load(reuse/f'student{n}.pt', map_location=device, weights_only=False)
            student.load_state_dict(checkpoint['state_dict'])
            students[n] = student.eval().requires_grad_(False)
            training[f'student{n}'] = old_training[f'student{n}']
            shutil.copy2(reuse/f'student{n}.pt', out/f'student{n}.pt')
            continue
        student = deepcopy(teacher).requires_grad_(True)
        optimizer = torch.optim.AdamW(student.parameters(), lr=.0001, weight_decay=.0001)
        rng = torch.Generator(device=device).manual_seed(44)
        with torch.no_grad(): best = endpoint_error(integrate(student, sources[512:], n, 'midpoint', False)[0], targets[512:]).item()
        initial_loss = best
        weights, selected = deepcopy(student.state_dict()), 0
        history = []
        torch.cuda.synchronize(); start = time.perf_counter()
        for update in range(1, args.distill_steps+1):
            ix = torch.randint(512, (32,), device=device, generator=rng)
            prediction = integrate(student, sources[ix], n, 'midpoint', False)[0]
            loss = endpoint_error(prediction, targets[ix])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite shortcut loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            if update % 100 == 0 or update == args.distill_steps:
                with torch.no_grad(): value = endpoint_error(integrate(student, sources[512:], n, 'midpoint', False)[0], targets[512:]).item()
                history.append({'update': update, 'selection_endpoint_loss': value})
                print(f'midpoint student N={n} update={update} selection_loss={value:.6f}', flush=True)
                if value < best: best, weights, selected = value, deepcopy(student.state_dict()), update
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        student.load_state_dict(weights)
        student.eval().requires_grad_(False)
        student.zero_grad(set_to_none=True)
        del optimizer
        students[n] = student
        training[f'student{n}'] = {'seconds': elapsed, 'initial_selection_loss': initial_loss,
                                  'selected_loss': best, 'selected_update': selected, 'history': history}
        torch.save({'state_dict': weights, 'steps': n, 'solver': 'midpoint', 'selected_update': selected}, out/f'student{n}.pt')
        (out/'training.json').write_text(json.dumps(training, indent=2))
    (out/'training.json').write_text(json.dumps(training, indent=2))
    body = OfficialBody(asset_root, device)
    gt = validation[ids]
    memory_rows = []
    # A fixed batch and identical loss isolate N-dependent backward storage.
    probe = gt[:2]
    for n in (25, 50, 100, 200):
        reference_gradient = None
        for method in ('full_graph', 'adjoint'):
            torch.cuda.empty_cache()
            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stats = {}
            delta = probe.new_zeros(2, 21, 3).requires_grad_()
            torch.cuda.synchronize(); start = time.perf_counter()
            x = probe@exp(delta)
            pred, penalty = (adjoint_integrate(teacher, x, n, stats=stats, tolerance=2e-7)
                             if method == 'adjoint' else integrate(teacher, x, n))
            loss = pred[..., 0, 1].sum()+(.001*100/n)*penalty.mean()
            gradient, = torch.autograd.grad(loss, delta)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            if method == 'full_graph': reference_gradient = gradient.detach()
            relative = ((gradient-reference_gradient).norm()/reference_gradient.norm().clamp_min(1e-8)).item()
            cosine = F.cosine_similarity(gradient.flatten()[None], reference_gradient.flatten()[None]).item()
            cell = {'steps': n, 'method': method, 'seconds': elapsed,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'incremental_peak_mib': (torch.cuda.max_memory_allocated()-baseline_allocated)/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                'gradient_relative_error': relative, 'gradient_cosine': cosine, 'reconstruction': stats}
            memory_rows.append(cell)
            print('GRADIENT/MEMORY', json.dumps(cell), flush=True)
            (out/'memory_gradient.json').write_text(json.dumps(memory_rows, indent=2))
            if method == 'adjoint' and (relative > .01 or cosine < .999):
                raise RuntimeError(f'Adjoint gradient gate failed at N={n}: {cell}')
            del delta, pred, penalty, gradient, loss, x
    (out/'state.json').write_text(json.dumps({'status': 'evaluating'}))
    results = []
    for mask, missing in manifest['masks'].items():
        visible = torch.ones(21, dtype=torch.bool, device=device)
        visible[missing] = False
        obs = gt.clone(); obs[:, ~visible] = mean[~visible]
        obs = obs.repeat_interleave(2, 0)
        objective = lambda r: log(r.transpose(-1, -2)@obs)[:, visible].norm(dim=-1).sum(-1).mean()
        for seed in (0, 1):
            initial = torch.cat([sample((2, 21), device,
                    generator=torch.Generator(device=device).manual_seed(seed*1000003+int(case))) for case in ids])
            candidates = [('euler100', teacher, 100, 'euler', False),
                ('euler50', teacher, 50, 'euler', False),
                ('adjoint100', teacher, 100, 'euler', True),
                ('adjoint50', teacher, 50, 'euler', True),
                ('midpoint2', teacher, 2, 'midpoint', False),
                ('midpoint4', teacher, 4, 'midpoint', False),
                ('student2', students[2], 2, 'midpoint', False),
                ('student4', students[4], 4, 'midpoint', False)]
            if seed % 2: candidates.reverse()
            for name, model, n, solver, adjoint in candidates:
                stats = {}
                def solver_fn(x):
                    return (adjoint_integrate(model, x, n, solver, stats=stats, tolerance=2e-7)
                            if adjoint else integrate(model, x, n, solver))
                measured = []
                # One optimization iteration warms this method's forward/backward kernels.
                optimize_with_rollout(initial, objective, solver_fn, 1, .1, .001*100/n)
                for repeat in range(args.repeats):
                    torch.cuda.empty_cache()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); start = time.perf_counter()
                    prediction, history = optimize_with_rollout(initial, objective, solver_fn,
                                                        args.opt_iters, .1, .001*100/n)
                    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                    measured.append({'seconds': elapsed, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                        'incremental_peak_mib': (torch.cuda.max_memory_allocated()-base)/2**20,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
                validate_rotations(prediction)
                metrics, replaced = completion_metrics(body, prediction, gt, visible, 2)
                row = {'method': name, 'mask': mask, 'seed': seed, 'steps': n, 'solver': solver,
                    'status': 'complete', 'iterations': args.opt_iters, 'timings': measured,
                    'metrics': {k: float(np.mean(v)) for k, v in metrics.items()}, 'per_case': metrics,
                    'reconstruction': stats, 'forward_nfe_per_rollout': n*(2 if solver == 'midpoint' else 1)}
                results.append(row)
                torch.save({'prediction': replaced.cpu(), 'gt': gt.cpu(), 'history': history}, out/f'{mask}_{seed}_{name}.pt')
                (out/'results.json').write_text(json.dumps(results, indent=2))
                print('CELL', json.dumps({k:v for k,v in row.items() if k != 'per_case'}), flush=True)
    lines = ['# Colab GPU inference improvement PoC', '', '## Material Passport', '',
        'Mode: run; date: 2026-10-14; stage: exploratory; outputs: this report, manifest, '
        'checkpoints, predictions, timing and gradient/memory JSON records.', '',
        f"GPU: {manifest['device']}; real train={len(train)}, validation={len(validation)}; "
        f'prior trained for {args.steps or 1000} updates. Eight validation poses, two hypotheses, '
        f'two masks, two inference seeds, {args.opt_iters} optimization iterations, '
        f'{args.repeats} timing repetitions. No test data used.', '',
        '| Method | Solver s/batch | Speedup | Peak allocated MiB | Hidden MPVPE mm | Visible rad | APD cm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    base_time = np.mean([t['seconds'] for r in results if r['method']=='euler100' for t in r['timings']])
    for name in ['euler100','euler50','adjoint100','adjoint50','midpoint2','midpoint4','student2','student4']:
        cells = [r for r in results if r['method']==name]
        elapsed = np.mean([t['seconds'] for r in cells for t in r['timings']])
        memory = max(t['peak_allocated_mib'] for r in cells for t in r['timings'])
        metric = {k: np.mean([r['metrics'][k] for r in cells]) for k in cells[0]['metrics']}
        lines.append(f"| {name} | {elapsed:.3f} | {base_time/elapsed:.2f}x | {memory:.1f} | "
                     f"{metric['mpvpe_mm']:.2f} | {metric['visible_rad']:.3f} | {metric['apd_cm']:.2f} |")
    lines += ['', '## Memory and gradient audit', '',
        '| Steps | Backend | Backward-probe s | Incremental peak MiB | Gradient relative error |',
        '|---|---|---:|---:|---:|']
    for r in memory_rows:
        lines.append(f"| {r['steps']} | {r['method']} | {r['seconds']:.3f} | {r['incremental_peak_mib']:.2f} | {r['gradient_relative_error']:.6f} |")
    lines += ['', 'The adjoint stores at most five fixed anchors and one local graph; O(1) is '
        'with respect to ODE depth for fixed model and batch. It reconstructs prior states '
        'to a tolerance and uses prefix replay on failed inverse residual checks. '
        'Inverse/replay work can make it slower; see reconstruction counters. It includes '
        'the running-cost adjoint forcing. It is not OC-Flow or an exact continuous adjoint.', '',
        'Student midpoint steps require two network evaluations each. Endpoint distillation '
        'does not preserve teacher guidance derivatives or covariance guarantees. Short prior '
        'training, small validation samples, changing trajectory discretization, and absence '
        'of a learned plausibility metric limit interpretation. Solver times exclude body '
        'metrics; peak memory includes resident models and the official body. '
        'Training/generation costs are recorded in training.json. The distillation selection '
        'cohort is not independent generalization evidence.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'state.json').write_text(json.dumps({'status':'complete_exploratory', 'cells':len(results)}, indent=2))
    print('COMPLETE:', out, flush=True)
