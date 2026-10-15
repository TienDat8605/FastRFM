"""Resumable controls, adaptation, and sequence-level reporting."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import torch

from .data import load, sha256
from .geometry import exp, log, sample
from .inference import initialize, optimize, rollout
from .runtime import (OfficialBody, completion_metrics, denoising_metrics,
                      load_prior, temporal, validate_rotations, weight_hash)


class BudgetExceeded(RuntimeError):
    pass


class Engine:
    def __init__(self, root, out, config, manifest):
        self.root, self.out = Path(root), Path(out)
        self.config, self.manifest = config, manifest
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config['device'])
        self.body = OfficialBody(self.root, self.device)
        self.original = self.root/'reference/pose_rfm_3dpw.pt'
        self.train = load(self.root, 'train')
        self.mean = exp(torch.tensor(self.train['poses'].mean(0), device=self.device).reshape(21, 3))
        self.started = time.monotonic()
        self.budget_path = self.out/'budget.json'
        self.previous = json.loads(self.budget_path.read_text()).get('elapsed_seconds', 0.) if self.budget_path.exists() else 0.

    def budget(self, estimated_seconds=0.):
        elapsed = self.previous+time.monotonic()-self.started
        self.budget_path.write_text(json.dumps({'elapsed_seconds': elapsed,
            'limit_seconds': self.config['budget_hours']*3600}, indent=2))
        if elapsed+estimated_seconds >= self.config['budget_hours']*3600:
            raise BudgetExceeded('GPU execution budget reached; partial artifacts preserved')

    def generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    def timed(self, fn):
        self.budget()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        value = fn()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        memory = torch.cuda.max_memory_allocated()/2**20 if self.device.type == 'cuda' else None
        return value, {'seconds': elapsed, 'peak_allocated_mib': memory}

    def write_cell(self, path, record, predictions=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if predictions is not None:
            torch.save(predictions, path.with_suffix('.pt'))
            record['predictions'] = str(path.with_suffix('.pt').relative_to(self.out))
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2, default=str))
        tmp.replace(path)

    def evaluate(self, checkpoint, cohort, seeds=None, methods=None, policy=None):
        model, saved = load_prior(self.root, checkpoint, self.device)
        initial_hash = weight_hash(model)
        digest = sha256(checkpoint)
        policy = policy or self.config['solver']
        seeds = self.config['inference_seeds'] if seeds is None else seeds
        methods = methods or ['simple', 'direct', 'initialization', 'dflow', 'no_trajectory', 'no_backward']
        entries = self.manifest['cohorts'][cohort]
        dataset = load(self.root, entries['split'])
        destination = self.out/'cells'/cohort/digest[:12]/policy
        c, d = self.config['completion'], self.config['denoising']
        steps = self.config['ode_steps']
        for task in ('completion', 'denoising'):
            for condition in (c['masks'] if task == 'completion' else range(len(d['sigmas_m']))):
                for seed in seeds:
                    for method in methods:
                        count = len(entries['poses'] if task == 'completion' else entries['segments'])
                        batch = c['batch_poses'] if task == 'completion' else d['batch_segments']
                        for begin in range(0, count, batch):
                            path = destination/f'{task}_{condition}_{seed}_{method}_{begin:04d}.json'
                            if path.exists():
                                continue
                            self.budget()
                            print(f'{cohort} {task} {condition} seed={seed} method={method} batch={begin}', flush=True)
                            record = {'status': 'complete', 'checkpoint_sha256': digest,
                                'checkpoint_epoch': saved.get('epoch'), 'body_sha256': self.body.model_hash,
                                'cohort': cohort, 'task': task, 'condition': str(condition),
                                'seed': seed, 'method': method, 'solver': policy,
                                'ode_steps': steps, 'config': self.config,
                                'reconstruction': ['3DPW residual/time-embedding architecture',
                                    'neutral zero-shape targets', 'SMPL dominant-skinning vertex regions',
                                    'SO(3)-factor scalar Adam second moment', 'loss averages over poses/frames']}
                            # Per-example generators make outcomes invariant to evaluation batching.
                            try:
                                fn = (lambda: self.complete(model, dataset, entries['poses'][begin:begin+batch],
                                          condition, seed, method, policy)) if task == 'completion' else (
                                      lambda: self.denoise(model, dataset, entries['segments'][begin:begin+batch],
                                          condition, seed, method, policy, begin))
                                (rows, predictions), timing = self.timed(fn)
                                for row in rows:
                                    row['seed'] = seed
                                record.update(rows=rows, timing=timing)
                                self.write_cell(path, record, predictions)
                            except BudgetExceeded:
                                raise
                            except (FloatingPointError, RuntimeError) as error:
                                record.update(status='failed', error=repr(error))
                                self.write_cell(path, record)
                                print(f'FAILED CELL: {error}', flush=True)
                            if weight_hash(model) != initial_hash:
                                raise RuntimeError('Frozen model weights changed')
        del model

    def complete(self, model, dataset, ids, mask, seed, method, policy):
        cfg = self.config['completion']
        hypotheses, steps = cfg['hypotheses'], self.config['ode_steps']
        gt = exp(torch.tensor(dataset['poses'][ids], device=self.device).reshape(-1, 21, 3))
        observed = torch.ones(21, dtype=torch.bool, device=self.device)
        observed[cfg['masks'][mask]] = False
        obs = gt.clone()
        obs[:, ~observed] = self.mean[~observed]
        obs = obs.repeat_interleave(hypotheses, 0)
        objective = lambda pred: log(pred.transpose(-1, -2)@obs)[:, observed].norm(dim=-1).sum(-1).mean()
        iterations = cfg['arms_iterations'] if mask == 'arms' else cfg['iterations']
        history = []
        if method in ('simple', 'direct'):
            pred = obs if method == 'simple' else optimize(model, obs, objective, steps, policy,
                                                        iterations, cfg['lr'], direct=True)[0]
        else:
            noise = torch.cat([sample((hypotheses, 21), self.device,
                              generator=self.generator(seed*1000003+int(case))) for case in ids])
            initial = initialize(model, obs, steps, policy, cfg['alpha'],
                                 backward=method != 'no_backward', noise=noise)
            if method == 'initialization':
                with torch.no_grad():
                    pred = rollout(model, initial, steps, policy)[0]
            else:
                pred, history = optimize(model, initial, objective, steps, policy, iterations,
                     cfg['lr'], cfg['trajectory_weight'] if method != 'no_trajectory' else 0.)
        metrics, replaced = completion_metrics(self.body, pred, gt, observed, hypotheses)
        rows = []
        for n, case in enumerate(ids):
            rows.append({'case': int(case), 'sequence': str(dataset['sequence'][case]),
                         'person': int(dataset['person'][case]), 'frame': int(dataset['frame'][case]),
                         **{key: value[n] for key, value in metrics.items()}})
        nfe = 0 if method in ('simple', 'direct') else steps*((0 if method == 'no_backward' else 1)+
               (1 if method == 'initialization' else iterations+1))
        for row in rows:
            row['forward_network_evaluations_per_hypothesis'] = nfe
            row['backpropagated_ode_steps_per_hypothesis'] = steps*iterations if method not in ('simple', 'direct', 'initialization') else 0
        return rows, {'predictions': replaced.cpu(), 'gt': gt.cpu(), 'history': history}

    def denoise(self, model, dataset, ids, condition, seed, method, policy, case):
        cfg = self.config['denoising']
        segments = len(ids)
        frames = cfg['segment_frames']
        flattened = np.asarray(ids).reshape(-1)
        gt = exp(torch.tensor(dataset['poses'][flattened], device=self.device).reshape(-1, 21, 3))
        with torch.no_grad():
            truth = self.body.joints(gt)
            noise = torch.cat([torch.randn(frames, 22, 3, device=self.device,
                      generator=self.generator(seed*1000003+case+i)) for i in range(segments)])
            observed = truth+cfg['sigmas_m'][condition]*noise
        weight = cfg['temporal_weights'][condition]
        def objective(pred):
            joints = self.body.joints(pred)
            return (joints-observed).square().sum((-1, -2)).mean()+weight*temporal(pred, segments)
        if method == 'simple':
            pred_joints, pred, history = observed, None, []
        else:
            # All methods share the same deterministic joint-to-pose fitting.
            fit, _ = optimize(model, self.mean.expand(len(gt), -1, -1, -1), objective,
                            self.config['ode_steps'], policy, cfg['initialization_iterations'],
                            cfg['initialization_lr'], direct=True)
            iterations = cfg['iterations'][condition]
            if method == 'direct':
                pred, history = optimize(model, fit, objective, self.config['ode_steps'], policy,
                                          iterations, cfg['lr'], direct=True)
            else:
                sources = torch.cat([sample((frames, 21), self.device,
                             generator=self.generator(seed*1000003+case+i)) for i in range(segments)])
                initial = initialize(model, fit, self.config['ode_steps'], policy, cfg['alpha'],
                                     backward=method != 'no_backward', noise=sources)
                if method == 'initialization':
                    with torch.no_grad():
                        pred = rollout(model, initial, self.config['ode_steps'], policy)[0]
                    history = []
                else:
                    pred, history = optimize(model, initial, objective, self.config['ode_steps'],
                        policy, iterations, cfg['lr'], cfg['trajectory_weight'] if method != 'no_trajectory' else 0.)
            with torch.no_grad():
                validate_rotations(pred)
                pred_joints = self.body.joints(pred)
        rows = []
        for i, segment_ids in enumerate(ids):
            section = slice(i*frames, (i+1)*frames)
            metrics = denoising_metrics(pred_joints[section], truth[section])
            rows.append({'case': case+i, 'sequence': str(dataset['sequence'][segment_ids[0]]),
               'person': int(dataset['person'][segment_ids[0]]), 'start_frame': int(dataset['frame'][segment_ids[0]]),
               'sigma_m': cfg['sigmas_m'][condition], **metrics})
        return rows, {'rotations': pred.cpu() if pred is not None else None,
                       'noisy_joints': observed.cpu(), 'gt_joints': truth.cpu(), 'history': history}

    def generation(self, checkpoint, cohort):
        target = self.out/'generation'/f'{cohort}_{sha256(checkpoint)[:12]}.json'
        if target.exists():
            return
        model, _ = load_prior(self.root, checkpoint, self.device)
        with torch.no_grad():
            initial = sample((128, 21), self.device, generator=self.generator(42))
            generated = rollout(model, initial, self.config['ode_steps'], self.config['solver'])[0]
            joints = self.body(generated)[0]
            ref_ids = np.random.default_rng(42).choice(len(self.train['poses']), 512, replace=False)
            ref = exp(torch.tensor(self.train['poses'][ref_ids], device=self.device).reshape(-1, 21, 3))
            ref_joints = self.body(ref)[0]
            pairs = torch.triu_indices(len(joints), len(joints), offset=1, device=self.device)
            apd = (joints[pairs[0]]-joints[pairs[1]]).norm(dim=-1).mean()*100
            nearest = []
            for chunk in generated.split(8):
                dist = log(chunk[:, None].transpose(-1, -2)@ref[None]).norm(dim=-1).mean(-1)
                nearest.extend(dist.min(-1).values.cpu().tolist())
        a, b = joints.reshape(128, -1).cpu().numpy(), ref_joints.reshape(512, -1).cpu().numpy()
        cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        # Symmetric PSD construction remains defined with fixed pelvis coordinates.
        eigenvalues, vectors = np.linalg.eigh(cov_a)
        square_root = (vectors*np.sqrt(eigenvalues.clip(min=0)))@vectors.T
        sandwich = square_root@cov_b@square_root
        root_trace = np.sqrt(np.linalg.eigvalsh((sandwich+sandwich.T)/2).clip(min=0)).sum()
        fid = float(np.square(a.mean(0)-b.mean(0)).sum()+np.trace(cov_a+cov_b)-2*root_trace)
        self.write_cell(target, {'scope': '128 samples, 512 train references; diagnostic, not paper Table 1',
            'checkpoint_sha256': sha256(checkpoint), 'apd_cm': apd.item(), 'dnn_mean_rotation_rad': float(np.mean(nearest)),
            'joint_position_fid': fid})

    def trigger_finetuning(self):
        records = [r for r in read_records(self.out, 'development', sha256(self.original))
                   if r['solver'] == self.config['solver']]
        decision = {'status': 'incomplete', 'trigger': False, 'tasks': {}}
        expected = self.manifest['cohorts']['development']
        for task, conditions, cases in [('completion', list(self.config['completion']['masks']), len(expected['poses'])),
                                        ('denoising', ['0', '1'], len(expected['segments']))]:
            for condition in conditions:
                for seed in self.config['inference_seeds']:
                    for method in ('simple', 'direct', 'dflow'):
                        cells = [r for r in records if r['task'] == task and r['condition'] == str(condition)
                                 and r['seed'] == seed and r['method'] == method and r['solver'] == self.config['solver']]
                        if any(r['status'] != 'complete' for r in cells) or sum(len(r.get('rows', [])) for r in cells) != cases:
                            self.write_cell(self.out/'adaptation_decision.json', decision)
                            return decision
        for task, metric in [('completion', 'mpvpe_mm'), ('denoising', 'mpjpe_mm')]:
            methods = {}
            for record in records:
                if record['task'] == task and record['status'] == 'complete':
                    methods.setdefault(record['method'], []).extend(row[metric] for row in record['rows'])
            if not all(methods.get(k) for k in ('simple', 'direct', 'dflow')):
                continue
            control = min(np.mean(methods['simple']), np.mean(methods['direct']))
            gain = 1-np.mean(methods['dflow'])/max(control, 1e-12)
            decision['tasks'][task] = {'control_error': float(control), 'dflow_error': float(np.mean(methods['dflow'])),
                                      'relative_gain': float(gain)}
        if len(decision['tasks']) == 2 and not any(r['status'] != 'complete' for r in records):
            decision['status'] = 'complete'
            decision['trigger'] = any(v['relative_gain'] < self.config['finetuning']['trigger_relative_gain']
                                      for v in decision['tasks'].values())
        self.write_cell(self.out/'adaptation_decision.json', decision)
        return decision

    def estimate(self):
        records = [r for r in read_records(self.out, 'benchmark', sha256(self.original))
                   if r['method'] == 'dflow' and r['status'] == 'complete']
        times = {task: sum(r['timing']['seconds'] for r in records if r['task'] == task)
                 for task in ('completion', 'denoising')}
        if not all(times.values()):
            raise RuntimeError('Run the representative-batch benchmark first')
        import math
        batch_c = self.config['completion']['batch_poses']
        batch_d = self.config['denoising']['batch_segments']
        completion_scale = batch_c/min(batch_c, self.config['benchmark']['poses'])
        estimates = {}
        for cohort in ('development', 'final'):
            c_batches = math.ceil(self.config[cohort]['poses']/batch_c)
            d_batches = math.ceil(self.config[cohort]['segments']/batch_d)
            # Three expensive guidance variants plus 10% allowance for controls.
            estimates[cohort+'_hours'] = ((times['completion']*completion_scale*c_batches+
                times['denoising']*d_batches)*len(self.config['inference_seeds'])*3.1)/3600
        estimates.update(scope='Measured batch-size benchmark extrapolation; fine-tuning and adapted-prior evaluations excluded',
                         benchmark_seconds=times, total_original_prior_hours=sum(estimates.values()))
        self.write_cell(self.out/'resource_estimate.json', estimates)
        return estimates

    def finetune(self, seed):
        cfg = self.config['finetuning']
        destination = self.out/'finetuning'/f'seed_{seed}'
        destination.mkdir(parents=True, exist_ok=True)
        best_path, last_path = destination/'best.pt', destination/'last.pt'
        model, saved = load_prior(self.root, self.original, self.device)
        saved['adaptation'] = {'config': cfg, 'original_sha256': sha256(self.original), 'seed': seed}
        model.requires_grad_(True).train()
        ema = deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        train = torch.tensor(self.train['poses'], device=self.device).reshape(-1, 21, 3)
        val = load(self.root, 'validation')['poses']
        val_ids = np.random.default_rng(42).choice(len(val), cfg['validation_poses'], replace=False)
        val_rot = exp(torch.tensor(val[val_ids], device=self.device).reshape(-1, 21, 3))
        source = sample(val_rot.shape[:2], self.device, generator=self.generator(7142))
        t_val = torch.rand(len(source), device=self.device, generator=self.generator(8142))
        target = log(source.transpose(-1, -2)@val_rot)
        interpolant = source@exp(t_val[:, None, None]*target)
        def validation(candidate):
            with torch.no_grad():
                return (candidate(interpolant, t_val)-target).square().mean().item()
        best_loss, step, history = validation(ema), 0, []
        torch.manual_seed(seed)
        if last_path.exists():
            resume = torch.load(last_path, map_location=self.device, weights_only=False)
            model.load_state_dict(resume['raw_model'])
            ema.load_state_dict(resume['model_state_dict'])
            optimizer.load_state_dict(resume['optimizer'])
            best_loss, step, history = resume['best_val_loss'], resume['step'], resume['history']
            torch.set_rng_state(resume['rng_cpu'].cpu())
            if self.device.type == 'cuda':
                torch.cuda.set_rng_state(resume['rng_cuda'].cpu())
        elif not best_path.exists():
            torch.save({**saved, 'model_state_dict': ema.state_dict(), 'adaptation_seed': seed,
                         'step': 0, 'val_loss': best_loss}, best_path)
        def save_last():
            torch.save({**saved, 'raw_model': model.state_dict(), 'model_state_dict': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'best_val_loss': best_loss,
                'history': history, 'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state() if self.device.type == 'cuda' else None}, last_path)
        try:
            while step < cfg['steps']:
                self.budget()
                ids = torch.randint(len(train), (cfg['batch_size'],), device=self.device)
                clean = exp(train[ids])
                source = sample(clean.shape[:2], self.device)
                t = torch.rand(len(clean), device=self.device)
                velocity = log(source.transpose(-1, -2)@clean)
                state = source@exp(t[:, None, None]*velocity)
                loss = (model(state, t)-velocity).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Fine-tuning diverged')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 1-cfg['ema'])
                step += 1
                if step % cfg['validation_interval'] == 0:
                    value = validation(ema)
                    history.append({'step': step, 'train_loss': loss.item(), 'val_loss': value})
                    if value < best_loss:
                        best_loss = value
                        torch.save({**saved, 'model_state_dict': ema.state_dict(),
                            'adaptation_seed': seed, 'step': step, 'val_loss': value}, best_path)
                    save_last()
                    print(f'Fine-tuning seed={seed} step={step} val={value:.6f}', flush=True)
        finally:
            save_last()
            self.write_cell(destination/'training.json', {'seed': seed, 'step': step, 'best_val_loss': best_loss,
                'original_sha256': sha256(self.original), 'config': cfg, 'history': history})
        return best_path


def read_records(out, cohort=None, checkpoint=None):
    result = []
    for path in (Path(out)/'cells').rglob('*.json'):
        record = json.loads(path.read_text())
        if (cohort is None or record['cohort'] == cohort) and (checkpoint is None or record['checkpoint_sha256'] == checkpoint):
            result.append(record)
    return result


def report(out, config):
    out = Path(out)
    records = read_records(out)
    groups, failures = {}, []
    for record in records:
        if record['status'] != 'complete':
            failures.append(record)
            continue
        key = '|'.join(record[k] for k in ('cohort', 'checkpoint_sha256', 'solver', 'task', 'condition', 'method'))
        entry = groups.setdefault(key, {'rows': [], 'seconds': [], 'peak_mib': []})
        entry['rows'].extend(record['rows'])
        entry['seconds'].append(record['timing']['seconds'])
        entry['peak_mib'].append(record['timing']['peak_allocated_mib'] or 0)
    summary = {}
    for key, entry in groups.items():
        task = key.split('|')[3]
        metrics = ('mpvpe_mm', 'apd_cm', 'visible_rad') if task == 'completion' else ('mpjpe_mm', 'acceleration_error_mm_s2')
        result = {'rows': len(entry['rows']), 'cases': len({r['case'] for r in entry['rows']}),
                  'seconds': sum(entry['seconds']), 'peak_allocated_mib': max(entry['peak_mib'])}
        for metric in metrics:
            values = {}
            for row in entry['rows']:
                values.setdefault(row['sequence'], []).append(row[metric])
            means = np.array([np.mean(v) for v in values.values()])
            rng = np.random.default_rng(42)
            boot = means[rng.integers(len(means), size=(config['bootstrap_samples'], len(means)))].mean(-1)
            result[metric] = {'sequence_mean': float(means.mean()),
                'ci95_sequence_bootstrap': np.quantile(boot, [.025, .975]).tolist()}
        summary[key] = result
    paired = {}
    audit_path = out/'audit.json'
    original_digest = json.loads(audit_path.read_text())['checkpoint_sha256'] if audit_path.exists() else None
    for key, entry in groups.items():
        if key.split('|')[-1] not in ('dflow', 'no_trajectory', 'no_backward'):
            continue
        metric = 'mpvpe_mm' if key.split('|')[3] == 'completion' else 'mpjpe_mm'
        comparisons = [(control, '|'.join(key.split('|')[:-1]+[control]))
                       for control in ('simple', 'direct', 'initialization')]
        if key.split('|')[-1] in ('no_trajectory', 'no_backward'):
            comparisons.append(('full_dflow', '|'.join(key.split('|')[:-1]+['dflow'])))
        if original_digest and key.split('|')[1] != original_digest and key.split('|')[-1] == 'dflow':
            target = key.split('|')
            target[1] = original_digest
            comparisons.append(('original_prior', '|'.join(target)))
        for control, target_key in comparisons:
            if target_key not in groups:
                continue
            targets = {(r['case'], r['seed']): r for r in groups[target_key]['rows']}
            differences = {}
            for row in entry['rows']:
                identity = (row['case'], row['seed'])
                if identity in targets:
                    differences.setdefault(row['sequence'], []).append(row[metric]-targets[identity][metric])
            means = np.array([np.mean(v) for v in differences.values()])
            if not len(means):
                continue
            rng = np.random.default_rng(42)
            boot = means[rng.integers(len(means), size=(config['bootstrap_samples'], len(means)))].mean(-1)
            paired[key+' minus '+control] = {'mean_difference_mm': float(means.mean()),
                'ci95': np.quantile(boot, [.025, .975]).tolist(), 'sequences': len(means)}
    payload = {'scope': 'Independent 3DPW transfer study; no paper-table reproduction claim',
        'coverage': {cohort: {'completed_rows': sum(len(r.get('rows', [])) for r in records if r['cohort'] == cohort and r['status'] == 'complete'),
                              'failed_cells': sum(r['status'] != 'complete' for r in records if r['cohort'] == cohort)}
                     for cohort in ('smoke', 'development', 'final')},
        'summary': summary, 'paired_comparisons': paired, 'failed_cells': failures}
    (out/'summary.json').write_text(json.dumps(payload, indent=2))
    lines = ['# PoseD-style 3DPW study', '', payload['scope'], '',
             'Negative paired error differences favor guidance. Intervals resample source sequences.', '',
             f'Completed metric groups: {len(summary)}. Failed cells: {len(failures)}.', '',
             '| Cohort / task / condition / method | Checkpoint | Error (mm), sequence mean |',
             '|---|---|---|']
    for key, values in summary.items():
        cohort, checkpoint, policy, task, condition, method = key.split('|')
        metric = 'mpvpe_mm' if task == 'completion' else 'mpjpe_mm'
        lines.append(f'| {cohort} / {task} / {condition} / {method} ({policy}) | {checkpoint[:12]} | {values[metric]["sequence_mean"]:.3f} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return payload
