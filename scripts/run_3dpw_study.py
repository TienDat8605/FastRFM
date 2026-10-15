"""Entry point for audited evaluation, fine-tuning, and reporting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from copy import deepcopy
from importlib.metadata import version

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.study.data import prepare, sha256
from src.study.engine import BudgetExceeded, Engine, read_records, report
from src.study.geometry import exp, log
from src.study.runtime import load_prior, weight_hash


def audit(engine):
    # Run the actual numerical checks in the same environment as the study.
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_study_*.py', '-v'],
        cwd=Path(__file__).resolve().parents[1], check=True)
    model, saved = load_prior(engine.root, engine.original, engine.device)
    initial = weight_hash(model)
    z = torch.zeros(2, 21, 3, device=engine.device, requires_grad=True)
    joints, vertices = engine.body(exp(z))
    grad, = torch.autograd.grad(vertices.square().mean()+joints.square().mean(), z)
    assert torch.isfinite(grad).all() and torch.isfinite(vertices).all()
    torch.manual_seed(42)
    varied = torch.randn(4, 21, 3, device=engine.device)*.3
    delta = torch.zeros_like(varied, requires_grad=True)
    rotations = exp(varied)@exp(delta)
    official = engine.body(rotations)[0]
    efficient = engine.body.joints(rotations)
    torch.testing.assert_close(official, efficient, atol=2e-5, rtol=2e-5)
    slow_grad, = torch.autograd.grad(official.square().sum(), delta, retain_graph=True)
    fast_grad, = torch.autograd.grad(efficient.square().sum(), delta)
    torch.testing.assert_close(slow_grad, fast_grad, atol=2e-4, rtol=2e-4)
    v = torch.full_like(z, .01)
    state = exp(v)
    assert torch.isfinite(model(state, torch.full((2,), .5, device=engine.device))).all()
    # Exercise the fine-tuning path on a disposable copy without changing the prior.
    candidate = deepcopy(model).requires_grad_(True)
    opt = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
    clean = exp(torch.tensor(engine.train['poses'][:32], device=engine.device).reshape(-1, 21, 3))
    from src.study.geometry import sample
    source = sample(clean.shape[:2], engine.device, generator=engine.generator(42))
    velocity = log(source.transpose(-1, -2)@clean)
    t = torch.full((len(clean),), .5, device=engine.device)
    loss = (candidate(source@exp(.5*velocity), t)-velocity).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
    opt.step()
    assert weight_hash(candidate) != initial and weight_hash(model) == initial
    del candidate, opt
    record = {'status': 'passed', 'checkpoint_sha256': sha256(engine.original),
        'weights_sha256': initial, 'checkpoint_args': saved['args'], 'epoch': saved['epoch'],
        'model_source_sha256': sha256(engine.root/'reference/hmr_pose_rfm.py'),
        'body_sha256': engine.body.model_hash,
        'environment': {'python': sys.version, 'torch': torch.__version__,
                        'numpy': version('numpy'), 'smplx': version('smplx'),
                        'gpu': torch.cuda.get_device_name() if engine.device.type == 'cuda' else 'CPU'},
        'body_vertices': list(vertices.shape), 'body_joints': list(joints.shape),
        'finite_body_gradient': True,
        'joint_only_forward_and_gradient_agreement': True,
        'disposable_copy_finetuning_step': 'passed; original weights unchanged',
        'source': 'Haar uniform, corroborated by HMR training script',
        'target': 'positive Log(x0^T x1) along x0 Exp(t Log(x0^T x1))',
        'provenance_limit': 'Checkpoint args and training-script inspection; original data checksum was not saved.',
        'prior_test_exposure': engine.manifest['prior_test_exposure'],
        'reconstruction': 'Per-joint scalar second moments; intrinsic momentum parallel transport; explicit loss normalization.'}
    engine.write_cell(engine.out/'audit.json', record)
    print('AUDIT PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'audit', 'smoke', 'benchmark', 'evaluate', 'finetune', 'report', 'all'])
    parser.add_argument('--root', type=Path, default=Path('/content/posedflow/study'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cohort', choices=['development', 'final'], default='development')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    config_path = args.config or args.root/'configs/study_3dpw.yaml'
    config = yaml.safe_load(config_path.read_text())
    out = args.output or args.root/'output/3dpw_study_v4'
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'config.json'
    encoded = json.dumps(config, indent=2, sort_keys=True)
    if snapshot.exists() and snapshot.read_text() != encoded:
        raise ValueError('Config changed; use a new output directory')
    snapshot.write_text(encoded)
    source_root = Path(__file__).resolve().parents[1]
    implementation = {str(p.relative_to(source_root)): sha256(p)
                      for p in sorted((source_root/'src/study').glob('*.py'))}
    implementation['scripts/run_3dpw_study.py'] = sha256(Path(__file__).resolve())
    implementation_path = out/'implementation.json'
    implementation_text = json.dumps(implementation, sort_keys=True, indent=2)
    if implementation_path.exists() and implementation_path.read_text() != implementation_text:
        raise ValueError('Implementation changed; use a new output directory rather than reuse old cells')
    implementation_path.write_text(implementation_text)
    manifest = prepare(args.root, config, out)
    if args.command == 'prepare':
        print(json.dumps({k: {"split": v['split'], "poses": len(v['poses']), "segments": len(v['segments'])}
                          for k, v in manifest['cohorts'].items()}, indent=2))
        return
    if args.command == 'report':
        report(out, config)
        return
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    engine = Engine(args.root, out, config, manifest)
    checkpoint = args.checkpoint or engine.original
    state = {'status': 'running', 'command': args.command, 'started_utc': time.time()}
    engine.write_cell(out/'state.json', state)
    try:
        if args.command in ('audit', 'smoke', 'benchmark', 'all'):
            audit(engine)
        elif not (out/'audit.json').exists():
            raise RuntimeError('Run audit first')
        if args.command in ('smoke', 'all'):
            engine.evaluate(checkpoint, 'smoke', seeds=[0], methods=['simple', 'direct', 'initialization', 'dflow'])
        if args.command in ('benchmark', 'all'):
            engine.evaluate(checkpoint, 'benchmark', seeds=[0], methods=['dflow'])
            estimate = engine.estimate()
            print(json.dumps(estimate, indent=2), flush=True)
            if args.command == 'all':
                remaining = config['budget_hours']-(engine.previous+time.monotonic()-engine.started)/3600
                if estimate['total_original_prior_hours'] > remaining:
                    raise BudgetExceeded('Estimated original-prior matrix exceeds remaining GPU budget; no final cohort unblinded')
        if args.command in ('evaluate', 'all'):
            engine.evaluate(checkpoint, args.cohort if args.command == 'evaluate' else 'development')
            engine.generation(checkpoint, args.cohort if args.command == 'evaluate' else 'development')
        if args.command in ('finetune', 'all'):
            decision = engine.trigger_finetuning()
            if decision['status'] != 'complete':
                raise RuntimeError('Development controls incomplete or failed; diagnose before fine-tuning')
            adapted = []
            if decision['trigger']:
                for seed in config['finetuning']['seeds']:
                    tuned = engine.finetune(seed)
                    adapted.append(tuned)
                    engine.evaluate(tuned, 'development', seeds=[0], methods=['dflow'])
                    engine.generation(tuned, 'development')
            if args.command == 'all':
                selected = {'original': str(engine.original), 'adapted': [str(p) for p in adapted],
                    'selection': 'Fixed validation RFM checkpoint selection; guidance configuration unchanged.',
                    'manifest_sha256': sha256(out/'manifest.json'), 'config_sha256': sha256(out/'config.json')}
                engine.write_cell(out/'frozen_final_protocol.json', selected)
                engine.evaluate(engine.original, 'final')
                engine.generation(engine.original, 'final')
                for tuned in adapted:
                    engine.evaluate(tuned, 'final', methods=['initialization', 'dflow'])
                    engine.generation(tuned, 'final')
        cohort = args.command if args.command in ('smoke', 'benchmark') else args.cohort
        records = read_records(out, None if args.command == 'all' else cohort)
        failed = sum(r['status'] != 'complete' for r in records) if args.command != 'audit' else 0
        state.update(status='completed_with_failures' if failed else 'complete', failed_cells=failed)
    except BudgetExceeded as error:
        state.update(status='budget_insufficient' if str(error).startswith('Estimated') else 'budget_exhausted', message=str(error))
        print(str(error), flush=True)
    except BaseException as error:
        state.update(status='failed', message=repr(error))
        raise
    finally:
        state['finished_utc'] = time.time()
        state['original_checkpoint_sha256'] = sha256(engine.original)
        engine.write_cell(out/'state.json', state)
        report(out, config)
        engine.budget_path.write_text(json.dumps({'elapsed_seconds': engine.previous+time.monotonic()-engine.started,
            'limit_seconds': config['budget_hours']*3600}, indent=2))
    print(json.dumps(state, indent=2), flush=True)
    if state['status'] == 'completed_with_failures':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
