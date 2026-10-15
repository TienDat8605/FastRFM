"""Validate real body models and run a small, explicitly scoped completion pilot."""
from pathlib import Path
import inspect
import json
import sys
import time

import numpy as np
import torch

# Official legacy SMPL pickles reference Chumpy, which predates these removals.
for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                    'object': object, 'str': str, 'unicode': str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import smplx

ROOT = Path('/content/posedflow/study')
sys.path.insert(0, str(ROOT / 'reference'))
from hmr_pose_rfm import PoseRFM, RiemannianODESolver, so3_exp, so3_log

torch.manual_seed(0)
device = torch.device('cuda')
assert torch.cuda.is_available(), 'CUDA is required'
output = ROOT / 'output'
output.mkdir(exist_ok=True)
report = {'scope': 'setup validation and 8-pose left-leg pilot; not paper reproduction',
          'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
          'manifest': json.loads((ROOT / 'manifest.json').read_text()), 'body_models': {}}

for kind, body_count in [('smpl', 23), ('smplx', 21)]:
    for gender in ('neutral', 'male', 'female'):
        print(f'Loading official {kind} {gender}', flush=True)
        body = smplx.create(str(ROOT / 'models'), model_type=kind, gender=gender,
                            ext='pkl' if kind == 'smpl' else 'npz',
                            num_betas=10, use_pca=False, batch_size=1).to(device)
        pose = torch.zeros(1, body_count * 3, device=device, requires_grad=True)
        result = body(body_pose=pose)
        loss = result.vertices.square().mean()
        gradient, = torch.autograd.grad(loss, pose)
        assert torch.isfinite(result.vertices).all() and torch.isfinite(gradient).all()
        report['body_models'][f'{kind}_{gender}'] = {
            'vertices': list(result.vertices.shape), 'joints': list(result.joints.shape),
            'finite_forward_and_gradient': True}
        del body, result, pose, gradient

checkpoint = torch.load(ROOT / 'reference/pose_rfm_3dpw.pt', map_location=device, weights_only=False)
args = checkpoint['args']
model = PoseRFM(num_joints=21, hidden_dim=args['hidden_dim'],
                time_embed_dim=args['time_embed_dim'], num_layers=args['num_layers']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().requires_grad_(False)
report['checkpoint'] = {'epoch': checkpoint['epoch'], 'val_loss': checkpoint['val_loss'], 'args': args}
train = np.load(ROOT / 'data/3dpw/train.npz')
test = np.load(ROOT / 'data/3dpw/test.npz')
# Spread the pilot across sequences rather than adjacent frames.
_, first = np.unique(test['sequence'], return_index=True)
indices = np.sort(first)[:8]
gt = so3_exp(torch.tensor(test['poses'][indices], device=device).reshape(-1, 21, 3))
mean = so3_exp(torch.tensor(train['poses'].mean(0), device=device).reshape(21, 3))
observed = torch.ones(21, dtype=torch.bool, device=device)
observed[[0, 3, 6, 9]] = False
obs = gt.clone()
obs[:, ~observed] = mean[~observed]
body = smplx.create(str(ROOT / 'models'), model_type='smpl', gender='neutral',
                     num_betas=10, batch_size=len(gt)).to(device)

def body_output(rotations):
    # The prior models 21 joints; SMPL also requires two hand rotations.
    aa = so3_log(rotations).reshape(len(rotations), 63)
    return body(body_pose=torch.cat([aa, aa.new_zeros(len(aa), 6)], dim=-1))

with torch.no_grad():
    gt_body = body_output(gt)
    base_body = body_output(obs)
    baseline_mm = ((base_body.joints[:, [1, 4, 7, 10]] - gt_body.joints[:, [1, 4, 7, 10]]).norm(dim=-1).mean() * 1000).item()
base = obs.detach()
delta = torch.zeros(len(gt), 21, 3, device=device, requires_grad=True)
optimizer = torch.optim.Adam([delta], lr=0.05)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
losses = []
for iteration in range(25):
    source = base @ so3_exp(delta)
    pred = RiemannianODESolver.solve(model, source, num_steps=10, method='euler')
    error = so3_log(pred.transpose(-1, -2) @ obs)[:, observed].square().sum(-1).mean()
    optimizer.zero_grad()
    error.backward()
    assert torch.isfinite(error) and torch.isfinite(delta.grad).all()
    optimizer.step()
    losses.append(error.item())
    print(f'Completion iteration {iteration + 1}/25 loss={error.item():.6f}', flush=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
with torch.no_grad():
    pred = RiemannianODESolver.solve(model, base @ so3_exp(delta), num_steps=10, method='euler')
    visible_error = so3_log(pred.transpose(-1, -2) @ gt)[:, observed].norm(dim=-1).mean().item()
    pred[:, observed] = gt[:, observed]
    result = body_output(pred)
    error_mm = ((result.joints[:, [1, 4, 7, 10]] - gt_body.joints[:, [1, 4, 7, 10]]).norm(dim=-1).mean() * 1000).item()
report['completion_pilot'] = {'poses': len(gt), 'hypotheses': 1, 'ode_steps': 10,
    'iterations': 25, 'initialization': 'mean-filled observed pose used directly as source',
    'optimizer': 'Adam in fixed exponential chart; reconstruction choice',
    'trajectory_penalty': False, 'mean_fill_occluded_mpjpe_mm': baseline_mm,
    'guided_occluded_mpjpe_mm': error_mm, 'visible_error_before_replacement_rad': visible_error,
    'elapsed_seconds': elapsed, 'peak_allocated_vram_mib': torch.cuda.max_memory_allocated()/2**20,
    'loss_history': losses, 'test_indices': indices.tolist()}
torch.save({'pred_rotations': pred.cpu(), 'gt_rotations': gt.cpu()}, output / 'predictions.pt')
(output / 'pilot_report.json').write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report['completion_pilot'], indent=2), flush=True)
print('PILOT COMPLETE', flush=True)
