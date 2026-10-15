"""Check adjoint task gradients on actual paired Haar inference sources after timing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posed_flow_colab_poc import PoseFlowMLP
from src.acceleration.colab_benchmark import BodyField, poses
from src.acceleration.adjoint import integrate, adjoint_integrate
from src.study.geometry import exp, log, sample
from src.study.data import sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
p.add_argument('--train', type=Path, default=Path('data/3dpw_train_smpl72.npz'))
p.add_argument('--validation', type=Path, default=Path('data/validation.npz'))
args = p.parse_args()
device = torch.device('cuda')
torch.set_num_threads(2)
manifest = json.loads((args.results/'manifest.json').read_text())
network = PoseFlowMLP(21,512,True).to(device)
saved = torch.load(args.results/'teacher.pt', map_location=device, weights_only=False)
network.load_state_dict(saved['state_dict'])
teacher = BodyField(network).eval().requires_grad_(False)
train, meta = poses(args.train, device)
validation, _ = poses(args.validation, device)
mean = exp(torch.tensor(meta['poses'].mean(0),device=device).reshape(21,3))
ids = manifest['validation_ids'][:2]
truth = validation[ids]
rows = []
for mask, missing in manifest['masks'].items():
    visible = torch.ones(21,dtype=torch.bool,device=device); visible[missing]=False
    observation = truth.clone(); observation[:,~visible]=mean[~visible]
    observation = observation.repeat_interleave(2,0)
    for seed in (0,1):
        source = torch.cat([sample((2,21),device,generator=torch.Generator(device=device).manual_seed(seed*1000003+int(case))) for case in ids])
        for n in (50,100,200):
            gradients = []
            stats = {}
            for checked in (False,True):
                delta = source.new_zeros(*source.shape[:2],3).requires_grad_()
                initial = source@exp(delta)
                prediction,penalty = (adjoint_integrate(teacher,initial,n,tolerance=2e-7,stats=stats)
                                      if checked else integrate(teacher,initial,n))
                task = log(prediction.transpose(-1,-2)@observation)[:,visible].norm(dim=-1).sum(-1).mean()
                gradient, = torch.autograd.grad(task+.001*100/n*penalty.mean(),delta)
                gradients.append(gradient.detach())
            full,adjoint = gradients
            relative = ((adjoint-full).norm()/full.norm().clamp_min(1e-8)).item()
            cosine = F.cosine_similarity(adjoint.flatten()[None],full.flatten()[None]).item()
            rows.append({'mask':mask,'seed':seed,'steps':n,'source':'paired Haar hypotheses',
                         'gradient_relative_error':relative,'gradient_cosine':cosine,'reconstruction':stats,
                         'passed':relative<=.01 and cosine>=.999})
            print(json.dumps(rows[-1]),flush=True)
(args.results/'adjoint_source_audit.json').write_text(json.dumps(rows,indent=2))
(args.results/'adjoint_source_audit_manifest.json').write_text(json.dumps({
    'source_sha256':sha256(Path(__file__)), 'teacher_sha256':sha256(args.results/'teacher.pt'),
    'validation_ids':ids,'probe_hypotheses':4,'includes_running_cost':True,
    'gate':{'relative_error':.01,'cosine':.999},'all_passed':all(r['passed'] for r in rows)},indent=2))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
