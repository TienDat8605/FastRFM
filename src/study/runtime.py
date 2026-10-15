"""Checkpoint adapter, official SMPL, and task metrics."""
import importlib.util
import inspect
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .geometry import exp, log


def load_prior(root, checkpoint, device):
    path = Path(root)/'reference/hmr_pose_rfm.py'
    spec = importlib.util.spec_from_file_location('study_hmr_prior', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    args = saved['args']
    model = module.PoseRFM(num_joints=21, hidden_dim=args['hidden_dim'],
                          time_embed_dim=args['time_embed_dim'], num_layers=args['num_layers']).to(device)
    model.load_state_dict(saved['model_state_dict'])
    return model.eval().requires_grad_(False), saved


def weight_hash(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class OfficialBody:
    def __init__(self, root, device):
        # Compatibility for official Chumpy-based pickles only.
        for key, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                           'object': object, 'str': str, 'unicode': str}.items():
            if key not in np.__dict__:
                setattr(np, key, value)
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
        import smplx
        self.model = smplx.create(str(Path(root)/'models'), model_type='smpl', gender='neutral',
                                  num_betas=10, batch_size=1).to(device).requires_grad_(False)
        self.model_hash = sha256(Path(root)/'models/smpl/SMPL_NEUTRAL.pkl')
        self.vertex_joint = self.model.lbs_weights.argmax(-1)
        self.rest_joints = self.model.J_regressor@self.model.v_template

    def __call__(self, rotations):
        n = len(rotations)
        aa = log(rotations).reshape(n, 63)
        # Explicit batch dimensions avoid implicit batch-size-1 parameters.
        z = aa.new_zeros
        out = self.model(body_pose=torch.cat((aa, z(n, 6)), -1), betas=z(n, 10),
                         global_orient=z(n, 3), transl=z(n, 3), return_verts=True)
        return out.joints[:, :22], out.vertices

    def joints(self, rotations):
        """Exact first-22 SMPL joints at fixed zero shape/root/translation.

        SMPL computes these before pose blend shapes and vertex skinning. Reuse
        its official rigid-transform implementation rather than compute unused
        meshes inside a joint-only denoising objective.
        """
        from smplx.lbs import batch_rigid_transform
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        full = torch.cat((identity.expand(len(rotations), 1, 3, 3), rotations,
                          identity.expand(len(rotations), 2, 3, 3)), dim=1)
        joints, _ = batch_rigid_transform(full, self.rest_joints.expand(len(rotations), -1, -1),
                                          self.model.parents)
        return joints[:, :22]


def completion_metrics(body, pred, gt, observed, hypotheses):
    validate_rotations(pred)
    n = len(gt)
    repeated = gt.repeat_interleave(hypotheses, 0)
    visible = log(pred.transpose(-1, -2)@repeated)[:, observed].norm(dim=-1).mean(-1)
    pred = pred.clone()
    pred[:, observed] = repeated[:, observed]
    with torch.no_grad():
        joints, vertices = body(pred)
        true_joints, true_vertices = body(gt)
        masked = torch.nonzero(~observed).flatten()+1
        region = torch.isin(body.vertex_joint, masked)
        if not region.any():
            raise ValueError('Empty evaluation vertex region')
        error = (vertices[:, region]-true_vertices.repeat_interleave(hypotheses, 0)[:, region]).norm(dim=-1).mean(-1)*1000
        joint_error = (joints[:, masked]-true_joints.repeat_interleave(hypotheses, 0)[:, masked]).norm(dim=-1).mean(-1)*1000
        sets = joints[:, masked].reshape(n, hypotheses, len(masked), 3)
        pairs = torch.triu_indices(hypotheses, hypotheses, offset=1, device=pred.device)
        distances = (sets[:, pairs[0]]-sets[:, pairs[1]]).norm(dim=-1).mean((-1, -2))*100
    return {'mpvpe_mm': error.reshape(n, hypotheses).mean(-1).cpu().tolist(),
            'mpjpe_mm': joint_error.reshape(n, hypotheses).mean(-1).cpu().tolist(),
            'apd_cm': distances.cpu().tolist(),
            'visible_rad': visible.reshape(n, hypotheses).mean(-1).cpu().tolist()}, pred


def denoising_metrics(pred_joints, true_joints):
    error = (pred_joints-true_joints).norm(dim=-1).mean()*1000
    acceleration = pred_joints[2:]-2*pred_joints[1:-1]+pred_joints[:-2]
    true_acceleration = true_joints[2:]-2*true_joints[1:-1]+true_joints[:-2]
    return {'mpjpe_mm': error.item(),
            'acceleration_error_mm_s2': (acceleration-true_acceleration).norm(dim=-1).mean().item()*1000*30**2}


def temporal(rotations, segments=1):
    frames = rotations.reshape(segments, -1, *rotations.shape[1:])
    return log(frames[:, :-1].transpose(-1, -2)@frames[:, 1:]).norm(dim=-1).sum(-1).mean()


def validate_rotations(rotations):
    if not torch.isfinite(rotations).all():
        raise FloatingPointError('Nonfinite recovered rotations')
    identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    error = (rotations.transpose(-1, -2)@rotations-identity).abs().max()
    if error > 2e-3 or (torch.linalg.det(rotations)-1).abs().max() > 2e-3:
        raise FloatingPointError('Recovered rotations are outside SO(3)')
