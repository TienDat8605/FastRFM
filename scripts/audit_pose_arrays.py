"""Inventory pose-only inputs and export real 3DPW training poses as SMPL-72."""
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root/'results/acceleration_pilot'
out.mkdir(parents=True, exist_ok=True)
files, frames = Counter(), Counter()
for path in sorted((root/'data/raw_amass').rglob('*.npz')):
    with np.load(path, allow_pickle=False) as source:
        poses = source['poses']
        if poses.ndim != 2 or not np.isfinite(poses).all():
            raise ValueError(f'Invalid poses: {path}')
        key = path.relative_to(root/'data/raw_amass').parts[0]
        files[(key, poses.shape[1])] += 1
        frames[(key, poses.shape[1])] += len(poses)
inventory = {'AMASS': [{'collection': key, 'dimensions': dims, 'files': count,
                        'frames': frames[(key, dims)]} for (key, dims), count in files.items()],
             'AMASS_layout': 'SMPL-H 156: root 3 + body 63 + articulated hands 90. '
                'Do not interpret the first 72 coordinates as SMPL-72. PoseRFM uses poses[:,3:66].',
             '3DPW': {}, 'exports': []}
for split in ('train', 'validation', 'test'):
    arrays, sequence, people, frame = [], [], [], []
    hashes, excluded = {}, []
    for path in sorted((Path('/home/dat/HMR/data/3dpw/sequenceFiles')/split).glob('*.pkl')):
        if path.stem.startswith('mock_'):
            excluded.append(str(path)); continue
        with path.open('rb') as stream:
            source = pickle.load(stream, encoding='latin1')
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        for person, pose in enumerate(source['poses']):
            pose = np.asarray(pose, dtype=np.float32)
            if pose.ndim != 2 or pose.shape[1] != 72 or not np.isfinite(pose).all():
                raise ValueError(f'Invalid SMPL-72: {path}')
            arrays.append(pose)
            sequence.extend([path.stem]*len(pose))
            people.extend([person]*len(pose))
            frame.extend(range(len(pose)))
    packed = np.concatenate(arrays)
    inventory['3DPW'][split] = {'frames': len(packed), 'dimensions': 72,
        'sequences': len(set(sequence)), 'excluded_mock_files': excluded, 'source_sha256': hashes}
    if split == 'train':
        destination = root/'data/pose_arrays/3dpw_train_smpl72.npz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, poses=packed, sequence=np.asarray(sequence),
                            person=np.asarray(people), frame=np.asarray(frame))
        with destination.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        inventory['exports'].append({'path': str(destination.relative_to(root)),
            'sha256': digest, 'bytes': destination.stat().st_size,
            'layout': 'Original 24-joint axis-angle SMPL poses, including root and hand joints; radians'})
(out/'pose_inventory.json').write_text(json.dumps(inventory, indent=2))
print(json.dumps({**inventory, '3DPW': {k: {f: v for f, v in d.items() if f != 'source_sha256'}
                                     for k, d in inventory['3DPW'].items()}}, indent=2))
