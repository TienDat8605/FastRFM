"""Package pose-only inputs and official assets for the reduced-data study."""
from pathlib import Path
import hashlib
import json
import pickle
import shutil
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/colab/staging"
STAGE.mkdir(parents=True, exist_ok=True)
manifest = {"scope": "3DPW poses and AMASS pose-only inputs; images excluded", "splits": {}}
for split in ("train", "validation", "test"):
    arrays, sequences, frames, people = [], [], [], []
    for path in sorted((Path('/home/dat/HMR/data/3dpw/sequenceFiles') / split).glob('*.pkl')):
        with path.open('rb') as stream:
            data = pickle.load(stream, encoding='latin1')
        for person, pose in enumerate(data['poses']):
            pose = np.asarray(pose[:, 3:66], dtype=np.float32)
            arrays.append(pose)
            sequences.extend([path.stem] * len(pose))
            frames.extend(range(len(pose)))
            people.extend([person] * len(pose))
    destination = STAGE / 'data/3dpw'
    destination.mkdir(parents=True, exist_ok=True)
    poses = np.concatenate(arrays)
    np.savez_compressed(destination / f'{split}.npz', poses=poses,
                        sequence=np.asarray(sequences), frame=np.asarray(frames),
                        person=np.asarray(people))
    manifest['splits'][split] = {"person_frames": len(poses), "sequences": len(set(sequences))}
    print(f'3DPW {split}: {len(poses)} poses', flush=True)

amass_files = 0
for path in sorted((ROOT / 'data/raw_amass').rglob('*.npz')):
    destination = STAGE / 'data/amass_pose' / path.relative_to(ROOT / 'data/raw_amass')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as source:
        fields = {key: source[key] for key in ('poses', 'trans', 'betas', 'mocap_framerate', 'mocap_frame_rate', 'gender') if key in source}
        np.savez_compressed(destination, **fields)
    amass_files += 1
manifest['amass_files'] = amass_files
shutil.copy2(ROOT / 'data/amass/split_manifest.json', STAGE / 'data/amass_split_manifest.json')

for folder in ('src', 'configs'):
    shutil.copytree(ROOT / folder, STAGE / folder, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
(STAGE / 'reference').mkdir(exist_ok=True)
shutil.copy2('/home/dat/HMR/code/src/models/pose_rfm.py', STAGE / 'reference/hmr_pose_rfm.py')
shutil.copy2('/home/dat/HMR/checkpoints/pose_rfm_3dpw.pt', STAGE / 'reference/pose_rfm_3dpw.pt')
shutil.copy2(ROOT / 'scripts/colab_pose_pilot.py', STAGE / 'colab_pose_pilot.py')

for gender, token in [('female', 'f'), ('male', 'm'), ('neutral', 'neutral')]:
    source = next((ROOT / 'models/smpl').rglob(f'basicmodel_{token}_*.pkl'))
    destination = STAGE / f'models/smpl/SMPL_{gender.upper()}.pkl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
for source in (ROOT / 'models/smplx').rglob('*.npz'):
    destination = STAGE / 'models/smplx' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
(STAGE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
archive = ROOT / 'results/colab/posedflow_inputs.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
    bundle.add(STAGE, arcname='study')
digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
(archive.parent / 'bundle.sha256').write_text(digest)
print(f'Bundle: {archive.stat().st_size / 2**20:.1f} MiB; SHA256 {digest}', flush=True)
