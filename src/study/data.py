"""Immutable balanced cohorts with sequence/person/frame provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(root, split):
    with np.load(Path(root)/'data/3dpw'/f'{split}.npz') as source:
        data = {key: source[key] for key in source.files}
    if data['poses'].ndim != 2 or data['poses'].shape[1] != 63 or not np.isfinite(data['poses']).all():
        raise ValueError(f'Invalid 21-joint poses in {split}')
    if any(len(data[key]) != len(data['poses']) for key in ('sequence', 'person', 'frame')):
        raise ValueError(f'Misaligned provenance in {split}')
    return data


def balanced(items, count, rng):
    groups = {key: list(value) for key, value in items.items()}
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < count:
        active = sorted(key for key, group in groups.items() if group)
        if not active:
            raise ValueError(f'Only {len(result)} eligible examples; requested {count}')
        rng.shuffle(active)
        for key in active:
            result.append(groups[key].pop())
            if len(result) == count:
                break
    return result


def prepare(root, config, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    datasets = {split: load(root, split) for split in ('train', 'validation', 'test')}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(datasets[left]['sequence']) & set(datasets[right]['sequence']), 'Sequence leakage'
    prior_exposed = {0, 1796, 4602, 8958, 11360, 13400, 14576, 15433}
    manifest = {'version': config['study'], 'seed': config['seed'],
        'prior_test_exposure': 'Prior HMR pilots used test data; no pristine-test claim.',
        'sampling': 'Balanced by sequence; poses at frame multiples of 10; disjoint 60-frame segments.',
        'fps_source': '3DPW readme specifies poses at 30 Hz; arrays retain original frame identifiers.',
        'source_hashes': {s: sha256(root/'data/3dpw'/f'{s}.npz') for s in datasets}, 'cohorts': {}}
    for cohort, split in [('development', 'validation'), ('final', 'test'), ('smoke', 'validation'), ('benchmark', 'validation')]:
        if cohort not in config:
            continue
        data = datasets[split]
        rng = np.random.default_rng(config['seed'])
        groups, segments = {}, {}
        for sequence in np.unique(data['sequence']):
            ids = np.flatnonzero(data['sequence'] == sequence)
            groups[str(sequence)] = [int(i) for i in ids if data['frame'][i] % 10 == 0
                                    and (split != 'test' or i not in prior_exposed)]
            segments[str(sequence)] = []
            for person in np.unique(data['person'][ids]):
                actor = ids[data['person'][ids] == person]
                actor = actor[np.argsort(data['frame'][actor])]
                for start in range(0, len(actor)-59, 60):
                    chunk = actor[start:start+60]
                    if np.all(np.diff(data['frame'][chunk]) == 1):
                        if split != 'test' or not prior_exposed.intersection(map(int, chunk)):
                            segments[str(sequence)].append(chunk.tolist())
        manifest['cohorts'][cohort] = {'split': split,
            'poses': balanced(groups, config[cohort]['poses'], rng),
            'segments': balanced(segments, config[cohort]['segments'], rng)}
    # Resume must use the original manifest; a changed input/config is an error.
    target = destination/'manifest.json'
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    if target.exists() and target.read_text() != encoded:
        raise ValueError('Frozen manifest differs; use a new output directory')
    target.write_text(encoded)
    return manifest
