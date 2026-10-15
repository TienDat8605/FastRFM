"""
AMASS dataset pipeline, split manifests, and preprocessing.

Handles:
- Train, val, test, and evaluation splits
- Resampling to 30 Hz and trimming first/last 10%
- Converting SMPL axis-angle body poses to SO(3) rotation matrices
- PyTorch Dataset and manifest generation
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.geometry.so3 import so3_exp_map, project_to_so3

# Standard AMASS split subsets according to NRDF / VPoser / PoseD-Flow
AMASS_SPLITS = {
    "train": [
        "CMU",
        "MPI_Limits",
        "TotalCapture",
        "Eyes_Japan_Dataset",
        "KIT",
        "BioMotionLab_NTroje",
        "BMLhandball",
        "BMLmovi",
        "EKUT",
        "TCD_handMocap",
        "ACCAD",
    ],
    "val": [
        "MPI_HDM05",
        "SFU",
        "MPI_mosh",
    ],
    "test": [
        "Transitions_mocap",
        "SSM_synced",
    ],
    # Reserved exclusively for denoising evaluation, excluded from tuning
    "humaneva": [
        "HumanEva",
    ],
}


def process_amass_npz(
    npz_path: Union[str, Path],
    target_fps: float = 30.0,
    trim_pct: float = 0.10,
) -> Optional[np.ndarray]:
    """
    Load an AMASS .npz file, trim the first/last trim_pct, resample to target_fps,
    and extract the 21 body joint rotations (excluding root).
    Returns:
        Array of shape (T, 21, 3) in axis-angle format, or None if invalid.
    """
    try:
        data = np.load(npz_path)
    except Exception:
        return None

    if "poses" not in data or "mocap_framerate" not in data:
        return None

    poses = data["poses"]  # (T_raw, 72) or (T_raw, 156)
    mocap_fps = float(data["mocap_framerate"])

    total_frames = poses.shape[0]
    if total_frames < 30:
        return None

    # 1. Trim first and last 10%
    start_idx = int(total_frames * trim_pct)
    end_idx = int(total_frames * (1.0 - trim_pct))
    if end_idx - start_idx < 10:
        return None

    trimmed_poses = poses[start_idx:end_idx]

    # 2. Resample to target_fps (30 Hz)
    step = max(1, int(round(mocap_fps / target_fps)))
    resampled_poses = trimmed_poses[::step]

    # 3. Extract 21 body joint rotations (indices 3:66 correspond to joints 1 to 21)
    body_poses_aa = resampled_poses[:, 3:66].reshape(-1, 21, 3)
    return body_poses_aa


class AMASSPoseDataset(Dataset):
    """
    PyTorch Dataset serving preprocessed AMASS poses as rotation matrices in SO(3)^21.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        subsample_step: int = 1,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.subsample_step = subsample_step

        self.allowed_subsets = AMASS_SPLITS.get(split, [])
        self.samples: List[torch.Tensor] = []

        self._load_dataset(max_samples)

    def _load_dataset(self, max_samples: Optional[int]):
        """Load processed poses from disk if directory exists."""
        if not self.data_dir.exists():
            return

        all_poses = []
        for subset in self.allowed_subsets:
            subset_dir = self.data_dir / subset
            if not subset_dir.exists():
                continue

            for npz_file in subset_dir.rglob("*.npz"):
                poses_aa = process_amass_npz(npz_file)
                if poses_aa is not None:
                    # Convert to SO(3) rotation matrices
                    poses_tensor = torch.from_numpy(poses_aa[::self.subsample_step]).float()
                    rot_matrices = so3_exp_map(poses_tensor)  # (N, 21, 3, 3)
                    rot_matrices = project_to_so3(rot_matrices)
                    all_poses.append(rot_matrices)

                    if max_samples and sum(p.shape[0] for p in all_poses) >= max_samples:
                        break
            if max_samples and sum(p.shape[0] for p in all_poses) >= max_samples:
                break

        if all_poses:
            self.samples = torch.cat(all_poses, dim=0)
            if max_samples:
                self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def generate_dataset_manifest(data_dir: Union[str, Path], output_json: Union[str, Path]) -> Dict:
    """
    Scan AMASS data directory and generate a split manifest with file counts.
    """
    data_dir = Path(data_dir)
    manifest = {k: [] for k in AMASS_SPLITS.keys()}

    if not data_dir.exists():
        manifest["status"] = "directory_not_found"
        with open(output_json, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    for split_name, subsets in AMASS_SPLITS.items():
        for subset in subsets:
            subset_path = data_dir / subset
            if subset_path.exists():
                files = [str(p.relative_to(data_dir)) for p in subset_path.rglob("*.npz")]
                manifest[split_name].extend(files)

    with open(output_json, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
