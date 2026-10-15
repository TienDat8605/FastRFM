"""Data subpackage for dataset loading and masks."""

from src.data.amass_dataset import (
    AMASS_SPLITS,
    AMASSPoseDataset,
    process_amass_npz,
    generate_dataset_manifest,
)
from src.data.masks import (
    OCCLUSION_JOINT_INDICES,
    get_occlusion_mask,
    apply_occlusion,
)

__all__ = [
    "AMASS_SPLITS",
    "AMASSPoseDataset",
    "process_amass_npz",
    "generate_dataset_manifest",
    "OCCLUSION_JOINT_INDICES",
    "get_occlusion_mask",
    "apply_occlusion",
]
