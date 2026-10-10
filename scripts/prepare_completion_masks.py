#!/usr/bin/env python3
"""
Precompute and audit completion masks and test frame sampling for Table 2.
"""

from pathlib import Path
import json
import torch
from src.data.masks import OCCLUSION_JOINT_INDICES, get_occlusion_mask


def main():
    out_dir = Path("data/completion")
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_spec = {}
    for mask_name, indices in OCCLUSION_JOINT_INDICES.items():
        mask_tensor = get_occlusion_mask(mask_name, num_joints=21)
        mask_spec[mask_name] = {
            "joint_indices": indices,
            "num_occluded_joints": len(indices),
            "num_visible_joints": int((~mask_tensor).sum().item()),
            "mask_boolean": mask_tensor.tolist(),
        }

    out_file = out_dir / "occlusion_masks.json"
    with open(out_file, "w") as f:
        json.dump(mask_spec, f, indent=2)

    print(f"Audited {len(mask_spec)} occlusion masks:")
    for k, v in mask_spec.items():
        print(f"  - {k:10s}: {v['num_occluded_joints']} occluded joints, {v['num_visible_joints']} visible joints")
    print(f"Saved mask specification to {out_file}")


if __name__ == "__main__":
    main()
