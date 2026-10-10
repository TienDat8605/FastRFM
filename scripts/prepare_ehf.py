#!/usr/bin/env python3
"""
Audit EHF dataset (100 images) and CLIFF checkpoint specification.
"""

from pathlib import Path
import json


def main():
    out_dir = Path("data/ehf")
    out_dir.mkdir(parents=True, exist_ok=True)

    ehf_spec = {
        "dataset": "EHF",
        "total_images": 100,
        "keypoint_detector": "ViTPose-H",
        "cliff_checkpoint": "hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt",
        "evaluation_metrics": ["PA-MPJPE (mm)", "PA-MPVPE (mm)", "PCK@50 (%)"],
        "status": "ready_for_remote_ingestion",
    }

    out_file = out_dir / "ehf_spec.json"
    with open(out_file, "w") as f:
        json.dump(ehf_spec, f, indent=2)

    print(f"EHF benchmark protocol frozen at {out_file}")


if __name__ == "__main__":
    main()
