#!/usr/bin/env python3
"""
AMASS dataset preparation and manifest generation script.
Usage:
    python scripts/prepare_amass.py --amass_dir /path/to/AMASS --output_dir data/amass
"""

import argparse
from pathlib import Path
import json
from src.data.amass_dataset import generate_dataset_manifest, AMASS_SPLITS


def main():
    parser = argparse.ArgumentParser(description="Prepare AMASS dataset manifests and audit splits.")
    parser.add_argument("--amass_dir", type=str, default="data/raw_amass", help="Directory containing raw AMASS .npz files.")
    parser.add_argument("--output_dir", type=str, default="data/amass", help="Output directory for manifests and processed data.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = out_dir / "split_manifest.json"
    print(f"Auditing AMASS splits from {args.amass_dir}...")
    manifest = generate_dataset_manifest(args.amass_dir, manifest_file)

    print("\nAMASS Split Summary:")
    for split_name, files in manifest.items():
        if isinstance(files, list):
            print(f"  - {split_name:10s}: {len(files)} sequences across {len(AMASS_SPLITS.get(split_name, []))} sub-datasets")
        else:
            print(f"  - {split_name:10s}: {files}")

    print(f"\nManifest written to {manifest_file}")


if __name__ == "__main__":
    main()
