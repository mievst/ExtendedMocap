"""Extract quaternion CSVs from every source blend via Blender headless.

Chains ``blender --background --python extract_quats_from_blends.py`` over all
six source .blend files and merges the per-blend manifests into one dataset
manifest keyed by ``motion_id``.

Run from repo root::

    uv run python scripts/run_extraction.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLEND_SCRIPT = ROOT / "scripts" / "extract_quats_from_blends.py"

BLEND_FILES = [
    "mocap.blend",
    "mocap_extended.blend",
    "mocap_accurig.blend",
    "mocap_mixamo.blend",
    "mocap_mocca1.blend",
    "data/mocap.blend",
]

MANIFEST_PATH = ROOT / "data" / "blend_manifest.json"


def find_blender() -> str | None:
    for candidate in ["blender", "blender.exe"]:
        if shutil.which(candidate):
            return candidate
    for base in [
        r"C:\Program Files\Blender Foundation",
        r"C:\Program Files (x86)\Blender Foundation",
    ]:
        root = Path(base)
        if not root.is_dir():
            continue
        for exe in sorted(root.rglob("blender.exe"), reverse=True):
            return str(exe)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", default=str(ROOT / "data" / "mocap" / "csv"), help="Output dir for quat CSVs")
    p.add_argument("--manifest", default=str(MANIFEST_PATH), help="Merged dataset manifest path")
    p.add_argument("--blender", default=None, help="Path to blender executable (auto-detect if omitted)")
    p.add_argument("--only", default=None, help="Comma-separated subset of blend filenames (e.g. mocap.blend,mocap_mixamo.blend)")
    args = p.parse_args()

    blender = args.blender or find_blender()
    if blender is None:
        raise SystemExit("blender executable not found; pass --blender PATH")

    csv_dir = ROOT / args.csv_dir
    manifest_path = ROOT / args.manifest
    if manifest_path.exists():
        manifest_path.unlink()  # fresh manifest each run

    only = set(args.only.split(",")) if args.only else None
    for blend_name in BLEND_FILES:
        if only and blend_name not in only:
            continue
        blend_path = ROOT / blend_name
        if not blend_path.is_file():
            print(f"[skip] {blend_name} not found")
            continue
        print(f"[run] {blend_name}")
        cmd = [
            blender, "--background", "--python", str(BLEND_SCRIPT),
            "--", "--blend", str(blend_path), "--csv-dir", str(csv_dir),
            "--manifest", str(manifest_path),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            print(f"[fail] {blend_name} (exit {result.returncode})")
            continue

    print(f"\nmanifest: {manifest_path}")
    print("next: python scripts/build_dataset.py --manifest ...")


if __name__ == "__main__":
    main()