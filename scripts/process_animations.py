"""Invoke the Blender headless pipeline for one or more animations.

Wraps ``blender --background --python blender/headless_process.py`` with
sensible defaults. Run from the repo root::

    uv run python scripts/process_animations.py --anim-dir data/mixamo_fbx --csv-dir data/mixamo_csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER_SCRIPT = ROOT / "src" / "extended_mocap" / "blender" / "headless_process.py"


def find_blender() -> str | None:
    for candidate in ["blender", "blender.exe"]:
        if shutil.which(candidate):
            return candidate
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anim-dir", required=True, help="Directory of Mixamo FBX files")
    p.add_argument("--csv-dir", required=True, help="Output dir for quaternion CSVs")
    p.add_argument("--render-dir", default=None, help="Optional output dir for MP4 renders")
    p.add_argument("--base", default=str(ROOT / "data" / "base.blend"), help="Base .blend file with CC_Base rig")
    p.add_argument("--model-name", default="party-m-0001", help="CC_Base object name in base scene")
    p.add_argument("--blender", default=None, help="Path to blender executable (auto-detect if omitted)")
    args = p.parse_args()

    blender = args.blender or find_blender()
    if blender is None:
        raise SystemExit("blender executable not found; pass --blender PATH")

    cmd = [
        blender, "--background", "--python", str(BLENDER_SCRIPT),
        "--", "--anim-dir", str(args.anim_dir), "--csv-dir", str(args.csv_dir),
        "--base", str(args.base), "--model-name", args.model_name,
    ]
    if args.render_dir:
        cmd += ["--render-dir", str(args.render_dir)]

    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
