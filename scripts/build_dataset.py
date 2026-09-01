"""Assemble the ExtendedMocap dataset from quat CSVs and MediaPipe feature CSVs.

Two data dirs must exist, keyed by a common filename prefix (e.g.
``Armature.001`` in both ``data/mocap/csv`` and ``data/mediapipe/csv``).

``motion_id`` for each clip comes from the blend manifest produced by
``extract_quats_from_blends.py`` (``data/blend_manifest.json``), which maps every
armature CSV to the normalized motion name. This keeps honest dedup: NLA-layer
repeats of one motion share one ``motion_id`` and must stay in one split.

Run from repo root::

    uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json

Optional pipeline steps (when rendered videos are missing):

    uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --render
    uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json --mediapipe
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prefix_of(fname: str) -> str:
    """Pair key = filename before the first '.'."""
    return fname.split(".", 1)[0]


def load_blend_manifest(path: Path) -> dict[str, str]:
    """Map CSV filename -> motion_id from the blend manifest.

    Returns ``{csv_name: motion_id}`` across all blends.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for blend_block in data:
        for entry in blend_block.get("entries", []):
            if entry.get("csv"):
                mapping[entry["csv"]] = entry["motion_id"]
    return mapping


def stem_of(fname: str) -> str:
    """Full filename stem (without .csv) used as the unique file key."""
    return fname.removesuffix(".csv")


def load_pairs(
    quat_dir: Path, mp_dir: Path, blend_map: dict[str, str]
) -> list[tuple[str, Path, Path, str]]:
    """Return ``(stem, quat_path, mp_path, motion_id)`` for every matching pair.

    Pairing is by **exact filename match** between quat and mediapipe dirs, as
    both sides share the same naming convention (``{armature}.{motion}.csv``).
    """
    quat = {p.name: p for p in quat_dir.glob("*.csv")}
    mp = {p.name: p for p in mp_dir.glob("*.csv")}

    pairs: list[tuple[str, Path, Path, str]] = []
    for q_name, q_path in sorted(quat.items()):
        if q_name not in mp:
            continue
        motion = blend_map.get(q_name)
        if motion is None:
            continue
        pairs.append((stem_of(q_name), q_path, mp[q_name], motion))
    return pairs


def build_manifest(pairs: list[tuple[str, Path, Path, str]]) -> dict:
    """Group clips by motion_id; record files, reps, frames."""
    by_motion: dict[str, list[str]] = {}
    files: dict[str, dict] = {}
    for stem, _q, mp_path, motion in pairs:
        by_motion.setdefault(motion, []).append(stem)
        files[stem] = {
            "motion_id": motion,
            "mediapipe": mp_path.name,
            "frames": _read_csv_len(mp_path),
        }
    motions = [
        {"motion_id": m, "reps": reps, "frames": files[reps[0]]["frames"]}
        for m, reps in sorted(by_motion.items())
    ]
    return {
        "generated_by": "scripts/build_dataset.py",
        "num_motions": len(motions),
        "num_clips": len(files),
        "motions": motions,
        "files": files,
    }


def _read_csv_len(path: Path) -> list:
    import pandas as pd

    return list(pd.read_csv(path))


def render_missing(pairs: list[tuple[str, Path, Path, str]], render_dir: Path, base: Path, model: str, blender: str) -> None:
    """Render each quat CSV to MP4 via scripts/render_from_quat_csv.py (if missing)."""
    render_dir.mkdir(parents=True, exist_ok=True)
    render_script = ROOT / "scripts" / "render_from_quat_csv.py"
    for stem, quat_path, _mp, _motion in pairs:
        out = render_dir / f"{stem}.mp4"
        if out.is_file():
            continue
        print(f"[render] {stem}")
        cmd = [
            blender, "--background", "--python", str(render_script),
            "--", "--csv", str(quat_path), "--out", str(out),
            "--base", str(base), "--model", model,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=False)


def extract_mediapipe(pairs: list[tuple[str, Path, Path, str]], render_dir: Path, mp_dir: Path) -> None:
    """Run MediaPipe on rendered MP4s to produce feature CSVs (if missing)."""
    from extended_mocap.extractor import MediapipeExtractor

    mp_dir.mkdir(parents=True, exist_ok=True)
    extractor = MediapipeExtractor()
    for stem, _q, mp_path, _motion in pairs:
        if mp_path.is_file():
            continue
        video = render_dir / f"{stem}.mp4"
        if not video.is_file():
            print(f"[skip] {stem}: no render")
            continue
        print(f"[mediapipe] {stem}")
        extractor.run(str(video), output_csv_path=str(mp_path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend-manifest", required=True, help="Path to data/blend_manifest.json from extraction")
    p.add_argument("--quat-dir", default=str(ROOT / "data" / "mocap" / "csv"))
    p.add_argument("--mp-dir", default=str(ROOT / "data" / "mediapipe" / "csv"))
    p.add_argument("--manifest", default=str(ROOT / "data" / "motion_manifest.json"))
    p.add_argument("--render", action="store_true", help="Render missing MP4s from quat CSVs (needs Blender)")
    p.add_argument("--mediapipe", action="store_true", help="Extract MediaPipe features from rendered MP4s")
    p.add_argument("--render-dir", default=str(ROOT / "data" / "renders"))
    p.add_argument("--base", default=str(ROOT / "data" / "base.blend"), help="Base scene for rendering")
    p.add_argument("--model", default="party-m-0001", help="CC_Base model object name")
    p.add_argument("--blender", default=None, help="Path to blender (auto-detect if omitted)")
    args = p.parse_args()

    blend_map = load_blend_manifest(Path(args.blend_manifest))
    print(f"blend manifest: {len(blend_map)} quat CSVs mapped to motions")

    quat_dir = Path(args.quat_dir)
    mp_dir = Path(args.mp_dir)
    pairs = load_pairs(quat_dir, mp_dir, blend_map)
    print(f"paired clips (quat+mediapipe): {len(pairs)}")

    if args.render:
        blender = args.blender or _find_blender()
        if blender is None:
            raise SystemExit("blender not found; pass --blender PATH")
        render_missing(pairs, Path(args.render_dir), Path(args.base), args.model, blender)

    if args.mediapipe:
        extract_mediapipe(pairs, Path(args.render_dir), mp_dir)

    pairs = load_pairs(quat_dir, mp_dir, blend_map)  # re-read after generation
    manifest = build_manifest(pairs)
    manifest_path = Path(args.manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"{manifest['num_clips']} clips / {manifest['num_motions']} motions")

    dup = Counter(f["motion_id"] for f in manifest["files"].values())
    multi = {m: c for m, c in dup.items() if c > 1}
    print(f"motions with >1 rep: {len(multi)}")
    for m, c in sorted(multi.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {c}x {m}")


def _find_blender() -> str | None:
    if shutil.which("blender"):
        return shutil.which("blender")
    for base in [r"C:\Program Files\Blender Foundation", r"C:\Program Files (x86)\Blender Foundation"]:
        root = Path(base)
        if root.is_dir():
            for exe in sorted(root.rglob("blender.exe"), reverse=True):
                return str(exe)
    return None


if __name__ == "__main__":
    main()