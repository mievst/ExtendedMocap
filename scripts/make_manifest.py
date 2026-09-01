"""Build a dataset manifest of motion groups.

MediaPipe CSVs are named ``<prefix>.<range>.csv``; the ``<range>`` suffix
groups every clip retargeted from the same source motion (repeats on different
NLA layers). This script writes a JSON manifest at ``data/motion_manifest.json``
that lists each motion, its repeat clips, and the frame range, so downstream
tools can deduplicate and split by motion rather than by clip.

Usage:  uv run python scripts/make_manifest.py [--mediapipe DIR] [--mocap DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extended_mocap.evaluation import load_dataset, split_motions

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mediapipe", default=str(ROOT / "data" / "mediapipe" / "csv"))
    parser.add_argument("--mocap", default=str(ROOT / "data" / "mocap" / "csv"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--out", default=str(ROOT / "data" / "motion_manifest.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples, _feature_columns = load_dataset(args.mediapipe, args.mocap, args.cache)

    groups: dict[str, dict] = {}
    for s in samples:
        g = groups.setdefault(
            s.motion_id,
            {"motion_id": s.motion_id, "reps": [], "frames": None, "mediapipe_files": {}},
        )
        g["reps"].append(s.name)
        g["frames"] = s.features.shape[0]

    for s in samples:
        mid = s.motion_id or ""
        mp_file = f"{s.name}.{mid}.csv" if mid else f"{s.name}.csv"
        groups[s.motion_id]["mediapipe_files"][s.name] = mp_file

    _train, test = split_motions(samples, test_size=args.test_size, seed=args.seed)
    test_ids = {s.motion_id for s in test}
    for mid, g in groups.items():
        g["split"] = "test" if mid in test_ids else "train"

    ordered = sorted(groups.values(), key=lambda g: g["motion_id"])
    manifest = {
        "generated_by": "scripts/make_manifest.py",
        "totals": {
            "animations": len(samples),
            "motions": len(ordered),
            "motions_train": sum(1 for g in ordered if g["split"] == "train"),
            "motions_test": sum(1 for g in ordered if g["split"] == "test"),
            "clips_train": sum(len(g["reps"]) for g in ordered if g["split"] == "train"),
            "clips_test": sum(len(g["reps"]) for g in ordered if g["split"] == "test"),
        },
        "motions": ordered,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    t = manifest["totals"]
    print(
        f"animations {t['animations']} | motions {t['motions']} "
        f"(train {t['motions_train']}, test {t['motions_test']}) | "
        f"clips train {t['clips_train']}, test {t['clips_test']}"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()