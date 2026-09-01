"""Upload the paired MediaPipe/quaternion CSVs to HuggingFace Datasets.

The training set consists of matched pairs of CSVs:
    data/mediapipe/csv/<name>.csv   -> engineered MediaPipe features
    data/mocap/csv/<name>.csv       -> target bone-rotation quaternions

Filenames differ between the two sets, but the sorted order is aligned
(see the original notebooks: ``mp_names[i]`` <-> ``quat_names[i]``).

This script builds a small metadata dataset and uploads the raw CSV files
alongside it. Requires the ``datasets`` and ``huggingface_hub`` packages
(optional extras) plus a logged-in ``huggingface-cli``.

Usage:
    uv run --extra hf python scripts/upload_to_hf.py \
        --repo-id yourname/extended-mocap-data \
        --limit 20
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def build_pairs(mediapipe_dir: str, mocap_dir: str, limit: int | None) -> list[dict]:
    mp_names = sorted(f for f in os.listdir(mediapipe_dir) if f.endswith(".csv"))
    quat_names = sorted(f for f in os.listdir(mocap_dir) if f.endswith(".csv"))

    min_len = min(len(mp_names), len(quat_names))
    if limit:
        min_len = min(min_len, limit)

    pairs = []
    for i in range(min_len):
        mp_file = mp_names[i]
        quat_file = quat_names[i]
        mp_df = pd.read_csv(os.path.join(mediapipe_dir, mp_file))
        pairs.append(
            {
                "id": os.path.splitext(quat_file)[0],
                "num_frames": len(mp_df),
                "mp_features": len(mp_df.columns),
                "mp_csv": mp_file,
                "quat_csv": quat_file,
                "mp_path": os.path.join(mediapipe_dir, mp_file),
                "quat_path": os.path.join(mocap_dir, quat_file),
            }
        )
    return pairs


def upload(repo_id: str, pairs: list[dict], private: bool) -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    meta = pd.DataFrame(
        [{k: v for k, v in p.items() if k not in ("mp_path", "quat_path")} for p in pairs]
    )
    dataset = Dataset.from_pandas(meta)
    ds = DatasetDict({"train": dataset})
    ds.push_to_hub(repo_id, private=private)

    api = HfApi()
    for p in pairs:
        api.upload_file(
            path_or_fileobj=p["mp_path"],
            path_in_repo=os.path.join("mediapipe", p["mp_csv"]),
            repo_id=repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=p["quat_path"],
            path_in_repo=os.path.join("mocap", p["quat_csv"]),
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"Uploaded {len(pairs)} pairs to {repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="extended-mocap-data")
    parser.add_argument("--mediapipe-dir", default="data/mediapipe/csv")
    parser.add_argument("--mocap-dir", default="data/mocap/csv")
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to upload")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    pairs = build_pairs(args.mediapipe_dir, args.mocap_dir, args.limit)
    print(f"Found {len(pairs)} matched pairs")
    if not pairs:
        return 1
    upload(args.repo_id, pairs, args.private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
