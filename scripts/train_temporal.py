"""Train per-segment GRU window models (batch CLI).

Prefer the interactive notebook for exploring/plotting training progress:
    notebooks/train_temporal.ipynb

Batch usage:
    uv run python scripts/train_temporal.py --epochs 30 --window 6 --stride 3 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from extended_mocap.evaluation import _quat_block_view, segment_subset
from extended_mocap.training import (
    SEGMENTS,
    TrainConfig,
    evaluate_segment_report,
    load_data,
    save_checkpoint,
    segment_bones,
    train_segment,
    write_metrics,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "models" / "temporal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--motion-split", dest="motion_split", action="store_true", default=True,
                        help="Split by unique motion (default)")
    parser.add_argument("--no-motion-split", dest="motion_split", action="store_false",
                        help="Split by animation instead of motion")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        window=args.window,
        hidden=args.hidden,
        layers=args.layers,
        stride=args.stride,
        test_size=args.test_size,
        seed=args.seed,
        device=args.device,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )

    samples, feature_columns, train, test, scaler = load_data(
        str(ROOT / "data" / "mediapipe" / "csv"),
        str(ROOT / "data" / "mocap" / "csv"),
        str(CACHE),
        cfg,
        motion_split=args.motion_split,
    )
    split_label = "motion" if args.motion_split else "animation"
    train_motions = len({s.motion_id for s in train})
    test_motions = len({s.motion_id for s in test})
    print(
        f"dataset {len(samples)} anims | split {split_label}: "
        f"train {len(train)} anims / {train_motions} motions | "
        f"test {len(test)} anims / {test_motions} motions | "
        f"features {len(feature_columns)} | device {device}"
    )

    report: dict[str, dict] = {}
    for segment in SEGMENTS:
        bones = segment_bones(samples, segment)
        subsets = [
            segment_subset(scaler.transform(s.features), feature_columns, segment) for s in train
        ]
        targets = [_quat_block_view(s, bones) for s in train]
        model, losses = train_segment(subsets, targets, cfg)
        save_checkpoint(OUT_DIR, segment, model, bones, feature_columns, scaler, cfg)

        scoring = evaluate_segment_report(
            model, test, feature_columns, scaler, segment, bones, cfg.window, reduce="motion"
        )
        report[segment] = {k: v for k, v in scoring.items() if k != "bones"}
        report[segment]["bones"] = scoring["bones"]
        report[segment]["train_loss_last"] = float(losses[-1]) if losses else None
        print(
            f"[{segment}] bones={len(bones)} mean={scoring['mean_deg']:.2f} deg  "
            f"median={scoring['median_deg']:.2f}  p90={scoring['p90_deg']:.2f}  "
            f"train_loss={report[segment]['train_loss_last']:.5f}"
        )

    write_metrics(OUT_DIR, report)

    baseline_path = ROOT / "models" / "baseline" / "metrics.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        print("\ncomparison (mean deg):")
        for segment, rep in report.items():
            b = baseline[segment]["mean_deg"]
            t = rep["mean_deg"]
            print(f"  {segment:11s} baseline {b:6.2f} | temporal {t:6.2f} | {t - b:+6.2f} deg")


if __name__ == "__main__":
    main()
