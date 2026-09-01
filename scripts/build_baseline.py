"""Train honest per-segment MLP baselines and evaluate them on a held-out
animation-level split.

Unlike the original notebooks — which leaked frames between train and test and
never recorded the model-output -> bone mapping — this script:

* splits by *animation* (no frame leakage),
* fixes a canonical bone order (persisted to JSON),
* applies the MinMax scaling the notebooks used for feature input,
* reports rotation-angle error in degrees per bone / segment / animation.

Usage:  uv run python scripts/build_baseline.py [--epochs 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from extended_mocap.evaluation import (
    SEGMENT_BONES,
    MinMax,
    _quat_block_view,
    load_dataset,
    predict_mlp,
    score_segment_over_anims,
    segment_subset,
    split_motions,
)
from extended_mocap.models import NeuralNetwork

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "models" / "baseline"
SEGMENTS = ["body", "left_hand", "right_hand"]


def train_mlp(X: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> NeuralNetwork:
    """Re-implementation of the notebook's loop: Adam, StepLR, MSE, batch 1000."""
    import torch

    torch.manual_seed(seed)
    model = NeuralNetwork(input_dim=X.shape[1], output_dim=y.shape[1])
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        ),
        batch_size=1000,
        shuffle=True,
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples, feature_columns = load_dataset(
        str(ROOT / "data" / "mediapipe" / "csv"),
        str(ROOT / "data" / "mocap" / "csv"),
        str(CACHE),
    )
    train, test = split_motions(samples, test_size=args.test_size, seed=args.seed)
    train_motions = len({s.motion_id for s in train})
    test_motions = len({s.motion_id for s in test})
    union = samples[0].bones
    print(
        f"dataset {len(samples)} anims / {len({s.motion_id for s in samples})} motions | "
        f"train {len(train)} anims / {train_motions} motions | "
        f"test {len(test)} anims / {test_motions} motions | "
        f"features {len(feature_columns)} | bones {len(union)}"
    )

    scaler = MinMax().fit(np.vstack([s.features for s in train]).astype(np.float64))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler.save(str(OUT_DIR / "scaler.npz"))
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(feature_columns))

    report: dict[str, dict] = {}

    for segment in SEGMENTS:
        bones = sorted(SEGMENT_BONES[segment] & set(union))
        Xs, ys = [], []
        for s in train:
            Xs.append(segment_subset(scaler.transform(s.features), feature_columns, segment))
            ys.append(_quat_block_view(s, bones))
        X_tr = np.vstack(Xs)
        y_tr = np.vstack(ys)
        model = train_mlp(X_tr, y_tr, args.epochs, args.seed)
        torch.save(
            {
                "arch": {"input_dim": X_tr.shape[1], "output_dim": y_tr.shape[1]},
                "state_dict": model.state_dict(),
            },
            OUT_DIR / f"{segment}.pt",
        )
        (OUT_DIR / f"{segment}_bones.json").write_text(json.dumps(bones))

        scoring = score_segment_over_anims(
            lambda x, m=model: predict_mlp(m, x),
            test,
            feature_columns,
            scaler,
            segment,
            bones,
            reduce="motion",
        )
        report[segment] = {k: v for k, v in scoring.items() if k != "bones"}
        report[segment]["bones"] = scoring["bones"]
        errs = list(scoring["per_motion"].values())
        print(
            f"[{segment}] bones={len(bones)} mean={scoring['mean_deg']:.2f} deg  "
            f"median={scoring['median_deg']:.2f}  p90={scoring['p90_deg']:.2f}  "
            f"motion_std={float(np.std(errs)):.2f}"
        )

    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    worst = {}
    for segment, rep in report.items():
        worst[segment] = sorted((v["mean_deg"], b) for b, v in rep["bones"].items())[-5:]
    print("\nworst bones:")
    for segment, rows in worst.items():
        for err, b in reversed(rows):
            print(f"  {segment:11s} {b:28s} {err:6.2f} deg")


if __name__ == "__main__":
    main()
