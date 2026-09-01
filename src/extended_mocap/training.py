"""Reusable training/evaluation helpers shared by ``scripts/train_temporal.py``
and the interactive ``notebooks/train_temporal.ipynb``.

Keeps the notebook thin: it configures data/model and calls
:func:`train_segment` / :func:`evaluate_segment_report`; both returning plain
dicts that are easy to plot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from .evaluation import (
    SEGMENT_BONES,
    MinMax,
    load_dataset,
    score_segment_over_anims,
    split_anims,
    split_motions,
)
from .temporal import TemporalModel, covered_mask, predict_windows

SEGMENTS = ["body", "left_hand", "right_hand"]


@dataclass
class TrainConfig:
    window: int = 6
    hidden: int = 128
    layers: int = 2
    epochs: int = 30
    stride: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    verbose: bool = True


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_batches(subsets, targets, window, stride, rng):
    """Yield (X (B,L,F), y (B,O)) mini-batches over training windows.

    Deterministic per call: each animation contributes every stride-th central
    window. ``rng`` is used only to reorder animations for a single pass.
    """
    anims = list(range(len(subsets)))
    rng.shuffle(anims)
    for ai in anims:
        X = subsets[ai]
        y = targets[ai]
        t = X.shape[0]
        if t <= 2 * window:
            continue
        centers = np.arange(window, t - window, stride)
        if centers.size == 0:
            continue
        offsets = np.arange(-window, window + 1)
        wins = centers[:, None] + offsets[None, :]
        for i in range(0, centers.size, 256):
            c = centers[i : i + 256]
            yield X[wins[i : i + 256]].astype(np.float32), y[c].astype(np.float32)


def train_segment(
    subsets: list[np.ndarray],
    targets: list[np.ndarray],
    cfg: TrainConfig,
) -> tuple[TemporalModel, list[float]]:
    """Train a GRU window model for one segment.

    Returns ``(model, per_step_losses)``. ``per_step_losses`` records one float
    per optimizer step; per-epoch array can be derived by ``np.mean`` over
    chunks (helper :func:`epoch_average`).
    """
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    in_dim = subsets[0].shape[1]
    out_dim = targets[0].shape[1]
    model = TemporalModel(in_dim, out_dim, cfg.hidden, cfg.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        Xb, yb = [], []
        epoch_loss = []
        for X, y in _make_batches(subsets, targets, cfg.window, cfg.stride, rng):
            Xb.append(X)
            yb.append(y)
            if sum(x.shape[0] for x in Xb) >= cfg.batch_size:
                batch = np.concatenate(Xb)
                yb_arr = np.concatenate(yb)
                Xb, yb = [], []
                loss = loss_fn(
                    model(torch.from_numpy(batch).to(device)),
                    torch.from_numpy(yb_arr).to(device),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lv = loss.item()
                losses.append(lv)
                epoch_loss.append(lv)
        if cfg.verbose:
            if epoch_loss:
                print(
                    f"  epoch {epoch}/{cfg.epochs}: loss={np.mean(epoch_loss):.5f} "
                    f"steps={len(epoch_loss)}"
                )
            else:
                print(f"  epoch {epoch}/{cfg.epochs}: no batches (too few frames)")
    model.to("cpu")
    return model, losses


def epoch_average(losses: list[float], assets_per_epoch: int) -> np.ndarray:
    """Mean loss per epoch from the per-step list.

    ``assets_per_epoch`` is the number of optimizer steps per epoch — pass
    ``None`` to fall back to a fixed number of equal chunks.
    """
    if assets_per_epoch is None:
        return np.array(losses)
    n = len(losses) // assets_per_epoch
    return np.array(
        [np.mean(losses[i * assets_per_epoch : (i + 1) * assets_per_epoch]) for i in range(n)]
    )


def load_data(
    mediapipe_dir: str,
    mocap_dir: str,
    cache_dir: str | None,
    cfg: TrainConfig,
    motion_split: bool = True,
):
    """Load dataset + MinMax scaler (fit on train frames only) + train/test.

    ``motion_split`` splits by unique motion instead of by animation, so a
    retargeted repeat never leaks the same movement into test.
    """
    samples, feature_columns = load_dataset(mediapipe_dir, mocap_dir, cache_dir)
    splitter = split_motions if motion_split else split_anims
    train, test = splitter(samples, test_size=cfg.test_size, seed=cfg.seed)
    scaler = MinMax().fit(np.vstack([s.features for s in train]).astype(np.float64))
    return samples, feature_columns, train, test, scaler


def segment_bones(samples, segment: str) -> list[str]:
    return sorted(SEGMENT_BONES[segment] & set(samples[0].bones))


def evaluate_segment_report(
    model: TemporalModel,
    test,
    feature_columns: list[str],
    scaler: MinMax,
    segment: str,
    bones: list[str],
    window: int,
    reduce: str = "frames",
) -> dict:
    """Run windowed prediction + rotation-angle scoring for one segment."""
    predictor = lambda x, m=model: predict_windows(m, x, window)
    coverage = lambda n, w=window: covered_mask(n, w)
    return score_segment_over_anims(
        predictor,
        test,
        feature_columns,
        scaler,
        segment,
        bones,
        covered=coverage,
        reduce=reduce,
    )


def save_checkpoint(
    out_dir: Path,
    segment: str,
    model: TemporalModel,
    bones: list[str],
    feature_columns: list[str],
    scaler: MinMax,
    cfg: TrainConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": {
                "input_dim": model.gru.input_size,
                "output_dim": model.head[-1].out_features,
                "hidden": cfg.hidden,
                "layers": cfg.layers,
                "window": cfg.window,
            },
            "state_dict": model.state_dict(),
        },
        out_dir / f"{segment}.pt",
    )
    (out_dir / f"{segment}_bones.json").write_text(json.dumps(bones))
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns))
    scaler.save(str(out_dir / "scaler.npz"))
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "hidden": cfg.hidden,
                "layers": cfg.layers,
                "window": cfg.window,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "seed": cfg.seed,
            }
        )
    )


def write_metrics(out_dir: Path, report: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))
