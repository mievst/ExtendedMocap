"""Temporal (GRU) model predicting center-frame bone quaternions from a window
of scaled feature frames, plus a sliding-window predictor for evaluation.

Per-frame MLPs (as in the original notebooks) see no temporal context, so they
flicker and can't recover from MediaPipe tracking noise. Feeding the model a
window ``[-w, +w]`` around the target frame gives it velocity/continuity cues;
the evaluation masks boundary frames so every model is scored on the same
frames.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemporalModel(nn.Module):
    """GRU encoder over ``2*window+1`` feature frames, quaternion head."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, L, H)
        return self.head(out[:, -1])  # (B, out)


def sliding_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Stack of centered windows ``X[t-w : t+w+1]`` for interior times ``t``.

    Returns ``(T - 2*w, 2*w+1, F)`` float32.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0] - 2 * window
    offsets = np.arange(-window, window + 1)  # [t-w .. t+w]
    idx = np.arange(n)[:, None] + offsets[None, :]
    return X[idx]


def predict_windows(model: TemporalModel, X: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window prediction over one animation.

    Returns ``(T, out_dim)`` float32 with NaN rows at boundary frames (no full
    window exists), which ``score_segment_over_anims(..., covered=...)`` drops.
    """
    if X.shape[0] <= 2 * window:
        return np.full((X.shape[0], model.head[-1].out_features), np.nan, dtype=np.float32)
    wins = sliding_windows(X, window)  # (n, L, F)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, wins.shape[0], 256):
            batch = torch.from_numpy(wins[i : i + 256])
            preds.append(model(batch).numpy())
    pred = np.concatenate(preds, axis=0)
    out = np.full((X.shape[0], pred.shape[1]), np.nan, dtype=np.float32)
    out[window : window + pred.shape[0]] = pred
    return out


def covered_mask(n_frames: int, window: int) -> np.ndarray:
    """Mask of frames that have a full context window (interior frames)."""
    mask = np.zeros(n_frames, dtype=bool)
    if n_frames > 2 * window:
        mask[window : n_frames - window] = True
    return mask
