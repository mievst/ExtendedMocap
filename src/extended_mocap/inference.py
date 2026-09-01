"""Inference: video -> bone-rotation quaternions via trained neural networks.

``MocapInferer`` ties the pipeline together: it needs a model per body segment
(``body``, ``left_hand``, ``right_hand``).

Two checkpoint formats are supported transparently:

* legacy notebook checkpoints — full ``torch.save(model)`` pickles of a
  ``NeuralNetwork`` saved as ``__main__.NeuralNetwork`` (no recorded column
  order, so predictions can no longer be labelled per-bone reliably);
* current checkpoints — a dict ``{"arch": {...}, "state_dict": ...}`` written
  by ``scripts/build_baseline.py`` / ``scripts/train_temporal.py`` together
  with a MinMax scaler (``scaler.npz``), a ``feature_columns.json`` and a
  ``<segment>_bones.json`` listing the output-bone order. These are exact.

The default model config prefers the retrained baseline models; pass an
explicit ``model_config`` to override.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from .evaluation import MinMax
from .models import NeuralNetwork
from .temporal import TemporalModel


def load_model_any(path: str) -> torch.nn.Module:
    """Load a checkpoint in either legacy pickle or state-dict bundle format."""
    import sys

    _main = sys.modules["__main__"]
    if not hasattr(_main, "NeuralNetwork"):
        _main.NeuralNetwork = NeuralNetwork
    _main.TemporalModel = TemporalModel  # legacy temporal pickles, if any

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and "arch" in obj:
        arch = obj["arch"]
        if "window" in arch:
            model = TemporalModel(
                arch["input_dim"],
                arch["output_dim"],
                hidden=arch.get("hidden", 128),
                layers=arch.get("layers", 2),
            )
        else:
            model = NeuralNetwork(arch["input_dim"], arch["output_dim"])
        model.load_state_dict(obj["state_dict"])
        model.eval()
        return model
    model = obj
    model.eval()
    return model


def _default_model_config() -> dict:
    """Point to retrained baselines when present, else legacy pickles."""
    baseline_dir = "models/baseline"
    if os.path.isdir(baseline_dir) and all(
        os.path.isfile(os.path.join(baseline_dir, f"{s}.pt"))
        for s in ("body", "left_hand", "right_hand")
    ):
        return {
            s: {"path": os.path.join(baseline_dir, f"{s}.pt")}
            for s in ("body", "left_hand", "right_hand")
        }
    return {
        "left_hand": {"path": "models/model_01.pt"},
        "right_hand": {"path": "models/model_02.pt"},
        "body": {"path": "models/model_03.pt"},
    }


DEFAULT_MODEL_CONFIG = _default_model_config()


def load_model_config(path: str) -> dict:
    """Load a model config from JSON (list of {segment, path})."""
    with open(path) as f:
        raw = json.load(f)
    return {item["segment"]: {"path": item["path"]} for item in raw}


class MocapInferer:
    """Predict bone-rotation quaternions from a MediaPipe feature frame.

    Parameters
    ----------
    model_config:
        Mapping ``segment -> {"path": str}``; optional per-segment ``"bones"``
        (ordered output-bone names) and ``"features"`` (explicit input column
        list, overriding the notebook's substring selection).
    model_dir:
        Directory containing ``feature_columns.json``, ``scaler.npz`` and
        ``<segment>_bones.json`` when ``model_config`` paths are relative.
    smooth_window:
        Savitzky-Golay window applied to inputs then outputs for denoising.
    """

    def __init__(
        self,
        model_config: dict | None = None,
        device: str = "cpu",
        smooth_window: int = 10,
        model_dir: str | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        if device and torch.cuda.is_available():
            self.device = torch.device(device)
        self.smooth_window = smooth_window
        self.model_dir = model_dir
        self.feature_columns: list[str] | None = None
        self.scaler: MinMax | None = None
        self.segments = self._build_segments(model_config or DEFAULT_MODEL_CONFIG)

    @staticmethod
    def _resolve(path: str | None, model_dir: str | None) -> str | None:
        if path is None or os.path.isabs(path):
            return path
        if model_dir and os.path.isfile(os.path.join(model_dir, path)):
            return os.path.join(model_dir, path)
        return path

    def _build_segments(self, config: dict) -> dict:
        segments = {}
        for segment, spec in config.items():
            path = spec["path"]
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model for segment '{segment}' not found: {path}")
            model = load_model_any(path).to("cpu")
            model.eval()

            bones_path = spec.get("bones") or self._resolve(f"{segment}_bones.json", self.model_dir)
            if bones_path and os.path.isfile(bones_path):
                with open(bones_path) as f:
                    bones = json.load(f)
            else:
                bones = list(spec.get("bones", []))

            if self.feature_columns is None:
                cols_path = self._resolve("feature_columns.json", self.model_dir)
                if cols_path and os.path.isfile(cols_path):
                    with open(cols_path) as f:
                        self.feature_columns = json.load(f)
            if self.scaler is None:
                scaler_path = self._resolve("scaler.npz", self.model_dir)
                if scaler_path and os.path.isfile(scaler_path):
                    self.scaler = MinMax.load(scaler_path)

            segments[segment] = {
                "model_": model,
                "features": list(spec.get("features", [])),
                "bones": bones,
            }
        return segments

    # ------------------------------------------------------------------ #
    # Feature masking (mirrors the notebook / evaluation.py selectors)
    # ------------------------------------------------------------------ #
    def _subset_columns(self, segment: str) -> list[str]:
        if segment == "body":
            return [c for c in self.feature_columns if "body" in c and "hand" not in c]
        return [c for c in self.feature_columns if segment in c]

    def _mask(self, scaled: pd.DataFrame) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for segment, spec in self.segments.items():
            if spec["features"]:
                avail = [c for c in spec["features"] if c in scaled.columns]
                masks[segment] = scaled[avail].to_numpy(dtype=np.float32)
            elif self.feature_columns is not None:
                cols = [c for c in self._subset_columns(segment) if c in scaled.columns]
                masks[segment] = scaled[cols].to_numpy(dtype=np.float32)
            else:
                masks[segment] = scaled.to_numpy(dtype=np.float32)
        return masks

    def predict_from_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Predict quaternions from an already-extracted feature DataFrame."""
        if self.smooth_window and len(feature_df) > self.smooth_window:
            feature_df = pd.DataFrame(
                savgol_filter(feature_df, self.smooth_window, 2, axis=0),
                columns=feature_df.columns,
                index=feature_df.index,
            )

        if self.scaler is not None and self.feature_columns is not None:
            common = [c for c in self.feature_columns if c in feature_df.columns]
            if len(common) != len(self.feature_columns):
                print(
                    f"WARNING: scaling on {len(common)}/{len(self.feature_columns)} "
                    "features; extractor schema may differ."
                )
            scaled = pd.DataFrame(
                self.scaler.transform(feature_df[common].to_numpy()),
                columns=common,
                index=feature_df.index,
            )
        else:
            scaled = feature_df

        input_masks = self._mask(scaled)
        summarized: dict[str, list[float]] = {}
        has_nan = False
        for segment, spec in self.segments.items():
            x = torch.from_numpy(np.ascontiguousarray(input_masks[segment])).to(self.device)
            with torch.no_grad():
                pred = spec["model_"](x).cpu().numpy()
            if np.isnan(pred).any():
                has_nan = True
            columns = spec["bones"] or self._infer_columns(segment, pred.shape[1])
            for b, bone in enumerate(columns):
                for q in range(4):
                    col = f"{bone}.Q{q}"
                    summarized.setdefault(col, []).extend(pred[:, 4 * b + q].tolist())

        out = pd.DataFrame(summarized)
        if not has_nan and self.smooth_window and len(out) > max(3, self.smooth_window // 2):
            out = pd.DataFrame(
                savgol_filter(out, max(3, self.smooth_window // 2), 2, axis=0),
                columns=out.columns,
                index=out.index,
            )
        return out

    def predict(self, feature_csv: str) -> pd.DataFrame:
        """Convenience wrapper reading a feature CSV produced by
        ``MediapipeExtractor.run(..., output_csv_path=...)``.
        """
        feature_df = pd.read_csv(feature_csv)
        return self.predict_from_features(feature_df)

    def predict_video(self, video_path: str, extractor) -> pd.DataFrame:
        """One-call: extract features from ``video_path`` then predict."""
        features = extractor.run(video_path)
        return self.predict_from_features(pd.DataFrame(features))

    @staticmethod
    def _infer_columns(segment: str, n: int) -> list[str]:
        if n % 4 != 0:
            return [f"{segment}_{i}" for i in range(n)]
        bones = n // 4
        return [f"{segment}_{b}.Q{q}" for b in range(bones) for q in range(4)]
