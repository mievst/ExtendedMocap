"""Dataset loading, feature/target subsetting, and evaluation metrics for the
MediaPipe -> quaternion pipeline.

The notebook pipeline trained one MLP per segment on hand-picked feature
columns, then dumped the checkpoints without recording which output index maps
to which bone. ``evaluation.py`` defines a *canonical* column order (bone names
fixed and persisted as JSON) so models can be trained, evaluated and deployed
without the set-iteration ambiguity of the original notebooks.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .models import NeuralNetwork
from .utils import BODY_BONES, LEFT_HAND_BONES, RIGHT_HAND_BONES

SEGMENT_BONES: dict[str, set[str]] = {
    "left_hand": set(LEFT_HAND_BONES),
    "right_hand": set(RIGHT_HAND_BONES),
    "body": set(BODY_BONES),
}


def segment_feature_columns(mp_df: pd.DataFrame, segment: str) -> list[str]:
    """Replicate the notebook's per-segment feature selection.

    ``body`` takes every column whose name contains ``"body"`` and not
    ``"hand"``; hand segments take every column whose name contains the
    ``segment`` substring (``"left_hand"`` / ``"right_hand"``), which the
    extractor guarantees is a stable naming scheme.
    """
    if segment == "body":
        return [c for c in mp_df.columns if "body" in c and "hand" not in c]
    return list(mp_df.filter(like=segment).columns)


def bones_of(quat_df: pd.DataFrame, segment: str) -> list[str]:
    """Canonical sorted bone names for a segment, restricted to bones that
    actually appear in the ground-truth DataFrame.
    """
    names = SEGMENT_BONES[segment]
    cols = {str(c).split(".")[0] for c in quat_df.columns if ".Q" in str(c)}
    present = sorted(names & cols)
    return present


def target_columns(bones: list[str]) -> list[str]:
    """Ordered ``bone.Q0..Q3`` columns for ``bones`` (block-per-bone layout)."""
    return [f"{b}.Q{q}" for b in bones for q in range(4)]


@dataclass
class Sample:
    """One animation: raw feature frame and ground-truth quaternions.

    ``motion_id`` identifies the underlying motion. All clips that are
    retargeted repeats of the same motion share one ``motion_id`` and must
    stay in the same train/test split (see :func:`split_motions`).
    """

    name: str
    features: np.ndarray  # (T, F) float32
    quats: np.ndarray  # (T, 4*N) float32, canonical bone order
    bones: list[str]
    motion_id: str | None = None


def motion_id_of(mediapipe_path: str) -> str:
    """Derive the motion id from a MediaPipe feature CSV filename.

    MediaPipe CSVs are named ``<prefix>.<range>.csv`` (e.g. ``test_031.0000-0461.csv``)
    where the ``<range>`` groups every clip retargeted from the same source
    motion. Falling back to the stripped name keeps the id well-defined for
    CSVs that lack a range suffix.
    """
    fname = os.path.basename(mediapipe_path)
    stem = fname.removesuffix(".csv")
    if "." in stem:
        return stem.split(".", 1)[1]
    return stem


def _split_prefix(fname: str) -> str:
    return fname.split(".")[0] if fname.endswith(".csv") else fname


def _union_feature_schema(mediapipe_dir: str) -> list[str]:
    """Union of feature-column schemas across all animation CSVs.

    Older extraction runs produced fewer columns (e.g. hand features absent),
    so aligning every animation to one shared schema avoids shape mismatches;
    missing columns are treated as "hand not detected" (zero-filled).
    """
    header_cache: dict[int, list[str]] = {}
    files = [f for f in os.listdir(mediapipe_dir) if f.endswith(".csv")]
    for fname in files:
        header = pd.read_csv(os.path.join(mediapipe_dir, fname), nrows=0).columns
        key = len(header)
        if key not in header_cache:
            header_cache[key] = [str(c) for c in header]
    if len(header_cache) == 1:
        return next(iter(header_cache.values()))
    # union: keep the widest schema order, append any extra names from narrower ones
    order: list[str] = []
    for key in sorted(header_cache, key=lambda k: len(header_cache[k]), reverse=True):
        for col in header_cache[key]:
            if col not in order:
                order.append(col)
    return order


def load_dataset(
    mediapipe_dir: str,
    mocap_dir: str,
    cachedir: str | None = None,
    smooth_window: int = 0,
) -> tuple[list[Sample], list[str]]:
    """Load feature/quaternion pairs aligned by common filename prefix.

    Returns ``(samples, feature_columns)`` where ``feature_columns`` is the
    union feature schema shared by every animation. If ``cachedir`` is given,
    parsed arrays are cached as ``.npz`` for faster reloads.
    """
    mp_map = {
        _split_prefix(f): os.path.join(mediapipe_dir, f)
        for f in os.listdir(mediapipe_dir)
        if f.endswith(".csv")
    }
    q_map = {
        _split_prefix(f): os.path.join(mocap_dir, f)
        for f in os.listdir(mocap_dir)
        if f.endswith(".csv")
    }
    keys = sorted(set(mp_map) & set(q_map))
    if not keys:
        raise FileNotFoundError(
            f"No matching mediapipe/mocap pairs between {mediapipe_dir} and {mocap_dir}."
        )
    if cachedir:
        os.makedirs(cachedir, exist_ok=True)

    feature_columns = _union_feature_schema(mediapipe_dir)
    samples: list[Sample] = []
    for i, key in enumerate(keys):
        cache_path = os.path.join(cachedir, f"{key}.npz") if cachedir else None
        if cache_path and os.path.isfile(cache_path):
            with np.load(cache_path, allow_pickle=True) as arr:
                feats = arr["features"]
                quats = arr["quats"]
                bones = list(arr["bones"])
        else:
            mp_df = pd.read_csv(mp_map[key])
            quat_df = pd.read_csv(q_map[key])
            if len(mp_df) != len(quat_df):
                n = min(len(mp_df), len(quat_df))
                mp_df = mp_df.iloc[:n]
                quat_df = quat_df.iloc[:n]
            if smooth_window:
                mp_df = pd.DataFrame(
                    savgol_filter(mp_df, smooth_window, 2, axis=0),
                    columns=mp_df.columns,
                    index=mp_df.index,
                )
            mp_df = mp_df.reindex(columns=feature_columns).fillna(0.0)
            bones = sorted({b for seg in SEGMENT_BONES for b in bones_of(quat_df, seg)})
            cols = target_columns(bones)
            if not all(c in quat_df.columns for c in cols):
                raise ValueError(f"{key}: expected quaternion columns missing.")
            feats = mp_df.to_numpy(dtype=np.float32)
            quats = quat_df[cols].to_numpy(dtype=np.float32)
            if cache_path:
                np.savez(
                    cache_path,
                    features=feats,
                    quats=quats,
                    feature_columns=np.array(feature_columns, dtype=object),
                    bones=np.array(bones, dtype=object),
                )
        if len(feats) < 2:
            raise ValueError(f"{key}: animation has fewer than 2 frames.")
        samples.append(
            Sample(
                name=key,
                features=feats,
                quats=quats,
                bones=bones,
                motion_id=motion_id_of(mp_map[key]),
            )
        )

    return samples, feature_columns


def split_anims(
    samples: list[Sample], test_size: float = 0.2, seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/test split by *animation* (no frame leakage)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, round(len(samples) * test_size))
    test_idx = set(idx[:n_test].tolist())
    train = [s for i, s in enumerate(samples) if i not in test_idx]
    test = [s for i, s in enumerate(samples) if i in test_idx]
    return train, test


def split_motions(
    samples: list[Sample], test_size: float = 0.2, seed: int = 42, motion_key: Callable | None = None
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/test split by *motion* (no motion leakage).

    All clips sharing a ``motion_id`` stay together: a motion and its
    retargeted repeats are never split across train and test. Without this the
    test set silently re-scores motions the model already saw in training.

    ``motion_key`` overrides the grouping (defaults to ``s.motion_id``); clips
    without a motion id fall back to being grouped by name.
    """
    if motion_key is None:
        motion_key = lambda s: s.motion_id or s.name

    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(motion_key(s), []).append(s)

    motion_ids = sorted(groups)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(motion_ids))
    n_test = max(1, round(len(motion_ids) * test_size))
    test_ids = {motion_ids[i] for i in order[:n_test].tolist()}
    train = [s for mid, reps in groups.items() for s in reps if mid not in test_ids]
    test = [s for mid, reps in groups.items() for s in reps if mid in test_ids]
    return train, test


class MinMax:
    """Column-wise min-max scaler compatible with sklearn's MinMaxScaler
    (constant columns map to 0). Persisted as a plain numpy npz.
    """

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> MinMax:
        X = X.astype(np.float64)
        self.min_ = X.min(axis=0)
        data_range = X.max(axis=0) - self.min_
        self.scale_ = np.where(data_range > 0, data_range, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) - self.min_) / self.scale_

    def save(self, path: str) -> None:
        np.savez(path, min_=self.min_, scale_=self.scale_)

    @classmethod
    def load(cls, path: str) -> MinMax:
        with np.load(path) as arr:
            scaler = cls()
            scaler.min_ = arr["min_"]
            scaler.scale_ = arr["scale_"]
            return scaler


def angular_error_deg(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Rotation-angle error between two Nx4 quaternion arrays, in degrees.

    Accounts for quaternion sign ambiguity (q and -q encode the same rotation).
    Returns shape (N,).
    """
    p = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-12)
    t = q_true / (np.linalg.norm(q_true, axis=1, keepdims=True) + 1e-12)
    dot = np.abs((p * t).sum(axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def evaluate_segment(pred: np.ndarray, true: np.ndarray, bones: list[str]) -> dict:
    """Per-bone and aggregate rotation error report.

    ``pred``/``true`` are (T, 4*len(bones)).
    """
    n_bones = len(bones)
    err = np.stack(
        [
            angular_error_deg(pred[:, 4 * b : 4 * b + 4], true[:, 4 * b : 4 * b + 4])
            for b in range(n_bones)
        ],
        axis=0,
    )  # (bones, frames)
    per_bone = {}
    for b, name in enumerate(bones):
        per_bone[name] = {
            "mean_deg": float(np.nanmean(err[b])),
            "median_deg": float(np.nanmedian(err[b])),
        }
    return {
        "bones": per_bone,
        "mean_deg": float(np.nanmean(err)),
        "median_deg": float(np.nanmedian(err)),
        "p90_deg": float(np.nanpercentile(err, 90)),
    }


def predict_mlp(model: NeuralNetwork, X: np.ndarray) -> np.ndarray:
    """Batch prediction in float32 on CPU without autograd."""
    import torch

    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    with torch.no_grad():
        return model(x).numpy()


def segment_subset(scaled: np.ndarray, feature_columns: list[str], segment: str) -> np.ndarray:
    """Columns of the (already scaled) feature frame that feed ``segment``,
    in model input order (see ``segment_feature_columns``).
    """
    idx = segment_feature_columns(pd.DataFrame(scaled, columns=feature_columns), segment)
    lookup = {c: i for i, c in enumerate(feature_columns)}
    return scaled[:, [lookup[c] for c in idx]]


def score_segment_over_anims(
    predictor,
    samples: list[Sample],
    feature_columns: list[str],
    scaler: MinMax,
    segment: str,
    bones: list[str],
    covered: callable | None = None,
    reduce: str = "frames",
) -> dict:
    """Score ``predictor`` on every animation in ``samples`` for one segment.

    ``predictor(scaled_subset: (T,F)) -> (T, 4*len(bones))`` is called once per
    animation with the MinMax-scaled feature subset. ``covered(n_frames)`` may
    return a boolean mask limiting evaluated frames (used by windowed models so
    boundary frames are dropped identically for every predictor).

    ``reduce`` controls how the headline stats are aggregated over samples:

    * ``"frames"`` (default) stacks every evaluated frame and computes global
      per-frame stats (the legacy behaviour).
    * ``"motion"`` weights each motion-group equally: every motion contributes
      the mean ``per_anim`` error of its repeat clips, so duplicates never
      inflate the test score. Median/p90 are taken over that per-motion
      distribution.

    Returns a report dict with per-bone stats plus ``per_anim`` mean errors
    and (for ``reduce="motion"``) a ``per_motion`` breakdown.
    """
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    per_anim: dict[str, float] = {}
    for s in samples:
        subset = segment_subset(scaler.transform(s.features), feature_columns, segment)
        pred = predictor(subset)
        mask = (
            covered(s.features.shape[0])
            if covered is not None
            else np.ones(s.features.shape[0], dtype=bool)
        )
        if mask.sum() == 0:
            continue
        true = _quat_block_view(s, bones)
        all_pred.append(pred[mask])
        all_true.append(true[mask])
        err = evaluate_segment(pred[mask], true[mask], bones)
        per_anim[s.name] = err["mean_deg"]
    combined = evaluate_segment(np.vstack(all_pred), np.vstack(all_true), bones)
    combined["per_anim"] = per_anim

    if reduce == "motion":
        per_motion: dict[str, list[float]] = {}
        for s in samples:
            key = s.motion_id or s.name
            if s.name in per_anim:
                per_motion.setdefault(key, []).append(per_anim[s.name])
        motion_mean = {k: float(np.mean(v)) for k, v in per_motion.items()}
        vals = np.array(list(motion_mean.values()), dtype=float)
        combined["mean_deg"] = float(vals.mean()) if vals.size else 0.0
        combined["median_deg"] = float(np.median(vals)) if vals.size else 0.0
        combined["p90_deg"] = float(np.percentile(vals, 90)) if vals.size else 0.0
        combined["per_motion"] = motion_mean
    return combined


def _quat_block_view(sample: Sample, bones: list[str]) -> np.ndarray:
    """(T, 4*len(bones)) view of ground truth for ``bones`` in union order."""
    positions = [sample.bones.index(b) for b in bones]
    idx = np.concatenate([np.arange(4 * p, 4 * p + 4) for p in positions])
    return sample.quats[:, idx]
