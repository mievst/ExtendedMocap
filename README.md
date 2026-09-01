# Extended Mocap

Video → 3D skeleton animation pipeline. Turn plain webcam/phone footage into
bone-rotation quaternions that drive a CC_Base (Mixamo) character in Blender.

**Stack:** MediaPipe pose/hand landmarks → engineered features →
per-segment neural networks → Blender retarget + render.

## Pipeline

```
video ──▶ MediaPipe landmarks ──▶ feature CSV ──▶ NN (quaternions) ──▶ Blender CGI
```

1. **Extract** — `MediapipeExtractor` reads a video, tracks body + hands,
   and builds engineered features (pairwise distances, joint angles,
   body–hand interactions).
2. **Infer** — `MocapInferer` runs a trained `NeuralNetwork` per body
   segment and outputs bone-rotation quaternions.
3. **Retarget** — Blender scripts copy rotations from Mixamo armatures onto
   a CC_Base rig, render to MP4, and export per-frame CSVs.

See [`PLAN.md`](PLAN.md) for the full project plan.

## Install

```bash
uv sync                          # core pipeline
uv sync --extra app              # + Gradio demo
uv sync --extra dev              # + tests/lint
```

Requirements: Python ≥ 3.10. The inference path needs PyTorch (CUDA build,
see `pyproject.toml`); the video demo needs MediaPipe and OpenCV.

## Quick start

Extract features from a video:

```bash
uv run extended-mocap extract input.mp4 -o features.csv
```

Run the Gradio 3D skeleton demo:

```bash
uv run --extra app python app/gradio_app.py
```

## Training & honest evaluation

The dataset is split **by motion**, not by animation, so NLA-layer repeats of
the same motion never leak across train/test. Train/test splits and the
manifest are produced from the source `.blend` files.

Build the baseline / temporal models (user only — Cuda ROM required):

```bash
uv run python scripts/build_baseline.py --epochs 50
uv run python scripts/train_temporal.py --epochs 30 --window 6 --stride 3 --device cuda
```

Both scripts use the motion split by default (`--motion-split` /
`--no-motion-split`) and report per-motion averaged metrics
(`reduce="motion"`). Training is heavy on 4 GB GPUs; only the GPU has CUDA.

## Data pipeline (regenerate from Blender sources)

CSV features and quaternions are **derived** outputs. The source of truth is
six `.blend` files at the repo root (`mocap.blend`, `mocap_extended`,
`mocap_accurig`, `mocap_mixamo`, `mocap_mocca1`, `data/mocap.blend`). They are
large and not committed; regenerate the CSVs with headless Blender 5.0.

**Step 1 — extract quaternions** from every blend, one armature at a time:

```bash
uv run python scripts/run_extraction.py
```

Writes `data/mocap/csv/{armature}.{motion}.csv` and `data/blend_manifest.json`.

**Step 2 — render MP4** (optional, if rendered clips are missing):

```bash
uv run python scripts/build_dataset.py \
    --blend-manifest data/blend_manifest.json --render --base data/base.blend
```

**Step 3 — MediaPipe features** (optional, from rendered MP4s):

```bash
uv run python scripts/build_dataset.py \
    --blend-manifest data/blend_manifest.json --mediapipe
```

**Step 4 — assemble the dataset manifest** (no Blender needed):

```bash
uv run python scripts/build_dataset.py --blend-manifest data/blend_manifest.json
```

Writes `data/motion_manifest.json` with correct `motion_id` (the normalized
motion name, not the frame range).

## Modules

| Path | Purpose |
|------|---------|
| `src/extended_mocap/extractor.py` | MediaPipe feature extraction |
| `src/extended_mocap/inference.py` | Quaternion prediction via `MocapInferer` |
| `src/extended_mocap/models.py` | `NeuralNetwork` |
| `src/extended_mocap/evaluation.py` | Dataset loading, motion split, per-motion metrics |
| `src/extended_mocap/training.py` | Data loading + baseline/temporal training helpers |
| `src/extended_mocap/temporal.py` | GRU/Conv temporal model |
| `src/extended_mocap/blender/` | Blender (bpy) retarget/render/extract scripts |
| `scripts/` | CLI + data-pipeline entry points |
| `app/gradio_app.py` | Interactive 3D skeleton demo |

## Blender

The Blender scripts import `bpy` and must run inside Blender's Python.
Bone mapping lives in `configs/mixamo_to_ccbase.json`.

## Docker

```bash
cd docker
docker compose up --build
# open http://localhost:7860
```

## Models

Trained checkpoints are **not** committed — train locally (`scripts/build_baseline.py`, `scripts/train_temporal.py`) or download published checkpoints from HuggingFace. Models land in `models/`, which is gitignored. The segment→checkpoint mapping is configurable — see `MocapInferer` `model_config`:

```json
[{"segment": "body", "path": "models/baseline/body.pt"},
 {"segment": "left_hand", "path": "models/baseline/left_hand.pt"},
 {"segment": "right_hand", "path": "models/baseline/right_hand.pt"}]
```

MediaPipe `.task` models (`pose_landmarker_heavy.task`, `hand_landmarker.task`) are **not** committed either — `MediapipeExtractor` downloads them on first use (see `src/extended_mocap/extractor.py`).

## Dataset

Paired MediaPipe-feature and quaternion CSVs are published to HuggingFace —
see `scripts/upload_to_hf.py`.

## License

MIT. See [LICENSE](LICENSE).
