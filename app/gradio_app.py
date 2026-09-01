"""Gradio demo: uploaded video -> animated 3D skeleton.

Run with:
    uv run python app/gradio_app.py
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
from visualization import build_skeleton

from extended_mocap.extractor import MediapipeExtractor

_EXTRACTOR = MediapipeExtractor()


def set_pose_task(path: str) -> None:
    global _EXTRACTOR
    _EXTRACTOR = MediapipeExtractor(pose_task_path=path)


def predict_plot(video_path: str, stride: int, max_frames: int):
    if not video_path:
        raise gr.Error("Upload a video first.")

    landmarks, connections = _EXTRACTOR.extract_pose_landmarks(
        video_path,
        stride=stride,
        max_frames=max_frames,
    )
    if len(landmarks) == 0:
        raise gr.Error("No person detected in the video.")

    fig = build_skeleton(landmarks, connections)
    return fig


def predict_csv(video_path: str):
    """Extract the engineered feature CSV (used as model input)."""
    if not video_path:
        raise gr.Error("Upload a video first.")
    with tempfile.TemporaryDirectory(prefix="em_") as td:
        out = os.path.join(td, "features.csv")
        _EXTRACTOR.run(video_path, output_csv_path=out)
        return out


def demo() -> gr.Blocks:
    with gr.Blocks(title="Extended Mocap") as demo:
        gr.Markdown(
            """
# Extended Mocap

Video → animated 3D skeleton, using MediaPipe pose landmarks.

**Pipeline:** webcam/phone video → MediaPipe pose landmarks → bone-rotation
quaternions (via trained ML models) → drive a CC_Base character
in Blender.

This demo shows the raw 3D pose skeleton extracted from your video.
        """
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Input video", sources=["upload"])
                stride = gr.Slider(
                    1, 10, value=1, step=1, label="Frame stride (higher = faster, skip frames)"
                )
                max_frames = gr.Slider(30, 3000, value=1000, step=10, label="Max frames to process")
                btn_plot = gr.Button("Extract 3D skeleton")
                btn_csv = gr.Button("Download feature CSV")
            with gr.Column(scale=2):
                plot = gr.Plot(label="3D skeleton")
                file_out = gr.File(label="Feature CSV")

        btn_plot.click(
            predict_plot,
            inputs=[video, stride, max_frames],
            outputs=[plot],
        )
        btn_csv.click(
            predict_csv,
            inputs=[video],
            outputs=[file_out],
        )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo().launch(server_name="0.0.0.0", server_port=port)
