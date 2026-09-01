"""ExtendedMocap: video-to-3D skeleton animation pipeline.

Converts plain video (webcam, phone) into bone-rotation quaternion
animation data that can drive a 3D character in Blender.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["MediapipeExtractor", "MocapInferer"]


def __getattr__(name):
    # Lazy imports so the core package stays light and the pure-torch
    # ``MocapInferer`` can be imported without MediaPipe/OpenCV installed.
    if name == "MocapInferer":
        from .inference import MocapInferer as _impl

        return _impl
    if name == "MediapipeExtractor":
        from .extractor import MediapipeExtractor as _impl

        return _impl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
