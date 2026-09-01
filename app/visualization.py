"""Plotly 3D skeleton visualisation for the Gradio demo."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _segment_xyz(pts: np.ndarray, conn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 1-D x/y/z vectors with NaN breaks between bone segments.

    NaN is a line break in Plotly, so each bone is drawn as its own
    disconnected segment.
    """
    n = len(conn)
    x = np.empty(n * 3, dtype=float)
    y = np.empty(n * 3, dtype=float)
    z = np.empty(n * 3, dtype=float)
    for i, (a, b) in enumerate(conn):
        x[i * 3] = pts[a, 0]
        y[i * 3] = pts[a, 1]
        z[i * 3] = pts[a, 2]
        x[i * 3 + 1] = pts[b, 0]
        y[i * 3 + 1] = pts[b, 1]
        z[i * 3 + 1] = pts[b, 2]
        x[i * 3 + 2] = np.nan
        y[i * 3 + 2] = np.nan
        z[i * 3 + 2] = np.nan
    return x, y, z


def build_skeleton(
    landmarks: np.ndarray,
    connections: list[tuple[int, int]],
    stride_step: int = 1,
) -> go.Figure:
    """Return an animated 3D figure of the pose skeleton.

    ``landmarks`` has shape ``(T, 33, 3)`` (world-space points). A slider
    steps through detected frames. ``stride_step`` subsamples frames to keep
    the figure small.
    """
    landmarks = landmarks[::stride_step]
    t, _, _ = landmarks.shape
    conn = np.array(connections, dtype=int)

    fig = go.Figure()

    # --- Initial frame (frame 0) and joint markers ---
    pts0 = landmarks[0]
    x0, y0, z0 = _segment_xyz(pts0, conn)

    fig.add_trace(
        go.Scatter3d(
            x=x0,
            y=y0,
            z=z0,
            mode="lines",
            line=dict(color="royalblue", width=4),
            name="bones",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts0[:, 0],
            y=pts0[:, 1],
            z=pts0[:, 2],
            mode="markers",
            marker=dict(size=3, color="crimson"),
            name="joints",
        )
    )

    # --- Animation frames ---
    frames = []
    for i in range(t):
        pts = landmarks[i]
        x, y, z = _segment_xyz(pts, conn)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=x, y=y, z=z, mode="lines"),
                    go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers"),
                ],
                name=str(i),
            )
        )
    fig.frames = frames

    slider = dict(
        active=0,
        currentvalue={"prefix": "frame: "},
        steps=[
            dict(
                method="animate",
                args=[
                    [str(k)],
                    {"mode": "immediate", "frame": {"duration": 40, "redraw": True}},
                ],
                label=str(k),
            )
            for k in range(t)
        ],
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        sliders=[slider],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=1.05,
                y=0.1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    return fig
