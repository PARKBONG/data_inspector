from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loader.session import SessionData
from loader.jsonl import JsonlSeries


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _segment_states(times: np.ndarray, path: np.ndarray, states: np.ndarray):
    segments: List[Dict[str, Any]] = []
    if len(path) == 0 or len(times) == 0:
        return segments
    if len(states) != len(times):
        states = np.zeros_like(times, dtype=bool)
    states = states.astype(bool)
    start = 0
    current_state = states[0]
    for idx in range(1, len(times)):
        if states[idx] != current_state:
            segments.append(
                {
                    "path": path[start : idx + 1],
                    "times": times[start : idx + 1],
                    "state": current_state,
                }
            )
            start = max(0, idx - 1)
            current_state = states[idx]
    segments.append(
        {
            "path": path[start:],
            "times": times[start:],
            "state": current_state,
        }
    )
    return segments


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_contour_points(raw_points) -> List[Sequence[float]]:
    if not raw_points:
        return []

    points: List[Sequence[float]] = []

    # Case 1: flattened list [x0, y0, x1, y1, ...]
    if isinstance(raw_points, list) and raw_points and isinstance(raw_points[0], (int, float, str)):
        flat = []
        for item in raw_points:
            value = _coerce_float(item)
            if value is None:
                flat = []
                break
            flat.append(value)
        if flat and len(flat) % 2 == 0:
            for idx in range(0, len(flat), 2):
                points.append((flat[idx], flat[idx + 1]))
            return points

    # Case 2: list of [x, y]
    if isinstance(raw_points, list):
        for entry in raw_points:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                x = _coerce_float(entry[0])
                y = _coerce_float(entry[1])
                if x is not None and y is not None:
                    points.append((x, y))
            elif isinstance(entry, dict):
                x = _coerce_float(entry.get("X") or entry.get("x"))
                y = _coerce_float(entry.get("Y") or entry.get("y"))
                if x is not None and y is not None:
                    points.append((x, y))

    return points


def build_timeline(session: SessionData, labels: List[Dict[str, Any]]):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(session.robot.series.times):
        fig.add_trace(
            go.Scatter(
                x=session.robot.series.times,
                y=session.robot.arc_on,
                name="Robot ArcOn",
                line_shape="hv",
            ),
            secondary_y=False,
        )

    if len(session.model.series.times):
        fig.add_trace(
            go.Scatter(
                x=session.model.series.times,
                y=session.model.laser_on,
                name="Model LaserOn",
                line_shape="hv",
            ),
            secondary_y=False,
        )

    if len(session.audio.rms.times):
        fig.add_trace(
            go.Scatter(
                x=session.audio.rms.times,
                y=_normalize(session.audio.rms.values),
                name="Audio RMS (norm)",
                mode="lines",
                line={"color": "#2474f7"},
            ),
            secondary_y=False,
        )

    if len(session.ir_high.json_series.times):
        fig.add_trace(
            go.Scatter(
                x=session.ir_high.json_series.times,
                y=session.ir_high.json_series.values.get("MaxTemp", np.array([])),
                name="IR High MaxTemp",
                mode="lines",
                line={"color": "#2ca02c"},
            ),
            secondary_y=True,
        )

    for item in labels or []:
        fig.add_shape(
            type="rect",
            x0=item["start"],
            x1=item["end"],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor="rgba(255,0,0,0.1)",
            line={"width": 0},
        )

    fig.update_layout(
        height=400,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=30, b=30),
    )
    fig.update_yaxes(title_text="Logic / Normalized")
    fig.update_yaxes(title_text="Temperature", secondary_y=True)
    fig.update_xaxes(
        title_text="Time (s)",
        range=[0, session.duration],
    )
    return fig


def build_path_3d(
    path: np.ndarray,
    times: np.ndarray,
    states: np.ndarray,
    timestamp: float,
    title: str,
):
    fig = go.Figure()
    if len(path) and path.shape[1] >= 3 and len(times):
        segments = _segment_states(times, path, states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#111111" if active else "#1f77b4"
            dash = "solid" if active else "dash"
            width = 4 if active else 2
            fig.add_trace(
                go.Scatter3d(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    z=seg_path[:, 2],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=width),
                    hovertemplate="t=%{customdata:.3f}s",
                    customdata=seg_times,
                    name="Arc On" if active else "Arc Off",
                    showlegend=False,
                )
            )
        idx = int(np.argmin(np.abs(times - timestamp)))
        idx = min(idx, len(path) - 1)
        fig.add_trace(
            go.Scatter3d(
                x=[path[idx, 0]],
                y=[path[idx, 1]],
                z=[path[idx, 2]],
                mode="markers",
                marker=dict(color="red", size=4),
                name="Current",
                hovertemplate=f"Current t={times[idx]:.3f}s",
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0)),
        ),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_path_xy(
    path: np.ndarray,
    times: np.ndarray,
    states: np.ndarray,
    timestamp: float,
    title: str,
):
    fig = go.Figure()
    if len(path) and path.shape[1] >= 2 and len(times):
        segments = _segment_states(times, path[:, :2], states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#111111" if active else "#1f77b4"
            dash = "solid" if active else "dash"
            width = 3 if active else 1.5
            fig.add_trace(
                go.Scatter(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=width),
                    customdata=seg_times,
                    hovertemplate="t=%{customdata:.3f}s",
                    showlegend=False,
                )
            )
        idx = int(np.argmin(np.abs(times - timestamp)))
        idx = min(idx, len(path) - 1)
        fig.add_trace(
            go.Scatter(
                x=[path[idx, 0]],
                y=[path[idx, 1]],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Current",
                hovertemplate=f"Current t={times[idx]:.3f}s",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_spectrogram(audio, current_time: Optional[float] = None):
    spec = audio.spectrogram
    fig = go.Figure()
    if len(spec.times) and spec.magnitude.size:
        magnitude = 20 * np.log10(spec.magnitude + 1e-6)
        fig.add_trace(
            go.Heatmap(
                x=spec.times,
                y=spec.frequencies,
                z=magnitude,
                colorscale="Viridis",
            )
        )
    if current_time is not None:
        fig.add_vline(x=current_time, line=dict(color="red", width=2))
    fig.update_layout(
        title="Audio Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_ir(reader, timestamp: float, title: str, metadata: Optional[Dict[str, Any]] = None, draw_contour: bool = True, draw_peak: bool = True):
    fig = go.Figure()
    if reader is None:
        fig.update_layout(title=f"{title} (not available)")
        return fig
    frame = reader.frame_values(timestamp)
    if frame is not None:
        vmin, vmax = reader.temp_range
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorscale="Inferno",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Temp"),
            )
        )
        
        # DEBUG: Log metadata info
        if metadata:
            print(f"[{title}] Metadata available. Keys: {metadata.keys()}")
            print(f"[{title}] draw_contour={draw_contour}, draw_peak={draw_peak}")
            raw_contour = metadata.get("ContourPoints")
            print(f"[{title}] Raw ContourPoints type: {type(raw_contour)}, value: {raw_contour}")
        
        if metadata and draw_contour:
            contour_points = _parse_contour_points(metadata.get("ContourPoints"))
            print(f"[{title}] Parsed contour_points: {len(contour_points) if contour_points else 0} points")
            if contour_points:
                print(f"[{title}] First few points: {contour_points[:3]}")
                # Close the contour by adding the first point at the end
                closed_points = list(contour_points) + [contour_points[0]]
                xs, ys = zip(*closed_points)
                print(f"[{title}] Adding contour trace with {len(xs)} points")
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#00ffff"),
                        name="Contour",
                        showlegend=False,
                    )
                )
            else:
                print(f"[{title}] No contour points after parsing")
        
        if metadata and draw_peak:
            peak_x = _coerce_float(metadata.get("PeakX"))
            peak_y = _coerce_float(metadata.get("PeakY"))
            print(f"[{title}] Peak: x={peak_x}, y={peak_y}")
            if peak_x is not None and peak_y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[peak_x],
                        y=[peak_y],
                        mode="markers",
                        marker=dict(color="white", size=8, symbol="x"),
                        name="Peak",
                        showlegend=False,
                    )
                )
    width = reader.width or 100
    height = reader.height or 100
    aspect = height / width if width else 1
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        width=420,
        height=420 * aspect,
        margin=dict(l=20, r=20, t=40, b=40),
    )
    fig.update_yaxes(
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def build_peak_y_plot(series: JsonlSeries, current_time: float):
    fig = go.Figure()
    times = series.times
    peaks = series.values.get("PeakY", np.array([]))
    if len(times) and len(peaks):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=peaks,
                mode="lines",
                line=dict(color="#ff7f0e"),
                customdata=times,
            )
        )
        fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))
    fig.update_layout(
        title="Peak Y Position",
        xaxis_title="Time (s)",
        yaxis_title="Peak Y",
        height=200,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig
