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


def _add_active_range(fig: go.Figure, active_range: Optional[List[float]], layer="above"):
    if active_range and len(active_range) == 2:
        t1, t2 = sorted(active_range)
        fig.add_vrect(
            x0=t1,
            x1=t2,
            fillcolor="rgba(255, 0, 0, 0.3)",
            layer=layer,
            line=dict(color="red", width=1),
            annotation_text="SETTING..." if t1 == t2 else "",
        )


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


def build_timeline(session: SessionData, labels: List[Dict[str, Any]], active_range: Optional[List[float]] = None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Base Data (Robot ArcOn, Model LaserOn)
    if len(session.robot.series.times):
        fig.add_trace(
            go.Scatter(
                x=session.robot.series.times,
                y=session.robot.arc_on,
                name="Robot ArcOn",
                line_shape="hv",
                line=dict(color="#1f77b4")
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
                line=dict(color="#ff7f0e", dash="dot")
            ),
            secondary_y=False,
        )

    # 2. Temperature Reference
    if len(session.ir_high.json_series.times):
        fig.add_trace(
            go.Scatter(
                x=session.ir_high.json_series.times,
                y=session.ir_high.json_series.values.get("MaxTemp", np.array([])),
                name="IR High MaxTemp",
                mode="lines",
                line=dict(color="#2ca02c", width=1),
                opacity=0.6,
            ),
            secondary_y=True,
        )
    
    # 3. User Labels
    for item in labels or []:
        fig.add_vrect(
            x0=item["start"],
            x1=item["end"],
            fillcolor="rgba(255,0,0,0.15)",
            layer="below",
            line_width=1,
            line_color="rgba(255,0,0,0.3)",
            annotation_text=item.get("label") or item.get("type") or "label",
        )
        
    # 3.1 Active Range (In progress)
    _add_active_range(fig, active_range)

    # 4. Arc Transition Markers (Requested)
    first_arc, last_arc = session.arc_on_range
    if first_arc is not None:
        fig.add_vline(x=first_arc, line=dict(color="green", dash="dash", width=2))
        fig.add_annotation(
            x=first_arc, y=1.05, yref="paper", 
            text="First Arc On", showarrow=False, 
            font=dict(color="green", size=10), yanchor="bottom"
        )
    if last_arc is not None:
        fig.add_vline(x=last_arc, line=dict(color="darkorange", dash="dash", width=2))
        fig.add_annotation(
            x=last_arc, y=1.05, yref="paper", 
            text="Last Arc Off", showarrow=False, 
            font=dict(color="darkorange", size=10), yanchor="bottom"
        )

    fig.update_layout(
        title="Session Overview & Logic Status",
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Logic (0/1)", secondary_y=False, range=[-0.1, 1.2])
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    fig.update_xaxes(title_text="Time (s)", range=[0, session.duration])
    
    return fig


def build_combined_path_3d(
    robot_path: np.ndarray,
    robot_times: np.ndarray,
    robot_states: np.ndarray,
    model_path: np.ndarray,
    model_times: np.ndarray,
    model_states: np.ndarray,
    timestamp: float,
    title: str,
):
    fig = go.Figure()

    # 1. Plot Robot Path
    if len(robot_path) and robot_path.shape[1] >= 3 and len(robot_times):
        segments = _segment_states(robot_times, robot_path, robot_states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#111111" if active else "#1f77b4"
            dash = "solid" if active else "dash"
            fig.add_trace(
                go.Scatter3d(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    z=seg_path[:, 2],
                    mode="lines+markers",
                    line=dict(color=color, dash=dash, width=3),
                    marker=dict(size=8, opacity=0.01), # Increased size and fixed opacity
                    hovertemplate="Robot t=%{customdata:.3f}s<extra></extra>", # Added <extra></extra> for cleaner hover
                    customdata=seg_times,
                    name="Robot (Arc On)" if active else "Robot (Arc Off)",
                    showlegend=True if active else False,
                )
            )

    # 2. Plot Adjusted Model Path
    # Model Z is added to the Robot Z at that time
    if len(model_path) and model_path.shape[1] >= 3 and len(model_times) and len(robot_path):
        # Interpolate robot Z to model timestamps
        adj_model_path = model_path.copy()
        robot_z_at_model_times = np.interp(model_times, robot_times, robot_path[:, 2])
        adj_model_path[:, 2] += robot_z_at_model_times

        segments = _segment_states(model_times, adj_model_path, model_states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#ff7f0e" if active else "#aec7e8"
            fig.add_trace(
                go.Scatter3d(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    z=seg_path[:, 2],
                    mode="lines+markers",
                    line=dict(color=color, width=5),
                    marker=dict(size=8, opacity=0.01),
                    hovertemplate="Model t=%{customdata:.3f}s<extra></extra>",
                    customdata=seg_times,
                    name="Model (Laser On)" if active else "Model (Laser Off)",
                    showlegend=True if active else False,
                )
            )

    # 3. Current Position Marker (Robot)
    if len(robot_path) and len(robot_times):
        idx = int(np.argmin(np.abs(robot_times - timestamp)))
        idx = min(idx, len(robot_path) - 1)
        fig.add_trace(
            go.Scatter3d(
                x=[robot_path[idx, 0]],
                y=[robot_path[idx, 1]],
                z=[robot_path[idx, 2]],
                mode="markers",
                marker=dict(color="red", size=8, symbol="diamond"),
                name="Current (Robot)",
                hovertemplate=f"Robot Current t={robot_times[idx]:.3f}s",
                customdata=[robot_times[idx]],
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
            # Light theme for 3D
            bgcolor="white",
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        clickmode="event+select",
        template="plotly_white",
    )
    return fig


def build_combined_path_xy(
    robot_path: np.ndarray,
    robot_times: np.ndarray,
    robot_states: np.ndarray,
    model_path: np.ndarray,
    model_times: np.ndarray,
    model_states: np.ndarray,
    timestamp: float,
    title: str,
):
    fig = go.Figure()

    # 1. Plot Robot Path
    if len(robot_path) and robot_path.shape[1] >= 2 and len(robot_times):
        segments = _segment_states(robot_times, robot_path[:, :2], robot_states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#111111" if active else "#1f77b4"
            dash = "solid" if active else "dash"
            fig.add_trace(
                go.Scatter(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=1.5),
                    customdata=seg_times,
                    hovertemplate="Robot t=%{customdata:.3f}s",
                    name="Robot",
                    showlegend=False,
                )
            )

    # 2. Plot Model Path
    if len(model_path) and model_path.shape[1] >= 2 and len(model_times):
        segments = _segment_states(model_times, model_path[:, :2], model_states)
        for segment in segments:
            seg_path = segment["path"]
            seg_times = segment["times"]
            if len(seg_path) < 2:
                continue
            active = bool(segment["state"])
            color = "#ff7f0e" if active else "#aec7e8"
            fig.add_trace(
                go.Scatter(
                    x=seg_path[:, 0],
                    y=seg_path[:, 1],
                    mode="lines",
                    line=dict(color=color, width=3),
                    customdata=seg_times,
                    hovertemplate="Model t=%{customdata:.3f}s",
                    name="Model",
                    showlegend=False,
                )
            )

    # 3. Current Position Marker (Robot)
    if len(robot_path) and len(robot_times):
        idx = int(np.argmin(np.abs(robot_times - timestamp)))
        idx = min(idx, len(robot_path) - 1)
        fig.add_trace(
            go.Scatter(
                x=[robot_path[idx, 0]],
                y=[robot_path[idx, 1]],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Current",
                hovertemplate=f"Current t={robot_times[idx]:.3f}s",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_spectrogram(audio, current_time: Optional[float] = None, active_range: Optional[List[float]] = None):
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
    
    _add_active_range(fig, active_range)
    fig.update_layout(
        title="Audio Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )
    return fig


def build_ir(reader, timestamp, title, metadata=None, draw_contour=True, draw_peak=True):
    if not reader:
        return go.Figure()
    frame = reader.frame_values(timestamp)
    if frame is None:
        return go.Figure()
    
    vmin, vmax = reader.temp_range
    # Using 'Inferno' as suggested by reference code
    fig = go.Figure(
        data=go.Heatmap(
            z=frame,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(title=dict(text="Temp (°C)", side="right")),
            hoverinfo="z+x+y",
            name="IR Heatmap",
        )
    )
    
    if metadata:
        if draw_contour:
            contour_points = _parse_contour_points(metadata.get("ContourPoints"))
            if contour_points:
                closed_points = list(contour_points) + [contour_points[0]]
                xs, ys = zip(*closed_points)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#00ffff", width=2),
                        name="Contour",
                        showlegend=False,
                    )
                )
        if draw_peak:
            peak_x = _coerce_float(metadata.get("PeakX"))
            peak_y = _coerce_float(metadata.get("PeakY"))
            area = _coerce_float(metadata.get("Area"))
            if peak_x is not None and peak_y is not None:
                if (peak_x > 0 or peak_y > 0) and (area is None or area > 0):
                    fig.add_trace(
                        go.Scatter(
                            x=[peak_x],
                            y=[peak_y],
                            mode="markers",
                            marker=dict(color="blue", size=10, symbol="x", line=dict(width=1, color="white")),
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
        width=480, # Increased slightly for colorbar
        height=420 * aspect,
        margin=dict(l=20, r=60, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        template="plotly_white",
    )
    fig.update_yaxes(
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
        gridcolor="#eeeeee",
    )
    fig.update_xaxes(
        gridcolor="#eeeeee",
    )
    return fig


def build_peak_y_plot(series: JsonlSeries, current_time: float, active_range: Optional[List[float]] = None):
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
    _add_active_range(fig, active_range)
    fig.update_layout(
        title="Peak Y Position",
        clickmode="event+select",
        xaxis_title="Time (s)",
        yaxis_title="Peak Y",
        height=220,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


def build_peak_x_plot(series: JsonlSeries, current_time: float, active_range: Optional[List[float]] = None):
    fig = go.Figure()
    times = series.times
    peaks = series.values.get("PeakX", np.array([]))
    if len(times) and len(peaks):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=peaks,
                mode="lines",
                line=dict(color="#17becf"),
                customdata=times,
            )
        )
        fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))
    _add_active_range(fig, active_range)
    fig.update_layout(
        title="Peak X Position",
        clickmode="event+select",
        xaxis_title="Time (s)",
        yaxis_title="Peak X",
        height=220,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


def build_robot_joints_plot(robot_data: RobotData, current_time: float, active_range: Optional[List[float]] = None):
    fig = go.Figure()
    times = robot_data.series.times
    
    # Joint Values
    joints = robot_data.joints
    for name, values in joints.items():
        if len(values):
            fig.add_trace(
                go.Scatter(x=times, y=values, name=name, mode="lines", opacity=0.8, customdata=times)
            )
            
    fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))
    _add_active_range(fig, active_range)
    
    fig.update_layout(
        title="Robot Joints",
        clickmode="event+select",
        xaxis_title="Time (s)",
        yaxis_title="Joint Angle (deg)",
        height=300,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_robot_velocity_plot(robot_data: RobotData, current_time: float, active_range: Optional[List[float]] = None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    times = robot_data.series.times
    
    # Velocity Components
    vel_xyz = robot_data.velocity_xyz
    
    # Left Axis: Vx, Vy (larger components)
    if len(vel_xyz["Vx"]):
        fig.add_trace(
            go.Scatter(x=times, y=vel_xyz["Vx"], name="Vx", mode="lines", line=dict(width=1.5), customdata=times),
            secondary_y=False
        )
    if len(vel_xyz["Vy"]):
        fig.add_trace(
            go.Scatter(x=times, y=vel_xyz["Vy"], name="Vy", mode="lines", line=dict(width=1.5), customdata=times),
            secondary_y=False
        )
    
    # Right Axis: Vz (smaller component)
    if len(vel_xyz["Vz"]):
        fig.add_trace(
            go.Scatter(x=times, y=vel_xyz["Vz"], name="Vz (Z-axis)", mode="lines", line=dict(dash="dot", width=1.5), customdata=times),
            secondary_y=True
        )
    
    fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))
    _add_active_range(fig, active_range)
    
    fig.update_layout(
        title="Robot Velocity (X, Y, Z)",
        clickmode="event+select",
        xaxis_title="Time (s)",
        height=300,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="XY Velocity (mm/s)", secondary_y=False)
    fig.update_yaxes(title_text="Z Velocity (mm/s)", secondary_y=True)
    return fig


def build_audio_energy_plot(audio_data: AudioData, current_time: float, active_range: Optional[List[float]] = None):
    fig = go.Figure()
    rms = audio_data.rms
    if len(rms.times):
        # Convert to dB: 20 * log10(RMS + eps)
        eps = 1e-10
        db_values = 20 * np.log10(rms.values + eps)
        fig.add_trace(
            go.Scatter(
                x=rms.times,
                y=db_values,
                mode="lines",
                name="Energy (dB)",
                line=dict(color="#1f77b4"),
                customdata=rms.times,
            )
        )
        fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))
    _add_active_range(fig, active_range)
    
    fig.update_layout(
        title="Audio Energy (Intensity)",
        clickmode="event+select",
        xaxis_title="Time (s)",
        yaxis_title="Energy (dB)",
        height=200,
        margin=dict(l=40, r=40, t=30, b=30),
        template="plotly_white",
    )
    return fig
