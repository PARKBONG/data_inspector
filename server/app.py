from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from loader.session import SessionData
from visualizer import figures, media


def build_layout(session_ids: List[str], default_session_id: str, default_session: SessionData) -> html.Div:
    duration = max(0.0, default_session.duration)
    start_time = default_session.start_time
    label_default_end = min(duration, start_time + 5)
    session_options = [{"label": sid, "value": sid} for sid in session_ids]
    return html.Div(
        [
            html.H2("Sensor Inspector"),
            dcc.Store(id="label-store", data={}),
            
            # 1. Session Control Section (Sticky)
            html.Div(
                [
                    html.Label("Session"),
                    dcc.Dropdown(
                        id="session-selector",
                        options=session_options,
                        value=default_session_id,
                        clearable=False,
                    ),
                    html.Label("Current time (seconds)"),
                    dcc.Slider(
                        id="time-slider",
                        min=0.0,
                        max=duration,
                        value=start_time,
                        step=0.01,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 10,
                    "background": "white",
                    "padding": "0.5rem 0",
                    "borderBottom": "1px solid #eee",
                },
            ),

            # --- Path Section ---
            html.Hr(),
            html.H3("Path Analysis"),
            html.Div(
                [
                    # Left: 3D Path
                    html.Div(
                        [
                            html.H4("3D Path (Robot & Model Combined)"),
                            dcc.Graph(id="path-3d-graph", className="path-graph"),
                        ],
                        style={"flex": "1"},
                    ),
                    # Right: Joints & Velocity
                    html.Div(
                        [
                            html.H4("Robot Joints & Velocity"),
                            dcc.Graph(id="robot-joints-velocity", className="path-graph"),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem"},
            ),

            # --- Audio Section ---
            html.Hr(),
            html.H3("Audio Analysis"),
            html.Div(
                [
                    html.H4("Spectrogram"),
                    dcc.Graph(id="audio-spectrogram", className="spectrogram"),
                    html.H4("Energy (dB)"),
                    dcc.Graph(id="audio-energy", className="spectrogram"),
                ]
            ),

            # --- Thermal Section ---
            html.Hr(),
            html.H3("Thermal Analysis"),
            html.Div(
                [
                    # Left: IR High (Large)
                    html.Div(
                        [
                            html.H4("IR High (Melt Pool)"),
                            dcc.Graph(id="ir-high-graph", className="ir-graph", style={"height": "460px"}),
                        ],
                        style={"flex": "2"},
                    ),
                    # Right: Peak X and Y (Stacked)
                    html.Div(
                        [
                            html.H4("Peak Coordinates (X, Y)"),
                            dcc.Graph(id="ir-high-peak-x", style={"height": "220px"}),
                            dcc.Graph(id="ir-high-peak-y", style={"height": "220px"}),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "alignItems": "flex-start"},
            ),

            # --- Supervision Section ---
            html.Hr(),
            html.H3("Supervision"),
            html.Div(
                [
                    # Left: IR Low
                    html.Div(
                        [
                            html.H4("IR Low"),
                            dcc.Graph(id="ir-low-graph", className="ir-graph"),
                        ],
                        style={"flex": "1"},
                    ),
                    # Right: RGB Frame
                    html.Div(
                        [
                            html.H4("RGB Frame"),
                            html.Img(
                                id="rgb-frame",
                                style={
                                    "maxWidth": "100%",
                                    "objectFit": "contain",
                                    "borderRadius": "8px",
                                    "border": "1px solid #ddd",
                                },
                            ),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                ],
                style={"display": "flex", "gap": "2rem", "alignItems": "center"},
            ),

            # --- Anomaly Labeling Section (Bottom) ---
            html.Hr(),
            html.H3("Anomaly Labeling"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Anomaly type"),
                            dcc.Dropdown(
                                id="anomaly-type",
                                options=[
                                    {"label": "Anomaly", "value": "anomaly"},
                                    {"label": "Interesting", "value": "interesting"},
                                    {"label": "Issue", "value": "issue"},
                                ],
                                value="anomaly",
                                clearable=False,
                            ),
                            dcc.Input(
                                id="anomaly-note",
                                placeholder="Notes",
                                type="text",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                            html.Label("Interval (seconds)", style={"marginTop": "1rem", "display": "block"}),
                            dcc.RangeSlider(
                                id="label-range",
                                min=0,
                                max=duration,
                                value=[start_time, label_default_end],
                                allowCross=False,
                            ),
                            html.Button("Add label", id="add-label-button", style={"marginTop": "1rem"}),
                            html.Div(id="label-list", style={"marginTop": "1rem"}),
                        ],
                        className="label-panel",
                        style={"flex": "1", "padding": "1rem", "border": "1px solid #eee", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [
                            html.H4("Timeline Labels"),
                            dcc.Graph(id="timeline-graph"),
                        ],
                        style={"flex": "3"},
                    ),
                ],
                style={"display": "flex", "gap": "2rem"},
            ),
        ]
    )


def _initial_time_values(session: SessionData):
    max_time = session.duration if session.duration else 0.0
    start = min(session.start_time, max_time)
    label_end = start if max_time == 0 else min(max_time, start + 5)
    
    # Marks for the slider
    marks = {0: "0s", int(max_time): f"{int(max_time)}s"}
    first_arc, last_arc = session.arc_on_range
    if first_arc is not None:
        marks[first_arc] = {"label": "⚡", "style": {"color": "green", "fontWeight": "bold"}}
    if last_arc is not None:
        marks[last_arc] = {"label": "🛑", "style": {"color": "darkorange", "fontWeight": "bold"}}
        
    return (
        0.0,
        max_time,
        start,
        marks,
        0.0,
        max_time,
        [start, label_end],
    )


def _extract_time_from_click(data: Dict[str, Any] | None, key: str) -> float | None:
    if not data:
        return None
    points = data.get("points") or []
    if not points:
        return None
    value = points[0].get(key)
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_dash_app(session_map: Dict[str, SessionData], default_session_id: str) -> Dash:
    if not session_map:
        raise ValueError("No sessions available for visualization.")
    if default_session_id not in session_map:
        default_session_id = next(iter(session_map))
    session_ids = sorted(session_map.keys())
    default_session = session_map[default_session_id]

    app = Dash(__name__)
    app.layout = build_layout(session_ids, default_session_id, default_session)

    def get_session(session_id: str) -> SessionData:
        return session_map.get(session_id, default_session)

    # Timeline
    @app.callback(
        Output("timeline-graph", "figure"),
        Input("label-store", "data"),
        Input("session-selector", "value"),
    )
    def update_timeline(label_data, session_id):
        session = get_session(session_id)
        labels = (label_data or {}).get(session_id, [])
        return figures.build_timeline(session, labels)

    # Add label per session
    @app.callback(
        Output("label-store", "data"),
        Input("add-label-button", "n_clicks"),
        State("session-selector", "value"),
        State("label-range", "value"),
        State("anomaly-type", "value"),
        State("anomaly-note", "value"),
        State("label-store", "data"),
        prevent_initial_call=True,
    )
    def add_label(n_clicks, session_id, range_value, label_type, note, existing):
        existing = existing or {}
        if not range_value:
            return existing
        start, end = sorted(range_value)
        session_labels = list(existing.get(session_id, []))
        session_labels.append(
            {
                "id": f"{session_id}-{len(session_labels)+1}",
                "label": label_type,
                "start": start,
                "end": end,
                "note": note or "",
            }
        )
        existing = dict(existing)
        existing[session_id] = session_labels
        return existing

    @app.callback(
        Output("label-list", "children"),
        Input("label-store", "data"),
        Input("session-selector", "value"),
    )
    def render_label_list(label_data, session_id):
        labels = (label_data or {}).get(session_id, [])
        if not labels:
            return html.P("No labels yet.")
        return html.Ul(
            [
                html.Li(
                    f"[{item['label']}] {item['start']:.3f}-{item['end']:.3f}s — {item.get('note','')}"
                )
                for item in labels
            ]
        )

    # Combined Media Update
    @app.callback(
        Output("path-3d-graph", "figure"),
        Output("robot-joints-velocity", "figure"),
        Output("audio-spectrogram", "figure"),
        Output("audio-energy", "figure"),
        Output("ir-high-graph", "figure"),
        Output("ir-high-peak-x", "figure"),
        Output("ir-high-peak-y", "figure"),
        Output("ir-low-graph", "figure"),
        Output("rgb-frame", "src"),
        Output("rgb-frame", "style"),
        Input("time-slider", "value"),
        Input("session-selector", "value"),
    )
    def update_dashboard(current_time, session_id):
        current_time = float(current_time or 0.0)
        session = get_session(session_id)
        
        # 1. Path Figs
        fig_3d = figures.build_combined_path_3d(
            session.robot.path_xyz,
            session.robot.series.times,
            session.robot.arc_on,
            session.model.path_xyz,
            session.model.series.times,
            session.model.laser_on,
            current_time,
            f"XYZ Path (t={current_time:.2f}s)",
        )
        fig_joints = figures.build_robot_joint_velocity_plot(session.robot, current_time)

        # 2. Audio Figs
        fig_spec = figures.build_spectrogram(session.audio, current_time)
        fig_energy = figures.build_audio_energy_plot(session.audio, current_time)

        # 3. Thermal & Supervision
        ir_low_reader = session.ir_low.raw_reader
        width = (ir_low_reader.width if ir_low_reader else 0) or 100
        height = (ir_low_reader.height if ir_low_reader else 0) or 100
        aspect = height / width
        calculated_height = f"{int(420 * aspect)}px"
        
        rgb_src = media.rgb_image_source(session.rgb_sequence, current_time)
        rgb_style = {
            "maxWidth": "100%",
            "height": calculated_height,
            "objectFit": "contain",
            "borderRadius": "8px",
            "border": "1px solid #ddd",
        }
        
        metadata = session.get_ir_high_metadata(current_time)
        ir_high = figures.build_ir(session.ir_high.raw_reader, current_time, "IR High", metadata=metadata)
        ir_low = figures.build_ir(session.ir_low.raw_reader, current_time, "IR Low", draw_contour=False, draw_peak=False)
        peak_x_fig = figures.build_peak_x_plot(session.ir_high.json_series, current_time)
        peak_y_fig = figures.build_peak_y_plot(session.ir_high.json_series, current_time)
        
        return fig_3d, fig_joints, fig_spec, fig_energy, ir_high, peak_x_fig, peak_y_fig, ir_low, rgb_src, rgb_style

    # Synchronize time controls (slider and label range)
    @app.callback(
        Output("time-slider", "min"),
        Output("time-slider", "max"),
        Output("time-slider", "value"),
        Output("time-slider", "marks"),
        Output("label-range", "min"),
        Output("label-range", "max"),
        Output("label-range", "value"),
        Input("session-selector", "value"),
        Input("timeline-graph", "clickData"),
        Input("path-3d-graph", "clickData"),
        Input("robot-joints-velocity", "clickData"),
        Input("audio-spectrogram", "clickData"),
        Input("audio-energy", "clickData"),
        Input("ir-high-peak-x", "clickData"),
        Input("ir-high-peak-y", "clickData"),
        State("time-slider", "value"),
        prevent_initial_call=False,
    )
    def sync_time_controls(
        session_id,
        timeline_click,
        path_click,
        joints_click,
        audio_click,
        energy_click,
        peak_x_click,
        peak_y_click,
        current_value,
    ):
        session = get_session(session_id)
        ctx = dash.callback_context
        if not ctx.triggered:
            return _initial_time_values(session)

        trigger = ctx.triggered[0]["prop_id"]
        if trigger == "session-selector.value":
            return _initial_time_values(session)

        trigger_map = {
            "timeline-graph.clickData": (timeline_click, "x"),
            "path-3d-graph.clickData": (path_click, "customdata"),
            "robot-joints-velocity.clickData": (joints_click, "x"),
            "audio-spectrogram.clickData": (audio_click, "x"),
            "audio-energy.clickData": (energy_click, "x"),
            "ir-high-peak-x.clickData": (peak_x_click, "x"),
            "ir-high-peak-y.clickData": (peak_y_click, "x"),
        }

        data_key = trigger_map.get(trigger)
        if not data_key:
            raise PreventUpdate

        new_time = _extract_time_from_click(*data_key)
        if new_time is None:
            raise PreventUpdate

        return no_update, no_update, new_time, no_update, no_update, no_update, no_update

    return app


def run_server(session_paths: Dict[str, Path], default_session: str, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    session_map = {sid: SessionData(path) for sid, path in session_paths.items()}
    app = create_dash_app(session_map, default_session)
    app.run(host=host, port=port, debug=debug)
