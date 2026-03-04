from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import dash
from dash import Dash, Input, Output, State, dcc, html, no_update, ALL
from dash.exceptions import PreventUpdate

from loader.session import SessionData
from visualizer import figures, media


def build_layout(session_ids: List[str], default_session_id: str, default_session: SessionData) -> html.Div:
    duration = max(0.0, default_session.duration)
    start_time = default_session.start_time
    label_default_end = 0.0
    session_options = [{"label": sid, "value": sid} for sid in session_ids]
    return html.Div(
        [
            html.H2("Sensor Inspector"),
            dcc.Store(id="label-store", data={}),
            dcc.Store(id="label-setup-store", data={"mode": 1, "time1": None}),
            dcc.Store(id="note-history-store", data=[], storage_type="local"),
            
            # 1. Session Control Section (Sticky)
            html.Div(
                [
                    html.Div([
                        html.Div([
                            html.Label("Session"),
                            dcc.Dropdown(
                                id="session-selector",
                                options=session_options,
                                value=default_session_id,
                                clearable=False,
                                style={"width": "300px"}
                            ),
                        ], style={"display": "inline-block", "verticalAlign": "middle"}),
                        html.Button(
                            "Set Time 1", 
                            id="set-time-button", 
                            n_clicks=0,
                            style={
                                "marginLeft": "2rem",
                                "height": "38px",
                                "verticalAlign": "bottom",
                                "backgroundColor": "#007bff",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "4px",
                                "padding": "0 1.5rem",
                                "fontWeight": "bold",
                                "cursor": "pointer"
                            }
                        ),
                    ], style={"marginBottom": "0.5rem"}),
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
                    # Right: Joints & Velocity (Separated)
                    html.Div(
                        [
                            html.H4("Robot Joints"),
                            dcc.Graph(id="robot-joints", style={"height": "300px"}),
                            html.H4("Robot Velocity"),
                            dcc.Graph(id="robot-velocity", style={"height": "300px"}),
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
                            html.Div([
                                dcc.Input(
                                    id="anomaly-note",
                                    placeholder="Notes (type or select from history)",
                                    type="text",
                                    list="note-history-list",
                                    style={"width": "100%", "marginTop": "0.5rem"},
                                ),
                                html.Datalist(id="note-history-list", children=[])
                            ]),
                            html.Label("Interval (seconds)", style={"marginTop": "1rem", "display": "block"}),
                            dcc.RangeSlider(
                                id="label-range",
                                min=0,
                                max=duration,
                                value=[0.0, 0.0],
                                allowCross=True,
                            ),
                            html.Button(
                                "Add Label",
                                id="add-label-button",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "marginTop": "1rem",
                                    "height": "40px",
                                    "backgroundColor": "#6f42c1", # Purple for Add
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "fontWeight": "bold",
                                    "cursor": "pointer"
                                }
                            ),
                            html.Div(id="label-list", style={"marginTop": "1rem"}),
                        ],
                        className="label-panel",
                        style={"flex": "1", "padding": "1rem", "border": "1px solid #eee", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div([
                                html.H4("Timeline Labels", style={"display": "inline-block", "marginRight": "1rem"}),
                                html.Button(
                                    "Export Labels (.jsonl)",
                                    id="btn-export",
                                    style={
                                        "backgroundColor": "#28a745",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "0.4rem 1rem",
                                        "cursor": "pointer",
                                        "fontSize": "0.9rem"
                                    }
                                ),
                                html.Span(id="export-status", style={"marginLeft": "1rem", "color": "green", "fontSize": "0.9rem"}),
                                dcc.Download(id="download-labels"),
                            ]),
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
        [0.0, 0.0],
    )


def _extract_time_from_click(data: Dict[str, Any] | None, key: str) -> float | None:
    if not data:
        return None
    points = data.get("points") or []
    if not points:
        return None
    
    # Debug print to help identify available keys in problematic environments
    # print(f"DEBUG: Point keys: {list(points[0].keys())}, Trigger Key: {key}")
    
    value = points[0].get(key)
    
    if value is None:
        # Fallback: some plotly versions/trace types might nest customdata differently
        return None
        
    # Robust extraction: unwrap potentially nested lists/arrays (e.g., [[timestamp]])
    while isinstance(value, (list, np.ndarray)) and len(value) > 0:
        value = value[0]
        
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_dash_app(
    session_map: Dict[str, SessionData], 
    session_paths: Dict[str, Path], 
    default_session_id: str
) -> Dash:
    if not session_paths:
        raise ValueError("No sessions found.")
    
    if default_session_id not in session_map and default_session_id in session_paths:
        session_map[default_session_id] = SessionData(session_paths[default_session_id])
    
    default_session = session_map[default_session_id]
    session_ids = sorted(session_paths.keys())

    app = Dash(__name__)
    app.layout = build_layout(session_ids, default_session_id, default_session)

    def get_session(session_id: str) -> SessionData:
        if session_id not in session_map and session_id in session_paths:
            print(f"Loading session: {session_id}")
            session_map[session_id] = SessionData(session_paths[session_id])
        return session_map.get(session_id, default_session)

    # Timeline
    @app.callback(
        Output("timeline-graph", "figure"),
        Input("label-store", "data"),
        Input("session-selector", "value"),
        Input("label-range", "value"),
    )
    def update_timeline(label_data, session_id, range_value):
        session = get_session(session_id)
        labels = (label_data or {}).get(session_id, [])
        return figures.build_timeline(session, labels, active_range=range_value)

    # 1단계: 시간 영역 지정 (Set Time 버튼)
    @app.callback(
        Output("label-setup-store", "data"),
        Output("set-time-button", "children"),
        Output("set-time-button", "style"),
        Output("label-range", "value", allow_duplicate=True),
        Input("set-time-button", "n_clicks"),
        State("time-slider", "value"),
        State("label-setup-store", "data"),
        prevent_initial_call=True,
    )
    def capture_set_time(n_clicks, t_current, setup_data):
        setup_data = setup_data or {"mode": 1, "time1": None}
        
        button_style = {
            "marginLeft": "2rem",
            "height": "38px",
            "verticalAlign": "bottom",
            "color": "white",
            "border": "none",
            "borderRadius": "4px",
            "padding": "0 1.5rem",
            "fontWeight": "bold",
            "cursor": "pointer"
        }

        if setup_data["mode"] == 1:
            # 첫 번째 포인트 캡처
            setup_data["mode"] = 2
            setup_data["time1"] = t_current
            button_style["backgroundColor"] = "#dc3545" # Red
            return setup_data, "Set Time 2", button_style, [t_current, t_current]
        else:
            # 두 번째 포인트 캡처 (저장은 아직 안 함)
            t1 = setup_data["time1"]
            t2 = t_current
            
            setup_data["mode"] = 1
            setup_data["time1"] = None
            button_style["backgroundColor"] = "#007bff" # Blue
            
            return setup_data, "Set Time 1", button_style, [min(t1, t2), max(t1, t2)]

    # 2단계: 실제로 Store에 저장 (Add Label 버튼)
    @app.callback(
        Output("label-store", "data", allow_duplicate=True),
        Output("note-history-store", "data"),
        Input("add-label-button", "n_clicks"),
        State("label-range", "value"),
        State("anomaly-type", "value"),
        State("anomaly-note", "value"),
        State("session-selector", "value"),
        State("label-store", "data"),
        State("note-history-store", "data"),
        prevent_initial_call=True,
    )
    def add_label_to_store(n_clicks, range_val, a_type, a_note, session_id, existing_labels, history):
        if not n_clicks:
            raise PreventUpdate
        
        # 1. Update Labels
        existing_labels = existing_labels or {}
        if session_id not in existing_labels:
            existing_labels[session_id] = []
            
        existing_labels[session_id].append({
            "start": float(range_val[0]),
            "end": float(range_val[1]),
            "label": a_type,
            "note": a_note or ""
        })

        # 2. Update History Cache (Most recent first)
        history = history or []
        if a_note and a_note.strip():
            note_stripped = a_note.strip()
            if note_stripped in history:
                history.remove(note_stripped)
            history.insert(0, note_stripped)
            # Limit history size to 20
            history = history[:20]
        
        return existing_labels, history

    # Update DataList for Note History
    @app.callback(
        Output("note-history-list", "children"),
        Input("note-history-store", "data")
    )
    def update_note_history_list(history):
        if not history:
            return []
        return [html.Option(value=note) for note in history]

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
                    [
                        html.Span(f"[{item['label']}] {item['start']:.3f}-{item['end']:.3f}s — {item.get('note','')}", style={"marginRight": "10px"}),
                        html.Button(
                            "❌", 
                            id={"type": "delete-label", "index": i},
                            n_clicks=0,
                            style={
                                "border": "none", 
                                "background": "none", 
                                "cursor": "pointer", 
                                "padding": "0 2px",
                                "fontSize": "14px"
                            },
                            title="Delete Label"
                        )
                    ],
                    style={"marginBottom": "5px", "display": "flex", "alignItems": "center"}
                )
                for i, item in enumerate(labels)
            ],
            style={"paddingLeft": "1.2rem"}
        )

    # Label delete functionality
    @app.callback(
        Output("label-store", "data", allow_duplicate=True),
        Input({"type": "delete-label", "index": ALL}, "n_clicks"),
        State("label-store", "data"),
        State("session-selector", "value"),
        prevent_initial_call=True,
    )
    def delete_label(n_clicks, label_data, session_id):
        if not dash.callback_context.triggered:
            raise PreventUpdate
        
        # Check if any click actually happened
        if not any(n_clicks):
            raise PreventUpdate

        triggered_id = dash.callback_context.triggered_id
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "delete-label":
            raise PreventUpdate
            
        idx = triggered_id["index"]
        
        if label_data and session_id in label_data:
            labels = label_data[session_id]
            if 0 <= idx < len(labels):
                labels.pop(idx)
                return label_data
        
        raise PreventUpdate

    # Export Labels functionality (Server-side writing)
    @app.callback(
        Output("export-status", "children"),
        Output("download-labels", "data"),
        Input("btn-export", "n_clicks"),
        State("label-store", "data"),
        State("session-selector", "value"),
        prevent_initial_call=True,
    )
    def export_labels(n_clicks, label_data, session_id):
        if not label_data:
            # Metadata might still be valuable even without annotations
            label_data = {}
        
        import json
        from datetime import datetime
        
        # 1. Prepare Nested Content
        try:
            session = get_session(session_id)
            first_arc, last_arc = session.arc_on_range
            
            # Automatically detect Components (Laser segments)
            m_times = session.model.series.times
            m_laser = session.model.laser_on
            components = []
            if len(m_times) > 0 and len(m_laser) > 0:
                # Detect starts and ends of laser events
                diff = np.diff(m_laser.astype(int), prepend=0, append=0)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0] - 1 
                for i, (s, e) in enumerate(zip(starts, ends)):
                    # Guard for index bounds
                    e_idx = min(e, len(m_times) - 1)
                    components.append({
                        "start": round(float(m_times[s]), 4),
                        "end": round(float(m_times[e_idx]), 4),
                        "label": "Normal",
                        "note": str(i + 1) # Component Number
                    })

            labels = label_data.get(session_id, [])
            
            report = {
                "session_id": session_id,
                "overall_process": {
                    "start": round(first_arc, 4) if first_arc is not None else None,
                    "end": round(last_arc, 4) if last_arc is not None else None,
                    "label": "Full Arc Operation"
                },
                "components": components,
                "anomaly_annotations": [
                    {
                        "start": round(item["start"], 4),
                        "end": round(item["end"], 4),
                        "label": item["label"],
                        "note": item.get("note", "")
                    }
                    for item in labels
                ],
                "exported_at": datetime.now().isoformat()
            }
            output_str = json.dumps(report, indent=4, ensure_ascii=False)

            # 2. Write directly to the session directory
            export_path = session.path / "session_report.json"
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(output_str)
            
            status_msg = f"Saved to {export_path.name} ✅"
            download_data = dict(content=output_str, filename=f"report_{session_id}.json")
            return status_msg, download_data
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Export failed: {str(e)}", no_update

    # Combined Media Update
    @app.callback(
        Output("path-3d-graph", "figure"),
        Output("robot-joints", "figure"),
        Output("robot-velocity", "figure"),
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
        Input("label-range", "value"),
    )
    def update_dashboard(current_time, session_id, range_value):
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
        fig_joints = figures.build_robot_joints_plot(session.robot, current_time, active_range=range_value)
        fig_vel = figures.build_robot_velocity_plot(session.robot, current_time, active_range=range_value)

        # 2. Audio Figs
        fig_spec = figures.build_spectrogram(session.audio, current_time, active_range=range_value)
        fig_energy = figures.build_audio_energy_plot(session.audio, current_time, active_range=range_value)

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
        peak_x_fig = figures.build_peak_x_plot(session.ir_high.json_series, current_time, active_range=range_value)
        peak_y_fig = figures.build_peak_y_plot(session.ir_high.json_series, current_time, active_range=range_value)
        
        return fig_3d, fig_joints, fig_vel, fig_spec, fig_energy, ir_high, peak_x_fig, peak_y_fig, ir_low, rgb_src, rgb_style

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
        Input("robot-joints", "clickData"),
        Input("robot-velocity", "clickData"),
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
        velocity_click,
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
            "robot-joints.clickData": (joints_click, "x"),
            "robot-velocity.clickData": (velocity_click, "x"),
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
    # Only load default session at startup
    session_map = {default_session: SessionData(session_paths[default_session])}
    
    app = create_dash_app(session_map, session_paths, default_session)
    app.run(host=host, port=port, debug=debug)
