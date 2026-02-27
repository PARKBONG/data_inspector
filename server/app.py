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
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Anomaly label"),
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
                            ),
                            html.Label("Interval (seconds)"),
                            dcc.RangeSlider(
                                id="label-range",
                                min=0,
                                max=duration,
                                value=[start_time, label_default_end],
                                allowCross=False,
                            ),
                            html.Button("Add label", id="add-label-button"),
                            html.Div(id="label-list"),
                        ],
                        className="label-panel",
                    ),
                    dcc.Graph(id="timeline-graph"),
                ],
                className="timeline-section",
            ),
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
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="robot-path-graph", className="path-graph"),
                    dcc.Graph(id="model-path-graph", className="path-graph"),
                ],
                style={"display": "flex", "gap": "1rem"},
            ),
            html.Div(
                [
                    dcc.Graph(id="robot-path-xy", className="path-graph"),
                    dcc.Graph(id="model-path-xy", className="path-graph"),
                ],
                style={"display": "flex", "gap": "1rem"},
            ),
            html.Div(
                [
                    dcc.Graph(id="audio-spectrogram", className="spectrogram"),
                ]
            ),
            html.Div(
                [
                    html.H3("IR High (with Contour & Peak)"),
                    dcc.Graph(id="ir-high-graph", className="ir-graph", style={"flex": "2"}),
                    dcc.Graph(id="ir-high-peak", style={"flex": "1", "height": "220px"}),
                ],
                style={"display": "flex", "gap": "1rem", "alignItems": "stretch"},
            ),
            html.Div(
                [
                    html.H3("Test: IR High Only (No Contour/Peak)"),
                    dcc.Graph(id="ir-high-test-only", className="ir-graph"),
                ],
                style={"border": "2px solid red", "padding": "1rem"},
            ),
            html.Div(
                [
                    html.H3("Test: Contour Only"),
                    dcc.Graph(id="ir-high-contour-only", className="ir-graph"),
                ],
                style={"border": "2px solid blue", "padding": "1rem"},
            ),
            html.Div(
                [
                    dcc.Graph(id="ir-low-graph", className="ir-graph", style={"flex": "1"}),
                    html.Div(
                        [
                            html.H4("RGB Frame"),
                            html.Img(
                                id="rgb-frame",
                                style={
                                    "maxWidth": "100%",
                                    "height": "420px",
                                    "objectFit": "contain",
                                },
                            ),
                        ],
                        style={"flex": "1", "textAlign": "center"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem", "alignItems": "center"},
            ),
        ]
    )


def _initial_time_values(session: SessionData):
    max_time = session.duration if session.duration else 0.0
    start = min(session.start_time, max_time)
    label_end = start if max_time == 0 else min(max_time, start + 5)
    return (
        0.0,
        max_time,
        start,
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

    # Paths (3D + XY)
    @app.callback(
        Output("robot-path-graph", "figure"),
        Output("model-path-graph", "figure"),
        Output("robot-path-xy", "figure"),
        Output("model-path-xy", "figure"),
        Input("time-slider", "value"),
        Input("session-selector", "value"),
    )
    def update_paths(current_time, session_id):
        current_time = float(current_time or 0.0)
        session = get_session(session_id)
        robot_fig = figures.build_path_3d(
            session.robot.path_xyz,
            session.robot.series.times,
            session.robot.arc_on,
            current_time,
            "Robot Path (XYZ)",
        )
        model_fig = figures.build_path_3d(
            session.model.path_xyz,
            session.model.series.times,
            session.model.laser_on,
            current_time,
            "Model Path (XYZ)",
        )
        robot_xy = figures.build_path_xy(
            session.robot.path_xyz,
            session.robot.series.times,
            session.robot.arc_on,
            current_time,
            "Robot Path (XY)",
        )
        model_xy = figures.build_path_xy(
            session.model.path_xyz,
            session.model.series.times,
            session.model.laser_on,
            current_time,
            "Model Path (XY)",
        )
        return robot_fig, model_fig, robot_xy, model_xy

    # Spectrogram with time indicator
    @app.callback(
        Output("audio-spectrogram", "figure"),
        Input("time-slider", "value"),
        Input("session-selector", "value"),
    )
    def update_spectrogram(current_time, session_id):
        current_time = float(current_time or 0.0)
        session = get_session(session_id)
        return figures.build_spectrogram(session.audio, current_time)

    # Media (IR + RGB)
    @app.callback(
        Output("rgb-frame", "src"),
        Output("ir-high-graph", "figure"),
        Output("ir-low-graph", "figure"),
        Output("ir-high-peak", "figure"),
        Output("ir-high-test-only", "figure"),
        Output("ir-high-contour-only", "figure"),
        Input("time-slider", "value"),
        Input("session-selector", "value"),
    )
    def update_media(current_time, session_id):
        current_time = float(current_time or 0.0)
        session = get_session(session_id)
        rgb_src = media.rgb_image_source(session.rgb_sequence, current_time)
        metadata = session.get_ir_high_metadata(current_time)
        ir_high = figures.build_ir(session.ir_high.raw_reader, current_time, "IR High (with Contour & Peak)", metadata=metadata, draw_contour=True, draw_peak=True)
        ir_low = figures.build_ir(session.ir_low.raw_reader, current_time, "IR Low")
        peak_fig = figures.build_peak_y_plot(session.ir_high.json_series, current_time)
        
        # Test: IR High only (no contour/peak)
        ir_high_test_only = figures.build_ir(session.ir_high.raw_reader, current_time, "IR High Only", metadata=metadata, draw_contour=False, draw_peak=False)
        
        # Test: Contour only (with peak)
        ir_high_contour_only = figures.build_ir(session.ir_high.raw_reader, current_time, "IR High + Contour Only", metadata=metadata, draw_contour=True, draw_peak=False)
        
        return rgb_src, ir_high, ir_low, peak_fig, ir_high_test_only, ir_high_contour_only

    # Synchronize time controls (slider and label range)
    @app.callback(
        Output("time-slider", "min"),
        Output("time-slider", "max"),
        Output("time-slider", "value"),
        Output("label-range", "min"),
        Output("label-range", "max"),
        Output("label-range", "value"),
        Input("session-selector", "value"),
        Input("timeline-graph", "clickData"),
        Input("robot-path-graph", "clickData"),
        Input("model-path-graph", "clickData"),
        Input("robot-path-xy", "clickData"),
        Input("model-path-xy", "clickData"),
        Input("ir-high-peak", "clickData"),
        State("time-slider", "value"),
    )
    def sync_time_controls(
        session_id,
        timeline_click,
        robot_click,
        model_click,
        robot_xy_click,
        model_xy_click,
        peak_click,
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
            "robot-path-graph.clickData": (robot_click, "customdata"),
            "model-path-graph.clickData": (model_click, "customdata"),
            "robot-path-xy.clickData": (robot_xy_click, "customdata"),
            "model-path-xy.clickData": (model_xy_click, "customdata"),
            "ir-high-peak.clickData": (peak_click, "x"),
        }

        data_key = trigger_map.get(trigger)
        if not data_key:
            raise PreventUpdate

        new_time = _extract_time_from_click(*data_key)
        if new_time is None:
            raise PreventUpdate

        return no_update, no_update, new_time, no_update, no_update, no_update

    return app


def run_server(session_paths: Dict[str, Path], default_session: str, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    session_map = {sid: SessionData(path) for sid, path in session_paths.items()}
    app = create_dash_app(session_map, default_session)
    app.run(host=host, port=port, debug=debug)
