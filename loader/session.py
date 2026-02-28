from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .audio import AudioLoader, AudioData
from .jsonl import JsonlLoader, JsonlSeries
from .ir import IRRawLoader
from .rgb import RGBLoader


@dataclass
class RobotData:
    series: JsonlSeries

    @property
    def arc_on(self) -> np.ndarray:
        values = self.series.values.get("ArcOn", np.array([]))
        return values.astype(int) if len(values) else values

    @property
    def path_xy(self) -> np.ndarray:
        x = self.series.values.get("Pose.X", np.array([]))
        y = self.series.values.get("Pose.Y", np.array([]))
        return np.column_stack([x, y]) if len(x) and len(y) else np.empty((0, 2))

    @property
    def path_xyz(self) -> np.ndarray:
        x = self.series.values.get("Pose.X", np.array([]))
        y = self.series.values.get("Pose.Y", np.array([]))
        z = self.series.values.get("Pose.Z", np.array([]))
        if len(x) and len(y) and len(z):
            return np.column_stack([x, y, z])
        return np.empty((0, 3))

    @property
    def joints(self) -> Dict[str, np.ndarray]:
        return {
            f"A{i}": self.series.values.get(f"Joints.A{i}", np.array([]))
            for i in range(1, 7)
        }

    @property
    def velocity(self) -> np.ndarray:
        path = self.path_xyz
        times = self.series.times
        if len(path) < 2 or len(times) != len(path):
            return np.zeros(len(path))
        
        # Central differencing for each coordinate using np.gradient
        # This provides v_x = dx/dt, etc.
        vx = np.gradient(path[:, 0], times)
        vy = np.gradient(path[:, 1], times)
        vz = np.gradient(path[:, 2], times)
        
        return np.sqrt(vx**2 + vy**2 + vz**2)


@dataclass
class ModelData:
    series: JsonlSeries

    @property
    def laser_on(self) -> np.ndarray:
        values = self.series.values.get("LaserOn", np.array([]))
        return values.astype(int) if len(values) else values

    @property
    def path_xy(self) -> np.ndarray:
        x = self.series.values.get("X", np.array([]))
        y = self.series.values.get("Y", np.array([]))
        return np.column_stack([x, y]) if len(x) and len(y) else np.empty((0, 2))

    @property
    def path_xyz(self) -> np.ndarray:
        x = self.series.values.get("X", np.array([]))
        y = self.series.values.get("Y", np.array([]))
        z = self.series.values.get("Z", np.array([]))
        if len(x) and len(y) and len(z):
            return np.column_stack([x, y, z])
        return np.empty((0, 3))


@dataclass
class IRData:
    json_series: JsonlSeries
    raw_reader: Optional[IRRawLoader]


class SessionData:
    def __init__(self, session_path: Path):
        self.path = Path(session_path)
        self.robot = RobotData(
            JsonlLoader(self.path / "Robot" / "0.jsonl").load(
                [
                    "ArcOn", "Pose.X", "Pose.Y", "Pose.Z",
                    "Joints.A1", "Joints.A2", "Joints.A3",
                    "Joints.A4", "Joints.A5", "Joints.A6"
                ]
            )
        )
        self.model = ModelData(
            JsonlLoader(self.path / "Model" / "0.jsonl").load(
                ["LaserOn", "X", "Y", "Z", "Layer"]
            )
        )
        self.audio = AudioLoader(self.path / "Audio" / "0.wav").run()
        self.ir_high = IRData(
            json_series=JsonlLoader(self.path / "IR_High" / "0.jsonl").load(
                ["MaxTemp", "PeakX", "PeakY", "Area", "ContourPoints"]
            ),
            raw_reader=IRRawLoader(self.path / "IR_High" / "0.raw", (1100, 2000)),
        )
        self.ir_low = IRData(
            json_series=JsonlSeries(times=np.array([]), values={}),
            raw_reader=IRRawLoader(self.path / "IR_Low" / "0.raw", (100, 950)),
        )
        self.rgb_sequence = RGBLoader(self.path / "Image")
        self.start_time = self._compute_start_time()

    @property
    def duration(self) -> float:
        candidates = [
            self.robot.series.times.max() if len(self.robot.series.times) else 0.0,
            self.model.series.times.max() if len(self.model.series.times) else 0.0,
            self.audio.duration,
        ]
        if len(self.ir_high.json_series.times):
            candidates.append(self.ir_high.json_series.times.max())
        if self.ir_high.raw_reader and len(self.ir_high.raw_reader.timestamps):
            candidates.append(self.ir_high.raw_reader.timestamps[-1])
        if self.ir_low.raw_reader and len(self.ir_low.raw_reader.timestamps):
            candidates.append(self.ir_low.raw_reader.timestamps[-1])
        return max(candidates) if candidates else 0.0

    @property
    def arc_on_range(self) -> tuple[Optional[float], Optional[float]]:
        arc = self.robot.arc_on
        times = self.robot.series.times
        if len(arc) == 0:
            return None, None
        
        indices = np.where(arc > 0)[0]
        if len(indices) == 0:
            return None, None
        
        return times[indices[0]], times[indices[-1]]

    def _compute_start_time(self) -> float:
        first_active, _ = self.arc_on_range
        if first_active is None:
            return 0.0
        return max(0.0, first_active)

    def get_ir_high_metadata(self, timestamp: float):
        series = self.ir_high.json_series
        if len(series.times) == 0:
            return None
        idx = int(np.argmin(np.abs(series.times - timestamp)))
        data = {"Time": series.times[idx]}
        for key, arr in series.values.items():
            if len(arr) > idx:
                value = arr[idx]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                data[key] = value
        return data
