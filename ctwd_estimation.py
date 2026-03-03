import numpy as np
from typing import Optional, Tuple, Protocol, List
from pathlib import Path
import re
import matplotlib.pyplot as plt
from loader.session import SessionData

class DynamicModel(Protocol):
    """Protocol for state transition models."""
    def predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ...

class ConstantVelocityModel:
    """Standard 1D Constant Velocity Model (state: [pos, vel])."""
    def __init__(self, q_pos: float = 1e-4, q_vel: float = 1e-2):
        self.q_pos = q_pos
        self.q_vel = q_vel

    def predict(self, x: np.ndarray, P: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if dt <= 0:
            return x, P
        
        # State transition matrix
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        
        # Process noise covariance (crude discretization)
        Q = np.array([[self.q_pos * dt, 0.0],
                      [0.0, self.q_vel * dt]], dtype=float)
        
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

class MeasurementModel(Protocol):
    """Protocol for measurement models."""
    def observe(self, x: np.ndarray) -> float:
        ...
    def get_H(self, x: np.ndarray) -> np.ndarray:
        ...
    def get_R(self) -> float:
        ...

class PositionMeasurementModel:
    """Observes the position component of the state [pos, vel]."""
    def __init__(self, r_meas: float = 1.0):
        self.r_meas = r_meas
        self.H = np.array([[1.0, 0.0]], dtype=float)

    def observe(self, x: np.ndarray) -> float:
        return float((self.H @ x).item())

    def get_H(self, x: np.ndarray) -> np.ndarray:
        return self.H

    def get_R(self) -> float:
        return self.r_meas

class RobustEstimator:
    """
    A modular Kalman Filter with outlier rejection and reacquisition logic.
    """
    def __init__(
        self,
        dynamic_model: DynamicModel,
        measurement_model: MeasurementModel,
        gate_threshold: float = 1.0,
        reacq_frames: int = 5,
        reacq_r_scale: float = 10.0,
        init_threshold: float = 8.0,
    ):
        self.dyn = dynamic_model
        self.meas = measurement_model
        
        self.gate_threshold = gate_threshold
        self.reacq_frames = reacq_frames
        self.reacq_r_scale = reacq_r_scale
        self.init_threshold = init_threshold

        # Filter state
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self._reacq_countdown = 0

    def reset(self):
        self.x = None
        self.P = None
        self._last_t = None
        self._reacq_countdown = 0

    def _initialize(self, t: float, z: float):
        # State: [pos, vel]
        self.x = np.array([z, 0.0], dtype=float)
        # Initial covariance: position matches measurement noise, velocity is uncertain
        r_init = self.meas.get_R()
        self.P = np.diag([max(r_init, 1.0), 10.0])
        self._last_t = t

    def step(self, t: float, z: Optional[float]) -> Tuple[float, float, bool, bool]:
        """
        Perform one prediction-update step.
        Returns:
            (pos_est, vel_est, used_update, rejected)
        """
        # 1. Initialization check
        if self.x is None:
            if z is None or not np.isfinite(z) or z < self.init_threshold:
                return (np.nan, np.nan, False, False)
            self._initialize(t, z)
            return (self.x[0], self.x[1], True, False)

        # 2. Prediction
        dt = t - self._last_t
        if dt < 0:
            return (self.x[0], self.x[1], False, False)
        
        x_pred, P_pred = self.dyn.predict(self.x, self.P, dt)

        # 3. Handle missing measurement
        if z is None or not np.isfinite(z):
            self._reacq_countdown = self.reacq_frames
            self.x, self.P = x_pred, P_pred
            self._last_t = t
            return (self.x[0], self.x[1], False, False)

        # 4. Measurement Update with Gating
        z_pred = self.meas.observe(x_pred)
        H = self.meas.get_H(x_pred)
        innov = z - z_pred

        if abs(innov) > self.gate_threshold:
            # Outlier rejection
            self.x, self.P = x_pred, P_pred
            self._last_t = t
            return (self.x[0], self.x[1], False, True)

        # 5. Kalman Gain and Update
        R = self.meas.get_R()
        if self._reacq_countdown > 0:
            R *= self.reacq_r_scale
            self._reacq_countdown -= 1

        S = float((H @ P_pred @ H.T).item() + R)
        K = (P_pred @ H.T) / S
        
        self.x = x_pred + (K.flatten() * innov)
        self.P = (np.eye(len(self.x)) - K @ H) @ P_pred
        self._last_t = t

        return (self.x[0], self.x[1], True, False)


def run_estimation(
    times: np.ndarray,
    measurements: np.ndarray,
    estimator: RobustEstimator
) -> dict:
    """ Utility to run estimator over a series of measurements. """
    n = len(times)
    est_pos = np.full(n, np.nan)
    est_vel = np.full(n, np.nan)
    used_update = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)

    estimator.reset()
    for i in range(n):
        p, v, used, rej = estimator.step(times[i], measurements[i])
        est_pos[i] = p
        est_vel[i] = v
        used_update[i] = used
        rejected[i] = rej

    return {
        "pos": est_pos,
        "vel": est_vel,
        "used_update": used_update,
        "rejected": rejected
    }

class SessionProcessor:
    """Helper class to extract data from a SessionData and run estimation."""
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def extract_series(self, session) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Extracts (times, peak_y, robot_times, arc_on) """
        # PeakY series
        series = session.ir_high.json_series
        times = np.asarray(series.times, dtype=float)
        peak_y = np.asarray(series.values.get("PeakY", []), dtype=float)
        n = min(len(times), len(peak_y))
        times, peak_y = times[:n], peak_y[:n]

        # Robot Arc series
        robot_times = np.asarray(session.robot.series.times, dtype=float)
        arc_on = np.asarray(session.robot.arc_on, dtype=float)
        
        return times, peak_y, robot_times, arc_on

    def process_session(self, session, estimator: RobustEstimator):
        times, peak_y, robot_times, arc_on = self.extract_series(session)
        
        # Raw CTWD measurement: z = a * peak_y + b
        z_raw = np.full_like(peak_y, np.nan)
        mask = np.isfinite(peak_y)
        z_raw[mask] = self.a * peak_y[mask] + self.b

        results = run_estimation(times, z_raw, estimator)
        results["z_raw"] = z_raw
        results["times"] = times
        results["robot_times"] = robot_times
        results["arc_on"] = arc_on
        results["session_id"] = session.path.name
        results["arc_range"] = session.arc_on_range
        
        return results

def build_ctwd_fig(results: dict):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) Raw series vs Filtered
    axes[0].plot(results["times"], results["z_raw"], linewidth=1, label="CTWD raw", alpha=0.7)
    axes[0].plot(results["times"], results["pos"], linewidth=2, label="CTWD filtered")
    axes[0].set_title(f"Modular CTWD Estimation: {results['session_id']}")
    axes[0].set_ylabel("CTWD (mm)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2) Robot Arc
    axes[1].plot(results["robot_times"], results["arc_on"], color='red', linewidth=1.5, drawstyle='steps-post', label='Arc On')
    axes[1].set_ylabel("Arc On (0/1)")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3) Diagnostics
    axes[2].plot(results["times"], results["used_update"].astype(int), linewidth=1, label="Used Update")
    axes[2].plot(results["times"], results["rejected"].astype(int), linewidth=1, label="Rejected (Outlier)")
    axes[2].set_ylabel("Flags")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0, 1])
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    # Add vertical lines for Arc transitions
    first_arc, last_arc = results["arc_range"]
    for ax in axes:
        if first_arc is not None:
            ax.axvline(x=first_arc, color='green', linestyle='--', alpha=0.8)
        if last_arc is not None:
            ax.axvline(x=last_arc, color='orange', linestyle='--', alpha=0.8)

    plt.tight_layout()
    return fig

def main():
    # Example standalone execution
    from loader.session import SessionData
    
    db_root = Path("DB")
    
    # 1. Config
    a, b = 0.437, 1.571
    dyn_model = ConstantVelocityModel(q_pos=1e-4, q_vel=1e-2)
    meas_model = PositionMeasurementModel(r_meas=1.2**2)
    
    estimator = RobustEstimator(
        dynamic_model=dyn_model,
        measurement_model=meas_model,
        gate_threshold=1.0,
        reacq_frames=10,
        reacq_r_scale=20.0,
        init_threshold=10.0
    )
    
    processor = SessionProcessor(a=a, b=b)

    # 2. Pick a session
    sessions = list(db_root.rglob("*"))
    session_paths = [p for p in sessions if p.is_dir() and re.match(r"^\d{6}_\d{6}$", p.name)]
    if not session_paths:
        print("No sessions found.")
        return
    
    target_path = sorted(session_paths)[1]
    print(f"Processing: {target_path.name}")
    session = SessionData(target_path)

    # 3. Run
    results = processor.process_session(session, estimator)

    # 4. Plot
    build_ctwd_fig(results)
    plt.show()

if __name__ == "__main__":
    main()
