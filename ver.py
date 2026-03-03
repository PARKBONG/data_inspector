from pathlib import Path
import re
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from loader.session import SessionData

SESSION_PATTERN = re.compile(r"^\d{6}_\d{6}$")


def discover_sessions(db_root: Path) -> Dict[str, Path]:
    sessions: Dict[str, Path] = {}
    if not db_root.exists():
        return {}
    for path in db_root.rglob("*"):
        if path.is_dir() and SESSION_PATTERN.fullmatch(path.name):
            sessions[path.name] = path
    return dict(sorted(sessions.items()))


class RobustCTWDEstimator:
    """
    Robust CTWD estimator:
      - state: x = [ctwd, ctwd_rate]^T
      - prediction: constant velocity model with variable dt
      - measurement: z = ctwd_meas = a * peak_v + b
      - handles:
          * missing measurements -> skip update
          * outlier jump -> innovation gating -> skip update
          * after-dropout stabilization window -> temporarily inflate R
    """

    def __init__(
        self,
        q_ctwd: float = 1e-4,
        q_rate: float = 1e-2,
        r_meas: float = 1.0,
        gate_ctwd_mm: float = 1.0,
        reacq_frames: int = 5,
        reacq_r_scale: float = 10.0,
        init_ctwd_min_mm: float = 8.0,
    ):
        # Process noise (tune)
        self.q_ctwd = float(q_ctwd)
        self.q_rate = float(q_rate)

        # Measurement noise variance (mm^2)
        self.r_meas = float(r_meas)

        # Gating threshold in CTWD domain (mm)
        self.gate_ctwd_mm = float(gate_ctwd_mm)

        # After dropout: for next reacq_frames, use R *= reacq_r_scale
        self.reacq_frames = int(reacq_frames)
        self.reacq_r_scale = float(reacq_r_scale)
        self._reacq_countdown = 0

        # Initial CTWD threshold for starting filter
        self.init_ctwd_min_mm = float(init_ctwd_min_mm)

        # Filter state
        self.x: Optional[np.ndarray] = None  # shape (2,)
        self.P: Optional[np.ndarray] = None  # shape (2,2)
        self._last_t: Optional[float] = None

    def reset(self):
        self.x = None
        self.P = None
        self._last_t = None
        self._reacq_countdown = 0

    def _init_from_measurement(self, t: float, z: float):
        # Initialize state with zero velocity
        self.x = np.array([z, 0.0], dtype=float)
        # Conservative initial covariance
        self.P = np.diag([max(self.r_meas, 1.0), 10.0])
        self._last_t = float(t)

    def step(
        self, t: float, z: Optional[float]
    ) -> Tuple[float, float, bool, bool]:
        """
        One step update.
        Args:
            t: time in seconds
            z: ctwd measurement in mm (None if missing)
        Returns:
            ctwd_est, ctwd_rate_est, used_update, rejected_as_outlier
        """
        t = float(t)

        # If not initialized, try to initialize when measurement exists
        if self.x is None or self.P is None or self._last_t is None:
            if z is None or not np.isfinite(z):
                # cannot initialize yet; output NaN
                return (np.nan, np.nan, False, False)
            
            if z < self.init_ctwd_min_mm:
                # ignore pre-arc meaningless measurements
                return (np.nan, np.nan, False, False)

            self._init_from_measurement(t, float(z))
            return (self.x[0], self.x[1], True, False)

        # dt
        dt = t - self._last_t
        if dt <= 0:
            # Non-increasing time: do not update time, just return current
            return (self.x[0], self.x[1], False, False)

        # ---- Prediction ----
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)

        # Simple diagonal continuous-time-ish noise -> discretize crudely
        # You can refine this if needed; this is robust enough in practice.
        Q = np.array([[self.q_ctwd * dt, 0.0],
                      [0.0, self.q_rate * dt]], dtype=float)

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # ---- If missing measurement: skip update ----
        if z is None or (not np.isfinite(z)):
            # Start/refresh reacquisition mode
            self._reacq_countdown = self.reacq_frames
            self.x, self.P = x_pred, P_pred
            self._last_t = t
            return (self.x[0], self.x[1], False, False)

        z = float(z)

        # ---- Innovation gating (outlier rejection) ----
        H = np.array([[1.0, 0.0]], dtype=float)  # observe ctwd only
        z_pred = float((H @ x_pred).item())
        r = z - z_pred  # innovation in mm

        if abs(r) > self.gate_ctwd_mm:
            # Reject as outlier: do not use measurement
            self.x, self.P = x_pred, P_pred
            self._last_t = t
            return (self.x[0], self.x[1], False, True)

        # ---- Measurement update ----
        R = self.r_meas
        if self._reacq_countdown > 0:
            R *= self.reacq_r_scale
            self._reacq_countdown -= 1

        S = float((H @ P_pred @ H.T).item() + R)
        K = (P_pred @ H.T) / S           # shape (2,1)

        x_upd = x_pred + (K.flatten() * r)
        P_upd = (np.eye(2) - K @ H) @ P_pred

        self.x, self.P = x_upd, P_upd
        self._last_t = t
        return (self.x[0], self.x[1], True, False)


def extract_peak_y_series(session: SessionData) -> Tuple[np.ndarray, np.ndarray]:
    series = session.ir_high.json_series
    times = np.asarray(series.times, dtype=float)
    peak_y = np.asarray(series.values.get("PeakY", []), dtype=float)
    # Make sure lengths align
    n = min(len(times), len(peak_y))
    return times[:n], peak_y[:n]


def extract_robot_arc_series(session: SessionData) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(session.robot.series.times, dtype=float)
    arc_on = np.asarray(session.robot.arc_on, dtype=float)
    return times, arc_on


def robust_ctwd_from_peak(
    times: np.ndarray,
    peak_v: np.ndarray,
    a: float,
    b: float,
    estimator: RobustCTWDEstimator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      ctwd_meas: a*v+b (NaN if v missing)
      ctwd_est: filtered estimate
      used_update: 1 if measurement update used
      rejected: 1 if rejected as outlier
    """
    ctwd_meas = np.full_like(times, np.nan, dtype=float)
    ctwd_est = np.full_like(times, np.nan, dtype=float)
    used_update = np.zeros_like(times, dtype=np.int32)
    rejected = np.zeros_like(times, dtype=np.int32)

    estimator.reset()

    for i, (t, v) in enumerate(zip(times, peak_v)):
        z = None
        if np.isfinite(v):
            z = a * float(v) + b
            ctwd_meas[i] = z

        d_hat, d_dot_hat, used, rej = estimator.step(float(t), z)
        ctwd_est[i] = d_hat
        used_update[i] = int(used)
        rejected[i] = int(rej)

    return ctwd_meas, ctwd_est, used_update, rejected


def build_ctwd_fig(
    session: SessionData,
    a: float,
    b: float,
    estimator: RobustCTWDEstimator,
):
    times, peak_y = extract_peak_y_series(session)
    robot_times, arc_on = extract_robot_arc_series(session)

    ctwd_meas, ctwd_est, used_update, rejected = robust_ctwd_from_peak(
        times, peak_y, a, b, estimator
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1) PeakY
    axes[0].plot(times, peak_y, linewidth=1, label="PeakY")
    axes[0].set_title(f"Robust CTWD Estimation: {session.path.name}")
    axes[0].set_ylabel("PeakY (pixel)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2) CTWD raw vs filtered
    axes[1].plot(times, ctwd_meas, linewidth=1, label="CTWD raw", alpha=0.7)
    axes[1].plot(times, ctwd_est, linewidth=2, label="CTWD filtered")
    axes[1].set_ylabel("CTWD (mm)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3) Arc On/Off
    axes[2].plot(robot_times, arc_on, color='red', linewidth=1.5, drawstyle='steps-post', label='Arc On')
    axes[2].set_ylabel("Arc On (0/1)")
    axes[2].set_yticks([0, 1])
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    # 4) Diagnostics: used update / rejected
    axes[3].plot(times, used_update, linewidth=1, label="Used update")
    axes[3].plot(times, rejected, linewidth=1, label="Rejected")
    axes[3].set_ylabel("Flags")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yticks([0, 1])
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend()

    # Add vertical lines for Arc transitions
    first_arc, last_arc = session.arc_on_range
    for ax in axes:
        if first_arc is not None:
            ax.axvline(x=first_arc, color='green', linestyle='--', alpha=0.8)
        if last_arc is not None:
            ax.axvline(x=last_arc, color='orange', linestyle='--', alpha=0.8)

    plt.tight_layout()
    return fig


def main():
    # settings
    db_root = Path("DB")
    session_id = None  # None -> pick first session

    # set your calibrated mapping here
    a = 0.437  # [mm/pixel]  <-- REPLACE
    b = 1.571   # [mm]        <-- REPLACE

    # Robust estimator hyperparameters (start here, then tune)
    estimator = RobustCTWDEstimator(
        q_ctwd=1e-4,
        q_rate=1e-2,
        r_meas=1.2**2,        # measurement noise variance in mm^2
        gate_ctwd_mm=1.0,       # reject if |innovation| > 3.0 mm
        reacq_frames=10,       # after dropout, first 5 frames are downweighted
        reacq_r_scale=20.0,
        init_ctwd_min_mm=10.0,
    )

    session_paths = discover_sessions(db_root)
    if not session_paths:
        print(f"No sessions found under {db_root}")
        return

    if session_id is None:
        session_id = next(iter(session_paths))

    if session_id not in session_paths:
        print(f"Session {session_id} not found.")
        return

    print(f"Loading session: {session_id}")
    session = SessionData(session_paths[session_id])

    build_ctwd_fig(session, a=a, b=b, estimator=estimator)
    plt.show()


if __name__ == "__main__":
    main()