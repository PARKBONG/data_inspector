import numpy as np
from typing import Optional, Tuple, Protocol, List, Dict, Any
from pathlib import Path
import re
import matplotlib.pyplot as plt
from loader.session import SessionData

# --- 1. Protocols & Component Classes (Top level for visibility) ---

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
        
        # Enforce symmetry (Better than just P_pred = F @ P @ F.T + Q)
        # Numerical errors can make P slightly non-symmetric, which breaks KF logic.
        P_pred = 0.5 * (P_pred + P_pred.T)
        
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

class AdaptiveMeasurementModel:
    """
    Measurement model that adjusts R based on a per-frame confidence score.
    Higher confidence (towards 1.0) keeps R near base value.
    Lower confidence (towards 0.0) inflates R to ignore the measurement.
    """
    def __init__(self, r_base: float = 1.0):
        self.r_base = r_base
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.current_confidence = 1.0

    def set_confidence(self, score: float):
        self.current_confidence = np.clip(score, 1e-6, 1.0)

    def observe(self, x: np.ndarray) -> float:
        return float((self.H @ x).item())

    def get_H(self, x: np.ndarray) -> np.ndarray:
        return self.H

    def get_R(self) -> float:
        # Increase noise R when confidence is low
        return self.r_base / self.current_confidence

# --- 2. Data Infrastructure (Decoupled) ---

class SessionDataLoader:
    """Responsibility: Extract raw data from SessionData object."""
    @staticmethod
    def load(session: SessionData) -> Dict[str, Any]:
        """ Extracts (times, peak_y, robot_times, arc_on) into a dictionary. """
        # PeakY and other series
        series = session.ir_high.json_series
        keys = list(series.values.keys())
        # print(f"DEBUG: Found keys in Session: {keys[:10]}") # Debugging
        
        times = np.asarray(series.times, dtype=float)
        peak_y = np.asarray(series.values.get("PeakY", []), dtype=float)
        peak_x = np.asarray(series.values.get("PeakX", []), dtype=float)
        area = np.asarray(series.values.get("Area", []), dtype=float) # Changed from ContourArea
        
        n = min(len(times), len(peak_y))
        
        # If keys missing, fallback to empty arrays
        if len(peak_x) == 0: peak_x = np.full(n, 320.0) # Assume center if missing
        if len(area) == 0: area = np.full(n, 1000.0)    # Assume medium area if missing
        
        # Make sure same length
        peak_x = peak_x[:n]
        area = area[:n]
        
        # Robot Arc series
        robot_times = np.asarray(session.robot.series.times, dtype=float)
        arc_on = np.asarray(session.robot.arc_on, dtype=float)
        
        return {
            "times": times[:n],
            "peak_y": peak_y[:n],
            "peak_x": peak_x,
            "area": area,
            "robot_times": robot_times,
            "arc_on": arc_on,
            "session_id": session.path.name,
            "arc_range": session.arc_on_range
        }

class CmtConfidenceScorer:
    """Analyzes CMT arc phases to determine data reliability."""
    def __init__(self, min_area: float = 200.0, max_area: float = 3000.0):
        self.min_area = min_area
        self.max_area = max_area

    def calculate(self, area: float, peak_x: float, img_width: float = 72.0) -> float:
        # Metric 1: Arc Size. 
        # Use log scale for area since CMT area can vary by orders of magnitude.
        if area <= 0: return 0.1
        
        log_area = np.log10(area)
        log_min = np.log10(max(1.0, self.min_area))
        log_max = np.log10(max(1.0, self.max_area))
        
        area_score = np.clip((log_area - log_min) / (log_max - log_min), 0.1, 1.0)
        
        # Metric 2: Horizontal stability. Off-center peaks are less reliable.
        dist = abs(peak_x - img_width/2)
        x_score = np.clip(1.0 - (dist / (img_width/2)), 0.3, 1.0)
        
        return float(area_score * x_score)

class EstimationProcessor:
    """Responsibility: Convert raw sensor data to physical units and run estimation."""
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def process_data(self, data: Dict[str, Any], estimator: 'RobustEstimator', scorer: Optional[CmtConfidenceScorer] = None) -> Dict[str, Any]:
        times = data["times"]
        peak_y = data["peak_y"]
        peak_x = data["peak_x"]
        area = data["area"]
        
        # 1. Physical Conversion (Calibration)
        z_raw = np.full_like(peak_y, np.nan)
        mask = np.isfinite(peak_y)
        z_raw[mask] = self.a * peak_y[mask] + self.b

        # 2. Compute Confidence for each frame
        conf_scores = np.ones_like(times)
        if scorer is not None:
            for i in range(len(times)):
                conf_scores[i] = scorer.calculate(area[i], peak_x[i])

        # 3. Run Filter
        results = run_estimation(times, z_raw, estimator, conf_scores)
        
        # 4. Combine results
        results.update({
            "z_raw": z_raw,
            "times": times,
            "conf_scores": conf_scores,
            "robot_times": data["robot_times"],
            "arc_on": data["arc_on"],
            "session_id": data["session_id"],
            "arc_range": data["arc_range"]
        })
        return results

# --- 3. Core Estimator Logic ---

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
        init_confirm_frames: int = 5,
    ):
        self.dyn = dynamic_model
        self.meas = measurement_model
        
        self.gate_threshold = gate_threshold
        self.reacq_frames = reacq_frames
        self.reacq_r_scale = reacq_r_scale
        self.init_threshold = init_threshold
        self.init_confirm_frames = init_confirm_frames

        # Filter state
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self._reacq_countdown = 0
        self._confirm_counter = 0  # Counter for consecutive valid frames before init

    def reset(self):
        self.x = None
        self.P = None
        self._last_t = None
        self._reacq_countdown = 0
        self._confirm_counter = 0

    def _initialize(self, t: float, z: float):
        # State: [pos, vel]
        self.x = np.array([z, 0.0], dtype=float)
        # Initial covariance: position matches measurement noise, velocity is uncertain
        r_init = self.meas.get_R()
        self.P = np.diag([max(r_init, 1.0), 10.0])
        self._last_t = t

    def step(self, t: float, z: Optional[float], confidence: float = 1.0) -> Tuple[float, float, bool, bool]:
        """
        Perform one prediction-update step.
        Returns: (pos_est, vel_est, used_update, rejected)
        """
        # Update adaptive model if supported
        if hasattr(self.meas, 'set_confidence'):
            self.meas.set_confidence(confidence)
            
        # 1. Initialization check
        if self.x is None:
            # Confirmation Logic: Wait for N consecutive frames above threshold
            # This prevents initializing on a single noise spike (e.g., spatter).
            is_valid = (z is not None) and np.isfinite(z) and (z >= self.init_threshold)
            
            if is_valid:
                self._confirm_counter += 1
            else:
                self._confirm_counter = 0 # Reset if any frame is invalid
                
            if self._confirm_counter >= self.init_confirm_frames:
                self._initialize(t, z)
                return (self.x[0], self.x[1], True, False)
            else:
                return (np.nan, np.nan, False, False)

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
        
        # ---- Numerical Stability: Joseph Form Update ----
        # Traditionally: self.P = (I - K @ H) @ P_pred
        # But (I - KH) can become non-positive-definite due to precision loss.
        # Joseph Form: P = (I - KH)P(I - KH).T + KRK.T is much more robust.
        I = np.eye(len(self.x))
        IKH = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + (K * R) @ K.T
        
        # Enforce symmetry (Prevents drifting due to floating point errors)
        self.P = 0.5 * (P_upd + P_upd.T)
        
        self._last_t = t
        return (self.x[0], self.x[1], True, False)

def run_estimation(
    times: np.ndarray,
    measurements: np.ndarray,
    estimator: RobustEstimator,
    conf_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """ Utility to run estimator over a series of measurements. """
    n = len(times)
    if conf_scores is None:
        conf_scores = np.ones(n)

    est_pos = np.full(n, np.nan)
    est_vel = np.full(n, np.nan)
    used_update = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)

    estimator.reset()
    for i in range(n):
        p, v, used, rej = estimator.step(times[i], measurements[i], conf_scores[i])
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

# --- 4. Visualization & Main ---

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
    axes[2].plot(results["times"], results["conf_scores"], linewidth=1, label="Confidence", color='gray', alpha=0.5)
    axes[2].set_ylabel("Flags / Score")
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
    db_root = Path("DB")
    
    # 1. Config
    a, b = 0.437, 1.571
    
    # 1. Component Setup
    dyn_model = ConstantVelocityModel(q_pos=1e-4, q_vel=1e-2)
    
    # Use Adaptive Measurement Model with Dynamic R
    meas_model = AdaptiveMeasurementModel(r_base=1.2**2)
    
    estimator = RobustEstimator(
        dyn_model, 
        meas_model, 
        gate_threshold=1.0, 
        reacq_frames=10, 
        init_threshold=10.0,
        init_confirm_frames=5
    )
    
    # 2. Data Infrastructure
    loader = SessionDataLoader()
    processor = EstimationProcessor(a=a, b=b)
    # Adjusted based on sample data (Area was ~256)
    scorer = CmtConfidenceScorer(min_area=50, max_area=1500)

    # 3. Execution
    session_paths = [p for p in db_root.rglob("*") if p.is_dir() and re.match(r"^\d{6}_\d{6}$", p.name)]
    if not session_paths:
        print("No sessions found.")
        return
    
    target_path = sorted(session_paths)[1]
    print(f"Processing: {target_path.name}")
    session = SessionData(target_path)

    # Decoupled Flow: Load -> Process (with Confidence Scorer) -> Plot
    raw_data = loader.load(session)
    results = processor.process_data(raw_data, estimator, scorer=scorer)

    # 4. Plot
    build_ctwd_fig(results)
    plt.show()

if __name__ == "__main__":
    main()
