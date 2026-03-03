from pathlib import Path
import re
from typing import Dict, Optional, Tuple

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


def stepwise_hold(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """
    Zero-order hold (steps-post): for each dst_t, pick the last src value whose time <= dst_t.
    Assumes src_t is non-decreasing.
    """
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)

    if len(src_t) == 0:
        return np.zeros_like(dst_t)

    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    idx = np.clip(idx, 0, len(src_y) - 1)
    return src_y[idx]


class PeakYTimeConsistency:
    """
    Time-consistent PeakY preprocessor:
      - suppresses implausible jumps in peak_y (pixel domain)
      - outputs a "quality" scalar q in [0,1] used to adapt measurement noise
      - does NOT introduce delay (uses only current and previous accepted)
    """

    def __init__(
        self,
        max_jump_px: float = 8.0,
        max_jump_px_per_s: float = 200.0,
        ema_alpha: float = 0.15,
        missing_value_is_nan: bool = True,
    ):
        self.max_jump_px = float(max_jump_px)
        self.max_jump_px_per_s = float(max_jump_px_per_s)
        self.ema_alpha = float(ema_alpha)
        self.missing_value_is_nan = bool(missing_value_is_nan)

        self._v_hat: Optional[float] = None
        self._t_last: Optional[float] = None

    def reset(self):
        self._v_hat = None
        self._t_last = None

    def step(self, t: float, v_raw: Optional[float], arc_on: bool) -> Tuple[Optional[float], float]:
        """
        Returns:
          v_fused: filtered peak_y (or None if missing / arc_off)
          q: quality in [0,1] (0=bad/untrusted, 1=good)
        """
        t = float(t)

        if (not arc_on) or (v_raw is None) or (not np.isfinite(v_raw)):
            # arc off or missing -> no measurement
            self._t_last = t
            return None, 0.0

        v_raw = float(v_raw)

        # First valid measurement: initialize without penalizing
        # We also want to avoid initializing with 0.0 or garbage early peaks
        if self._v_hat is None or self._t_last is None:
            if v_raw < 5.0:  # Threshold to ignore initial noise/zeros
                self._t_last = t
                return None, 0.0
            self._v_hat = v_raw
            self._t_last = t
            return v_raw, 1.0

        dt = max(1e-6, t - self._t_last)
        self._t_last = t

        # Allowed jump threshold: absolute and rate-based
        allow = min(self.max_jump_px, self.max_jump_px_per_s * dt)
        resid = v_raw - self._v_hat

        # Quality: decay smoothly as residual exceeds allowed band
        # q ~ 1 when |resid| << allow; q -> 0 when |resid| >> allow
        q = float(np.exp(-0.5 * (abs(resid) / (allow + 1e-6)) ** 2))
        q = float(np.clip(q, 0.0, 1.0))

        # Soft clamp the update (no hard gate): limit the effective innovation
        resid_clamped = float(np.clip(resid, -allow, allow))

        # EMA update (time-consistent)
        v_update = self._v_hat + resid_clamped
        self._v_hat = (1.0 - self.ema_alpha) * self._v_hat + self.ema_alpha * v_update

        return float(self._v_hat), q


class KalmanCV1D:
    """
    1D constant-velocity Kalman filter:
      state x = [d, d_dot]
      measurement z = d + noise
    """

    def __init__(self, q_ctwd: float, q_rate: float):
        self.q_ctwd = float(q_ctwd)
        self.q_rate = float(q_rate)
        self.x: Optional[np.ndarray] = None  # shape (2,)
        self.P: Optional[np.ndarray] = None  # shape (2,2)
        self.t_last: Optional[float] = None

    def reset(self):
        self.x = None
        self.P = None
        self.t_last = None

    def init(self, t: float, d0: float, p_d: float = 4.0, p_ddot: float = 25.0):
        self.x = np.array([float(d0), 0.0], dtype=float)
        self.P = np.diag([float(p_d), float(p_ddot)])
        self.t_last = float(t)

    def predict(self, t: float):
        if self.x is None or self.P is None or self.t_last is None:
            return
        t = float(t)
        dt = t - self.t_last
        if dt <= 0:
            return

        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)

        # Simple dt-scaled Q
        Q = np.array([[self.q_ctwd * dt, 0.0],
                      [0.0, self.q_rate * dt]], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_last = t

    def update(self, z: float, R: float) -> Tuple[float, float]:
        """
        Returns:
          innovation r, innovation variance S
        """
        assert self.x is not None and self.P is not None
        z = float(z)
        R = float(R)

        # H = [1, 0]
        z_pred = float(self.x[0])
        r = z - z_pred
        S = float(self.P[0, 0] + R)

        # Kalman gain K = P H^T / S -> [P00, P10]^T / S
        K0 = self.P[0, 0] / S
        K1 = self.P[1, 0] / S

        # Update
        self.x[0] = self.x[0] + K0 * r
        self.x[1] = self.x[1] + K1 * r

        # Joseph form simplified for scalar measurement
        P00 = self.P[0, 0] - K0 * self.P[0, 0]
        P01 = self.P[0, 1] - K0 * self.P[0, 1]
        P10 = self.P[1, 0] - K1 * self.P[0, 0]
        P11 = self.P[1, 1] - K1 * self.P[0, 1]
        self.P = np.array([[P00, P01],
                           [P10, P11]], dtype=float)

        return float(r), float(S)


class IMMRobustCTWD:
    """
    Two-model IMM for robust CTWD estimation.

    Model 0 (normal): trusts measurement with base R (scaled by quality)
    Model 1 (contaminated/jumpy): uses inflated R and/or larger Q (more forgiving)

    Also includes:
      - arc_on/off handling (no update when arc_off)
      - adaptive R based on quality and reacquisition
      - soft weighting based on innovation magnitude (no hard gate)
    """

    def __init__(
        self,
        a: float,
        b: float,
        # dynamics
        q0_ctwd: float = 1e-2,  # 뻣뻣함을 해결하기 위해 상향
        q0_rate: float = 1e-1,
        q1_ctwd: float = 1e-1,  # 점프 시 기동성 확보를 위해 대폭 상향
        q1_rate: float = 5e-1,
        # measurement base noise (mm^2)
        r_base: float = 1.2**2,
        # mixture / mode
        p_stay: float = 0.98,
        p_switch: float = 0.02,
        # adaptive measurement scaling
        reacq_frames: int = 10,
        reacq_r_scale: float = 1.0,  # 데이터 복귀 시 즉시 100% 신뢰
        # soft robust weighting
        robust_c_mm: float = 10.0,
        robust_power: float = 1.0,
        # init
        init_ctwd_min_mm: float = 10.0,
    ):
        self.a = float(a)
        self.b = float(b)

        self.kf0 = KalmanCV1D(q0_ctwd, q0_rate)
        self.kf1 = KalmanCV1D(q1_ctwd, q1_rate)

        self.r_base = float(r_base)

        # Markov transition matrix for 2 modes
        # [p00 p01
        #  p10 p11]
        self.p00 = float(p_stay)
        self.p01 = float(p_switch)
        self.p10 = float(p_switch)
        self.p11 = float(p_stay)

        self.mu = np.array([0.8, 0.2], dtype=float)  # mode probabilities

        self.reacq_frames = int(reacq_frames)
        self.reacq_r_scale = float(reacq_r_scale)
        self._reacq_countdown = 0

        self.robust_c_mm = float(robust_c_mm)
        self.robust_power = float(robust_power)

        self.init_ctwd_min_mm = float(init_ctwd_min_mm)

        self.valid = False
        self._was_holding = False # 결측 후 복귀 감지용

    def reset(self):
        self.kf0.reset()
        self.kf1.reset()
        self.mu[:] = [0.8, 0.2]
        self._reacq_countdown = 0
        self.valid = False
        self._was_holding = False

    @staticmethod
    def _gaussian_likelihood(r: float, S: float) -> float:
        # N(r;0,S)
        S = max(1e-12, float(S))
        return float(np.exp(-0.5 * (r * r) / S) / np.sqrt(2.0 * np.pi * S))

    def _soft_R(self, R: float, r_mm: float) -> float:
        """
        Soft robust weighting: enlarge R as |r| grows.
        Example: R_eff = R * (1 + (|r|/c)^p)
        """
        c = max(1e-6, self.robust_c_mm)
        scale = 1.0 + (abs(float(r_mm)) / c) ** self.robust_power
        return float(R * scale)

    def step(
        self,
        t: float,
        peak_v: Optional[float],
        arc_on: bool,
        q_peak: float,
    ) -> Tuple[float, float, np.ndarray, bool]:
        """
        Simplified Single-Model Robust Filter (IMM Disabled)
        """
        t = float(t)
        arc_on = bool(arc_on)
        q_peak = float(np.clip(q_peak, 0.0, 1.0))

        # Predict if valid
        if self.valid:
            x0_pre_pred = self.kf0.x.copy()
            self.kf0.predict(t)

        # If arc is off or measurement missing -> Hold mode
        if (not arc_on) or (peak_v is None) or (not np.isfinite(peak_v)):
            self._reacq_countdown = self.reacq_frames
            if not self.valid:
                return (np.nan, np.nan, np.array([1.0, 0.0]), False)
            
            # [Hold 로직] 드리프트 차단을 위해 위치를 고정하고 속도를 0으로 설정
            self.kf0.x = x0_pre_pred
            self.kf0.x[1] = 0.0
            self._was_holding = True # 결측 상태 기록
            
            return (float(self.kf0.x[0]), float(self.kf0.x[1]), np.array([1.0, 0.0]), False)

        # Measurement in mm
        z = self.a * float(peak_v) + self.b

        # [회복 메커니즘] 결측 후 첫 복귀 시, 과거 상태를 의심하고 현재 측정값을 세게 수용
        if self._was_holding and self.valid:
            # 공분산을 크게 키워 Kalman Gain을 최대화 (Snap-back 유도)
            self.kf0.P = np.diag([25.0, 100.0])
            self._was_holding = False

        # Initialize
        if not self.valid:
            if z < self.init_ctwd_min_mm:
                return (np.nan, np.nan, np.array([1.0, 0.0]), False)
            self.kf0.init(t, z, p_d=max(self.r_base, 4.0), p_ddot=25.0)
            self.valid = True
            return (float(z), 0.0, np.array([1.0, 0.0]), True)

        # Adaptive R
        Rq = self.r_base / (max(0.15, q_peak) ** 2)
        if self._reacq_countdown > 0:
            Rq *= self.reacq_r_scale
            self._reacq_countdown -= 1

        # Soft robust weighting
        z_pred = float(self.kf0.x[0])
        r_pred = z - z_pred
        R_eff = self._soft_R(Rq, r_pred)

        # Single KF Update
        innov, innov_var = self.kf0.update(z, R_eff)

        # diagnosic info: (innov, innov_var, R_eff)
        return (float(self.kf0.x[0]), float(self.kf0.x[1]), np.array([innov, innov_var, R_eff]), True)


def extract_peak_y_series(session: SessionData) -> Tuple[np.ndarray, np.ndarray]:
    series = session.ir_high.json_series
    times = np.asarray(series.times, dtype=float)
    peak_y = np.asarray(series.values.get("PeakY", []), dtype=float)
    n = min(len(times), len(peak_y))
    return times[:n], peak_y[:n]


def extract_robot_arc_series(session: SessionData) -> Tuple[np.ndarray, np.ndarray]:
    # Adjust these fields if your SessionData differs
    times = np.asarray(session.robot.series.times, dtype=float)
    arc_on = np.asarray(session.robot.arc_on, dtype=float)
    return times, arc_on


def run_ctwd_pipeline(
    times: np.ndarray,
    peak_y: np.ndarray,
    robot_times: np.ndarray,
    arc_on_robot: np.ndarray,
    a: float,
    b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      ctwd_raw: a*peak_y+b (NaN if missing)
      ctwd_est: IMM robust estimate
      q_peak: peak quality
      mu0: mode prob for normal model
      mu1: mode prob for contaminated model
    """
    # Arc-on per IR timestamp
    arc_on_ir = stepwise_hold(robot_times, arc_on_robot, times) > 0.5

    peak_proc = PeakYTimeConsistency(
        max_jump_px=8.0,          # tune
        max_jump_px_per_s=250.0,  # tune
        ema_alpha=0.2,            # tune
    )

    imm = IMMRobustCTWD(
        a=a,
        b=b,
        q0_ctwd=1e-4, q0_rate=1e-2,
        q1_ctwd=5e-4, q1_rate=5e-2,
        r_base=1.2**2,
        p_stay=0.985, p_switch=0.015,
        reacq_frames=10,
        reacq_r_scale=20.0,
        robust_c_mm=3.0,
        robust_power=2.0,
        init_ctwd_min_mm=10.0,
    )

    ctwd_raw = np.full_like(times, np.nan, dtype=float)
    ctwd_est = np.full_like(times, np.nan, dtype=float)
    ctwd_vel = np.zeros_like(times, dtype=float)
    status_flag = np.zeros_like(times, dtype=float) # 1: Update, 0: Hold
    innov_arr = np.zeros_like(times, dtype=float)

    peak_proc.reset()
    imm.reset()

    for i, t in enumerate(times):
        v = float(peak_y[i]) if np.isfinite(peak_y[i]) else None
        arc_on = bool(arc_on_ir[i])

        # Raw PeakY sanity (treat very small values as missing)
        fixed_v = v
        if v is not None and v < 2.0: # peak height < 2px is likely noise
            fixed_v = None

        q = 1.0
        if fixed_v is not None:
            ctwd_raw[i] = a * fixed_v + b
        
        d_hat, d_dot_hat, diag, used = imm.step(t, fixed_v, arc_on, q)
        
        ctwd_est[i] = d_hat
        ctwd_vel[i] = d_dot_hat
        status_flag[i] = 1.0 if used else 0.0
        if used:
            innov_arr[i] = diag[0] # innovation

    return ctwd_raw, ctwd_est, ctwd_vel, status_flag, innov_arr, arc_on_ir.astype(float)


def build_fig(session: SessionData, a: float, b: float):
    times, peak_y = extract_peak_y_series(session)
    robot_times, arc_on = extract_robot_arc_series(session)

    ctwd_raw, ctwd_est, ctwd_vel, status, innov, arc_on_ir = run_ctwd_pipeline(
        times, peak_y, robot_times, arc_on, a, b
    )

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # 1) PeakY & ArcOn
    axes[0].plot(times, peak_y, linewidth=1, label="PeakY raw", alpha=0.5)
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(times, arc_on_ir, 'r--', alpha=0.3, label="ArcOn")
    axes[0].set_title(f"Diagnostic Plot: {session.path.name}")
    axes[0].set_ylabel("PeakY (px)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2) CTWD
    axes[1].plot(times, ctwd_raw, '.', markersize=2, label="CTWD raw", alpha=0.5)
    axes[1].plot(times, ctwd_est, 'k-', linewidth=2, label="CTWD Filtered")
    axes[1].set_ylabel("CTWD (mm)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3) Velocity (Drift check)
    axes[2].plot(times, ctwd_vel, 'g-', label="Estimated Velocity")
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].set_ylabel("Vel (mm/s)")
    axes[2].set_ylim([-50, 50])
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    # 4) Innovation (Error check)
    axes[3].plot(times, innov, 'm.', markersize=3, label="Innovation (z - z_pred)")
    axes[3].set_ylabel("Innov (mm)")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend()

    # 5) Status & Quality
    axes[4].fill_between(times, 0, status, color='cyan', alpha=0.2, label="Action: Update (vs Hold)")
    axes[4].set_ylabel("Status (0=Hold, 1=Update)")
    axes[4].set_yticks([0, 1])
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, linestyle="--", alpha=0.6)
    axes[4].legend()

    plt.tight_layout()
    return fig


def main():
    db_root = Path("DB")
    session_id = None  # None -> pick first session

    # calibrated mapping
    a = 0.437
    b = 1.571

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

    build_fig(session, a=a, b=b)
    plt.show()


if __name__ == "__main__":
    main()