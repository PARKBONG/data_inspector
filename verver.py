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
        q0_ctwd: float = 1e-4,
        q0_rate: float = 1e-2,
        q1_ctwd: float = 5e-4,
        q1_rate: float = 5e-2,
        # measurement base noise (mm^2)
        r_base: float = 1.2**2,
        # mixture / mode
        p_stay: float = 0.98,
        p_switch: float = 0.02,
        # adaptive measurement scaling
        reacq_frames: int = 10,
        reacq_r_scale: float = 20.0,
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

    def reset(self):
        self.kf0.reset()
        self.kf1.reset()
        self.mu[:] = [0.8, 0.2]
        self._reacq_countdown = 0
        self.valid = False

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
        Args:
          t: time (s)
          peak_v: PeakY (pixel) after time-consistency (or None)
          arc_on: whether arc is on
          q_peak: quality in [0,1] from peak tracker

        Returns:
          d_hat (mm), d_dot_hat (mm/s), mu (2,), used_update (bool)
        """
        t = float(t)
        arc_on = bool(arc_on)
        q_peak = float(np.clip(q_peak, 0.0, 1.0))

        # Predict always if initialized
        if self.valid:
            self.kf0.predict(t)
            self.kf1.predict(t)

        # If arc is off -> no measurement update (safe for control)
        if (not arc_on) or (peak_v is None) or (not np.isfinite(peak_v)):
            self._reacq_countdown = self.reacq_frames
            # Output mixed state if valid, else NaN
            if not self.valid:
                return (np.nan, np.nan, self.mu.copy(), False)
            x_mix = self.mu[0] * self.kf0.x + self.mu[1] * self.kf1.x
            return (float(x_mix[0]), float(x_mix[1]), self.mu.copy(), False)

        # Measurement in mm
        z = self.a * float(peak_v) + self.b

        # Initialize on first meaningful arc-on measurement
        if not self.valid:
            if z < self.init_ctwd_min_mm:
                return (np.nan, np.nan, self.mu.copy(), False)
            self.kf0.init(t, z, p_d=max(self.r_base, 4.0), p_ddot=25.0)
            self.kf1.init(t, z, p_d=max(self.r_base, 9.0), p_ddot=49.0)
            self.valid = True
            return (float(z), 0.0, self.mu.copy(), True)

        # Base R adapted by peak quality (lower q -> larger R)
        # Example: q=1 -> scale 1, q=0.2 -> scale 1/0.2^2=25
        q_floor = 0.15
        q_eff = max(q_floor, q_peak)
        Rq = self.r_base / (q_eff ** 2)

        # Reacquisition scaling after dropout/arc-off
        if self._reacq_countdown > 0:
            Rq *= self.reacq_r_scale
            self._reacq_countdown -= 1

        # IMM mixing (prior)
        mu_prev = self.mu.copy()
        c0 = self.p00 * mu_prev[0] + self.p10 * mu_prev[1]
        c1 = self.p01 * mu_prev[0] + self.p11 * mu_prev[1]
        c0 = max(1e-12, c0)
        c1 = max(1e-12, c1)
        # Mixing probabilities
        mu0_given0 = self.p00 * mu_prev[0] / c0
        mu1_given0 = self.p10 * mu_prev[1] / c0
        mu0_given1 = self.p01 * mu_prev[0] / c1
        mu1_given1 = self.p11 * mu_prev[1] / c1

        # Mixed initial conditions for each model (state only; covariance mixing omitted for simplicity)
        x0_mix = mu0_given0 * self.kf0.x + mu1_given0 * self.kf1.x
        x1_mix = mu0_given1 * self.kf0.x + mu1_given1 * self.kf1.x

        # Set mixed states (covariance mixing can be added, but this already works well in practice)
        self.kf0.x = x0_mix.copy()
        self.kf1.x = x1_mix.copy()

        # Model-specific measurement noise:
        # - model0: normal (Rq)
        # - model1: contaminated/jumpy (inflate R more)
        R0 = float(Rq)
        R1 = float(Rq * 100)

        # Soft robust weighting: increase R based on each model's innovation magnitude
        r0_pred = z - float(self.kf0.x[0])
        r1_pred = z - float(self.kf1.x[0])
        R0_eff = self._soft_R(R0, r0_pred)
        R1_eff = self._soft_R(R1, r1_pred)

        # Update both models
        r0, S0 = self.kf0.update(z, R0_eff)
        r1, S1 = self.kf1.update(z, R1_eff)

        # Mode likelihoods
        L0 = self._gaussian_likelihood(r0, S0)
        L1 = self._gaussian_likelihood(r1, S1)

        # Posterior mode probabilities
        mu_post_unnorm = np.array([c0 * L0, c1 * L1], dtype=float)
        s = float(mu_post_unnorm.sum())
        if s < 1e-24:
            self.mu[:] = [0.5, 0.5]
        else:
            self.mu = mu_post_unnorm / s

        # Mixed output
        x_mix = self.mu[0] * self.kf0.x + self.mu[1] * self.kf1.x
        return (float(x_mix[0]), float(x_mix[1]), self.mu.copy(), True)


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
    q_arr = np.zeros_like(times, dtype=float)
    mu0 = np.zeros_like(times, dtype=float)
    mu1 = np.zeros_like(times, dtype=float)

    peak_proc.reset()
    imm.reset()

    for i, t in enumerate(times):
        v = float(peak_y[i]) if np.isfinite(peak_y[i]) else None
        arc_on = bool(arc_on_ir[i])

        # -- Previous implementation with fused peakY --
        # v_fused, q = peak_proc.step(t, v, arc_on)
        # q_arr[i] = q
        # if v_fused is not None and np.isfinite(v_fused):
        #     ctwd_raw[i] = a * v_fused + b
        # d_hat, d_dot_hat, mu, used = imm.step(t, v_fused, arc_on, q)

        # -- Current implementation using raw peakY --
        q = 1.0
        q_arr[i] = q
        if v is not None:
            ctwd_raw[i] = a * v + b
        d_hat, d_dot_hat, mu, used = imm.step(t, v, arc_on, q)
        
        ctwd_est[i] = d_hat
        mu0[i], mu1[i] = float(mu[0]), float(mu[1])

    return ctwd_raw, ctwd_est, q_arr, mu0, mu1, arc_on_ir.astype(float)


def build_fig(session: SessionData, a: float, b: float):
    times, peak_y = extract_peak_y_series(session)
    robot_times, arc_on = extract_robot_arc_series(session)

    ctwd_raw, ctwd_est, q_peak, mu0, mu1, arc_on_ir = run_ctwd_pipeline(
        times, peak_y, robot_times, arc_on, a, b
    )

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(times, peak_y, linewidth=1, label="PeakY raw")
    axes[0].set_title(f"CTWD Robust Estimation (Arc-aware + Adaptive R + Soft + IMM): {session.path.name}")
    axes[0].set_ylabel("PeakY (px)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(times, arc_on_ir, drawstyle="steps-post", linewidth=1.5, label="ArcOn (IR time)")
    axes[1].set_ylabel("Arc (0/1)")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    axes[2].plot(times, ctwd_raw, linewidth=1, alpha=0.7, label="CTWD raw (from fused PeakY)")
    axes[2].plot(times, ctwd_est, linewidth=2, label="CTWD estimate (IMM robust)")
    axes[2].set_ylabel("CTWD (mm)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    axes[3].plot(times, q_peak, linewidth=1.5, label="Peak quality q(t)")
    axes[3].set_ylabel("q (0..1)")
    axes[3].set_yticks([0, 0.5, 1.0])
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend()

    axes[4].plot(times, mu0, linewidth=1.5, label="Mode prob: normal")
    axes[4].plot(times, mu1, linewidth=1.5, label="Mode prob: contaminated")
    axes[4].set_ylabel("IMM prob")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_yticks([0, 0.5, 1.0])
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