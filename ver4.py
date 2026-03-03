import numpy as np
from typing import Optional, Tuple

class CTWDKFMinimal:
    """
    Minimal CTWD estimator for control (0-lag):
      - state x = [d, d_dot]
      - predict: constant velocity
      - update: z = a*peak_y + b
      - arc_off / missing -> no update, velocity damping
      - reacq -> inflate covariance to re-lock quickly
    """

    def __init__(
        self,
        a: float,
        b: float,
        # process noise
        q_d: float = 1e-4,
        q_v: float = 3e-2,
        # measurement noise (mm^2)
        r_meas: float = 1.2**2,
        # missing detection in pixel domain
        v_min_px: float = 1.0,     # <= this => treat as missing
        v_max_px: float = 1e6,     # optional saturation guard
        # init
        init_ctwd_min_mm: float = 8.0,
        # dropout behavior
        vel_damp: float = 0.85,    # applied when no update
        reacq_frames: int = 10,
        reacq_r_scale: float = 20.0,
        reacq_P_scale_d: float = 20.0,   # inflate P00 on reacq
        reacq_P_scale_v: float = 5.0,    # inflate P11 on reacq
        # safety (optional)
        clamp_min_mm: float = 0.0,
        clamp_max_mm: float = 100.0,
    ):
        self.a = float(a)
        self.b = float(b)

        self.q_d = float(q_d)
        self.q_v = float(q_v)
        self.r_meas = float(r_meas)

        self.v_min_px = float(v_min_px)
        self.v_max_px = float(v_max_px)

        self.init_ctwd_min_mm = float(init_ctwd_min_mm)

        self.vel_damp = float(vel_damp)
        self.reacq_frames = int(reacq_frames)
        self.reacq_r_scale = float(reacq_r_scale)
        self.reacq_P_scale_d = float(reacq_P_scale_d)
        self.reacq_P_scale_v = float(reacq_P_scale_v)

        self.clamp_min_mm = float(clamp_min_mm)
        self.clamp_max_mm = float(clamp_max_mm)

        self.x: Optional[np.ndarray] = None  # [d, d_dot]
        self.P: Optional[np.ndarray] = None  # 2x2
        self.t_last: Optional[float] = None

        self._reacq_countdown = 0
        self._was_missing = True

    def reset(self):
        self.x = None
        self.P = None
        self.t_last = None
        self._reacq_countdown = 0
        self._was_missing = True

    def _is_valid_peak(self, v: Optional[float], arc_on: bool) -> bool:
        if not arc_on or v is None or (not np.isfinite(v)):
            return False
        v = float(v)
        if v <= self.v_min_px:
            return False
        if v >= self.v_max_px:
            return False
        return True

    def _predict(self, t: float):
        if self.x is None or self.P is None or self.t_last is None:
            return
        dt = float(t) - float(self.t_last)
        if dt <= 0:
            return

        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        Q = np.array([[self.q_d * dt, 0.0],
                      [0.0, self.q_v * dt]], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_last = float(t)

    def _update(self, z: float, R: float):
        # H = [1, 0]
        z_pred = float(self.x[0])
        r = float(z) - z_pred
        S = float(self.P[0, 0] + R)

        K0 = self.P[0, 0] / S
        K1 = self.P[1, 0] / S

        self.x[0] = self.x[0] + K0 * r
        self.x[1] = self.x[1] + K1 * r

        # covariance update (scalar measurement)
        P00 = self.P[0, 0] - K0 * self.P[0, 0]
        P01 = self.P[0, 1] - K0 * self.P[0, 1]
        P10 = self.P[1, 0] - K1 * self.P[0, 0]
        P11 = self.P[1, 1] - K1 * self.P[0, 1]
        self.P = np.array([[P00, P01],
                           [P10, P11]], dtype=float)

        return r

    def step(self, t: float, peak_y: Optional[float], arc_on: bool) -> Tuple[float, float, bool]:
        """
        Returns:
          d_hat, d_dot_hat, used_update
        """
        t = float(t)

        # init time if first call
        if self.t_last is None:
            self.t_last = t

        valid = self._is_valid_peak(peak_y, arc_on)

        # If uninitialized: only initialize when valid measurement exists
        if self.x is None or self.P is None:
            if not valid:
                return (np.nan, np.nan, False)
            z = self.a * float(peak_y) + self.b
            if z < self.init_ctwd_min_mm:
                return (np.nan, np.nan, False)
            self.x = np.array([z, 0.0], dtype=float)
            self.P = np.diag([max(self.r_meas, 4.0), 25.0])
            self.t_last = t
            self._was_missing = False
            return (float(self.x[0]), float(self.x[1]), True)

        # predict
        self._predict(t)

        # missing / arc_off -> no update, apply velocity damping
        if not valid:
            self._reacq_countdown = self.reacq_frames
            self._was_missing = True
            self.x[1] *= self.vel_damp
            d = float(np.clip(self.x[0], self.clamp_min_mm, self.clamp_max_mm))
            return (d, float(self.x[1]), False)

        # measurement
        z = self.a * float(peak_y) + self.b

        # reacq handling: first valid after missing -> inflate P, temporarily increase R
        if self._was_missing:
            self.P[0, 0] *= self.reacq_P_scale_d
            self.P[1, 1] *= self.reacq_P_scale_v
            self._was_missing = False

        R = self.r_meas
        if self._reacq_countdown > 0:
            R *= self.reacq_r_scale
            self._reacq_countdown -= 1

        self._update(z, R)

        d = float(np.clip(self.x[0], self.clamp_min_mm, self.clamp_max_mm))
        return (d, float(self.x[1]), True)