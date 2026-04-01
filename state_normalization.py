# state_normalization.py
import numpy as np


class StateNormalization:
    """
    Normalizes UAV state vector for stable learning.

    Expected state format:
        s = [x, y, z, vx, vy, vz]

    Returns task-relative normalized state:
        [
            (target_x - x) / width,
            (y - window_y) / height,
            (z - window_z) / depth,
            vx / v_max,
            vy / v_max,
            vz / v_max,
        ]
    """

    def __init__(self, v_max=5.0, width=100.0, height=100.0, depth=100.0):
        self.v_max = float(v_max) if v_max is not None else 5.0
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)

        if self.v_max <= 1e-6:
            self.v_max = 1.0

        self.width = max(self.width, 1e-6)
        self.height = max(self.height, 1e-6)
        self.depth = max(self.depth, 1e-6)

    def state_normal(self, s, target_x=None, window_y=None, window_z=None):
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        out = np.zeros_like(s, dtype=np.float32)

        x = float(s[0]) if s.shape[0] >= 1 else 0.0
        y = float(s[1]) if s.shape[0] >= 2 else 0.0
        z = float(s[2]) if s.shape[0] >= 3 else 0.0
        vx = float(s[3]) if s.shape[0] >= 4 else 0.0
        vy = float(s[4]) if s.shape[0] >= 5 else 0.0
        vz = float(s[5]) if s.shape[0] >= 6 else 0.0

        tx = float(target_x) if target_x is not None else 0.0
        wy = float(window_y) if window_y is not None else 0.0
        wz = float(window_z) if window_z is not None else 0.0

        if out.shape[0] >= 1:
            out[0] = (tx - x) / self.width
        if out.shape[0] >= 2:
            out[1] = (y - wy) / self.height
        if out.shape[0] >= 3:
            out[2] = (z - wz) / self.depth
        if out.shape[0] >= 4:
            out[3] = vx / self.v_max
        if out.shape[0] >= 5:
            out[4] = vy / self.v_max
        if out.shape[0] >= 6:
            out[5] = vz / self.v_max

        out = np.clip(out, -5.0, 5.0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out