# UAV_window_env.py
import numpy as np


class UAVWindowEnv:
    """
    3D UAV navigation through 3 sequential rectangular windows.

    The UAV must cross the window planes in order:
      window_planes[0] -> window_planes[1] -> window_planes[2]

    A reward is given at each successful crossing.
    Final success happens only after crossing all target windows in order.
    """

    def __init__(
        self,
        box_x=None,
        corridor_start_box_index=2,
        corridor_length=10.0,
        success_reward=200.0,
        width=100.0,
        height=100.0,
        depth=100.0,
        max_steps=220,
        dt=0.5,
        v_max=3.0,
        action_bound=0.7,
        seed=7,
    ):
        self.WIDTH = float(width)
        self.HEIGHT = float(height)
        self.DEPTH = float(depth)

        self.max_steps = int(max_steps)
        self.dt = float(dt)

        self.V_MAX = float(v_max)
        self._action_bound = float(action_bound)

        self.rng = np.random.default_rng(seed)

        if box_x is None:
            box_x = [20.0, 40.0, 60.0, 80.0]
        self.BOX_X = [float(x) for x in box_x]

        self.corridor_start_box_index = int(corridor_start_box_index)
        self.corridor_length = float(corridor_length)
        self.success_reward = float(success_reward)

        self.window_y = self.HEIGHT / 2.0
        self.window_z = self.DEPTH / 2.0
        self.window_w = 20.0
        self.window_h = 20.0

        # 3 sequential targets
        x1 = self._compute_entry_x()
        x2 = x1 + self.corridor_length
        x3 = x2 + self.corridor_length
        self.window_planes = [x1, x2, x3]

        self.state_dim = 6
        self.action_dim = 3
        self.action_bound = np.array([-self._action_bound, self._action_bound], dtype=np.float32)

        self.s = None
        self.steps = 0
        self._done = False
        self.current_target_idx = 0
        self.passed_windows = 0

    def _compute_entry_x(self):
        idx = max(0, min(self.corridor_start_box_index, len(self.BOX_X) - 1))
        return float(self.BOX_X[idx])

    def set_window_size(self, w, h):
        self.window_w = float(w)
        self.window_h = float(h)

        x1 = self._compute_entry_x()
        x2 = x1 + self.corridor_length
        x3 = x2 + self.corridor_length
        self.window_planes = [x1, x2, x3]

    def _inside_rect_yz(self, y, z):
        half_w = self.window_w / 2.0
        half_h = self.window_h / 2.0
        in_y = (self.window_y - half_w) <= y <= (self.window_y + half_w)
        in_z = (self.window_z - half_h) <= z <= (self.window_z + half_h)
        return bool(in_y and in_z)

    def _out_of_bounds(self, x, y, z):
        return (
            x < 0.0 or x > self.WIDTH or
            y < 0.0 or y > self.HEIGHT or
            z < 0.0 or z > self.DEPTH
        )

    def reset(self):
        self.steps = 0
        self._done = False
        self.current_target_idx = 0
        self.passed_windows = 0

        x0 = max(0.0, self.window_planes[0] - 30.0)
        y0 = self.window_y + self.rng.normal(0.0, 5.0)
        z0 = self.window_z + self.rng.normal(0.0, 5.0)

        vx0 = self.rng.normal(0.0, 0.1)
        vy0 = self.rng.normal(0.0, 0.1)
        vz0 = self.rng.normal(0.0, 0.1)

        y0 = float(np.clip(y0, 0.0, self.HEIGHT))
        z0 = float(np.clip(z0, 0.0, self.DEPTH))

        self.s = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float32)
        return self.s.copy()

    def step(self, action):
        if self._done:
            return self.s.copy(), 0.0, True, {"event": "terminal"}

        self.steps += 1

        a = np.asarray(action, dtype=np.float32).reshape(3,)
        a = np.clip(a, -self._action_bound, self._action_bound)

        x, y, z, vx, vy, vz = self.s.tolist()

        vx = float(np.clip(vx + float(a[0]) * self.dt, -self.V_MAX, self.V_MAX))
        vy = float(np.clip(vy + float(a[1]) * self.dt, -self.V_MAX, self.V_MAX))
        vz = float(np.clip(vz + float(a[2]) * self.dt, -self.V_MAX, self.V_MAX))

        x2 = x + vx * self.dt
        y2 = y + vy * self.dt
        z2 = z + vz * self.dt

        reward = -0.1
        info = {"event": None, "passed_windows": self.passed_windows}

        if self._out_of_bounds(x2, y2, z2):
            self._done = True
            reward -= 50.0
            x2 = float(np.clip(x2, 0.0, self.WIDTH))
            y2 = float(np.clip(y2, 0.0, self.HEIGHT))
            z2 = float(np.clip(z2, 0.0, self.DEPTH))
            self.s = np.array([x2, y2, z2, vx, vy, vz], dtype=np.float32)
            return self.s.copy(), float(reward), True, {"event": "out_of_bounds", "passed_windows": self.passed_windows}

        self.s = np.array([x2, y2, z2, vx, vy, vz], dtype=np.float32)

        # Mild centering and stability shaping
        dy = abs(y2 - self.window_y)
        dz = abs(z2 - self.window_z)
        reward -= 0.003 * (dy + dz)
        reward -= 0.02 * (abs(vy) + abs(vz))

        # Sequential window logic
        target_x = self.window_planes[self.current_target_idx]
        crossed_target = (x < target_x) and (x2 >= target_x)
        inside_yz = self._inside_rect_yz(y2, z2)

        if crossed_target and inside_yz:
            self.passed_windows += 1
            self.current_target_idx += 1

            stage_rewards = [40.0, 80.0, 150.0]
            reward += stage_rewards[min(self.passed_windows - 1, len(stage_rewards) - 1)]

            info["event"] = f"passed_window_{self.passed_windows}"
            info["passed_windows"] = self.passed_windows

            # Final success after third window
            if self.current_target_idx >= len(self.window_planes):
                self._done = True
                reward += self.success_reward
                info["event"] = "final_success"
                return self.s.copy(), float(reward), True, info

            return self.s.copy(), float(reward), False, info

        if crossed_target and (not inside_yz):
            reward -= 50.0
            self._done = True
            info["event"] = f"missed_window_{self.current_target_idx + 1}"
            info["passed_windows"] = self.passed_windows
            return self.s.copy(), float(reward), True, info

        if self.steps >= self.max_steps:
            self._done = True
            reward -= 10.0
            info["event"] = "max_steps"
            info["passed_windows"] = self.passed_windows
            return self.s.copy(), float(reward), True, info

        return self.s.copy(), float(reward), False, info