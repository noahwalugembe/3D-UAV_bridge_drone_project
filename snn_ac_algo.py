# snn_ac_algo.py
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import snntorch as snn
from snntorch import surrogate

from UAV_window_env import UAVWindowEnv
from state_normalization import StateNormalization

#MAX_EPISODE = 2000
MAX_EPISODE = 3000
GAMMA = 0.99
LR_A = 3e-4
LR_C = 1e-3

T_SNN = 20
BETA = 0.90
SPIKE_GRAD = surrogate.fast_sigmoid()

PPO_EPOCHS = 10
CLIP_EPS = 0.2
ENTROPY_BETA = 0.0005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0
SEED = 7

ROLLING_WINDOW = 20

# Reward shaping
INSIDE_CORRIDOR_REWARD = 0.10
TOWARD_TARGET_REWARD_K = 0.25
TOWARD_TARGET_CLIP = 0.40
YZ_ALIGN_REWARD_MAX = 0.30
STABILITY_PENALTY_K = 0.02


def set_seed(seed=7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SNNActor(nn.Module):
    def __init__(self, n_features: int, a_dim: int, action_bound: float):
        super().__init__()
        self.a_dim = a_dim
        self.action_bound = action_bound
        self.num_steps = T_SNN

        self.fc1 = nn.Linear(n_features, 256)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.fc2 = nn.Linear(256, 64)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.mu_head = nn.Linear(64, a_dim)
        self.mu_lif = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.sigma_head = nn.Linear(64, a_dim)
        self.sigma_lif = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.softplus = nn.Softplus()

        for layer in [self.fc1, self.fc2, self.mu_head, self.sigma_head]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.08)
            nn.init.constant_(layer.bias, 0.05)

    def forward(self, state: torch.Tensor):
        batch_size = state.size(0)
        device = state.device

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_mu = self.mu_lif.init_leaky()
        mem_sigma = self.sigma_lif.init_leaky()

        mem_mu_last = torch.zeros((batch_size, self.a_dim), device=device)
        mem_sigma_last = torch.zeros((batch_size, self.a_dim), device=device)

        for _ in range(self.num_steps):
            z1 = self.fc1(state)
            spk1, mem1 = self.lif1(z1, mem1)

            z2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(z2, mem2)

            mu_in = self.mu_head(spk2)
            _, mem_mu = self.mu_lif(mu_in, mem_mu)

            sigma_in = self.sigma_head(spk2)
            _, mem_sigma = self.sigma_lif(sigma_in, mem_sigma)

            mem_mu_last = mem_mu
            mem_sigma_last = mem_sigma

        mu = torch.tanh(mem_mu_last) * self.action_bound
        sigma = self.softplus(mem_sigma_last) + 0.03

        mu = torch.nan_to_num(mu, nan=0.0, posinf=self.action_bound, neginf=-self.action_bound)
        sigma = torch.nan_to_num(sigma, nan=0.1, posinf=1.0, neginf=0.1)
        sigma = torch.clamp(sigma, 0.03, 1.0)

        return mu, sigma

    def dist(self, state: torch.Tensor):
        mu, sigma = self.forward(state)
        return Normal(mu, sigma)


class SNNCritic(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.num_steps = T_SNN

        self.fc1 = nn.Linear(n_features, 256)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.fc2 = nn.Linear(256, 64)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        self.v_head = nn.Linear(64, 1)
        self.v_lif = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD, init_hidden=False)

        for layer in [self.fc1, self.fc2, self.v_head]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.08)
            nn.init.constant_(layer.bias, 0.05)

    def forward(self, state: torch.Tensor):
        batch_size = state.size(0)
        device = state.device

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_v = self.v_lif.init_leaky()

        mem_v_last = torch.zeros((batch_size, 1), device=device)

        for _ in range(self.num_steps):
            z1 = self.fc1(state)
            spk1, mem1 = self.lif1(z1, mem1)

            z2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(z2, mem2)

            v_in = self.v_head(spk2)
            _, mem_v = self.v_lif(v_in, mem_v)
            mem_v_last = mem_v

        return mem_v_last


def compute_returns(rewards, dones, gamma):
    R = 0.0
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0.0
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return np.array(returns, dtype=np.float32)


def curriculum_window_size(ep, base=20.0, start=60.0, warmup=800):
    if ep >= warmup:
        return base, base
    frac = ep / float(warmup)
    size = start - (start - base) * frac
    return float(size), float(size)


def moving_average(data, window):
    if len(data) < window:
        return np.array(data, dtype=np.float32)
    arr = np.array(data, dtype=np.float32)
    return np.convolve(arr, np.ones(window, dtype=np.float32), "valid") / window


def draw_rectangle(ax, x_plane, y0, z0, w, h, label, color):
    half_w = w / 2.0
    half_h = h / 2.0
    corners = np.array([
        [x_plane, y0 - half_w, z0 - half_h],
        [x_plane, y0 - half_w, z0 + half_h],
        [x_plane, y0 + half_w, z0 + half_h],
        [x_plane, y0 + half_w, z0 - half_h],
        [x_plane, y0 - half_w, z0 - half_h],
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            linestyle='--', linewidth=2, color=color, label=label)


def draw_box_planes_from_third(ax, box_x_list, y0, z0, w, h):
    for i in range(2, len(box_x_list)):
        x = box_x_list[i]
        draw_rectangle(ax, x, y0, z0, w, h, f"Box {i+1}", "deepskyblue")
        ax.text(x, y0, z0 + (h / 2.0) + 4, f"Box {i+1}",
                color="deepskyblue", fontsize=9, ha="center")


def save_plot_with_caption(fig, filepath, caption, dpi=150):
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_plot_report(filepath, run_title, summary_lines, plot_rows):
    with open(filepath, "w") as f:
        f.write(run_title + "\n")
        f.write("=" * len(run_title) + "\n\n")

        f.write("TRAINING SUMMARY\n")
        f.write("----------------\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("SAVED PLOTS (WITH CAPTIONS)\n")
        f.write("---------------------------\n")
        col1 = "File"
        col2 = "Caption / Description"
        f.write(f"{col1:<40} | {col2}\n")
        f.write("-" * 40 + "-+-" + "-" * 70 + "\n")
        for filename, desc in plot_rows:
            f.write(f"{filename:<40} | {desc}\n")


def _inside_corridor(env, s):
    x, y, z = float(s[0]), float(s[1]), float(s[2])
    half_w = env.window_w / 2.0
    half_h = env.window_h / 2.0

    in_y = (env.window_y - half_w) <= y <= (env.window_y + half_w)
    in_z = (env.window_z - half_h) <= z <= (env.window_z + half_h)

    if env.current_target_idx >= len(env.window_planes):
        target_x = env.window_planes[-1]
    else:
        target_x = env.window_planes[env.current_target_idx]

    in_x = abs(x - target_x) <= max(1.5, env.dt * env.V_MAX + 0.5)
    return in_x and in_y and in_z


def _progress_reward_toward_current_target(env, s_prev, s_next):
    idx = min(env.current_target_idx, len(env.window_planes) - 1)
    target_x = float(env.window_planes[idx])

    x_prev = float(s_prev[0])
    x_next = float(s_next[0])

    d_prev = abs(target_x - x_prev)
    d_next = abs(target_x - x_next)

    progress = d_prev - d_next
    shaped = TOWARD_TARGET_REWARD_K * progress
    return float(np.clip(shaped, -TOWARD_TARGET_CLIP, TOWARD_TARGET_CLIP))


def _yz_alignment_reward(env, s):
    y = float(s[1])
    z = float(s[2])

    half_w = max(env.window_w / 2.0, 1e-6)
    half_h = max(env.window_h / 2.0, 1e-6)

    dy = abs(y - env.window_y) / half_w
    dz = abs(z - env.window_z) / half_h

    dist = dy + dz
    reward = YZ_ALIGN_REWARD_MAX - 0.10 * dist
    return float(np.clip(reward, -YZ_ALIGN_REWARD_MAX, YZ_ALIGN_REWARD_MAX))


def _stability_penalty(s):
    vy = abs(float(s[4]))
    vz = abs(float(s[5]))
    return float(-STABILITY_PENALTY_K * (vy + vz))


def make_env():
    desired_box_x = [20.0, 40.0, 60.0, 80.0]
    desired_corridor_start_box_index = 2
    desired_corridor_length = 10.0
    desired_success_reward = 200.0
    

    try:
        return UAVWindowEnv(
            box_x=desired_box_x,
            corridor_start_box_index=desired_corridor_start_box_index,
            corridor_length=desired_corridor_length,
            success_reward=desired_success_reward,
            dt=0.5,
            v_max=3.0,
            action_bound=0.7,
            max_steps=220,
        )
    except TypeError:
        pass

    try:
        return UAVWindowEnv(
            desired_box_x,
            desired_corridor_start_box_index,
            desired_corridor_length,
            desired_success_reward,
        )
    except TypeError:
        pass

    env = UAVWindowEnv()

    if hasattr(env, "BOX_X"):
        try:
            env.BOX_X = desired_box_x
        except Exception:
            pass
    if hasattr(env, "box_x"):
        try:
            env.box_x = desired_box_x
        except Exception:
            pass
    if hasattr(env, "corridor_start_box_index"):
        try:
            env.corridor_start_box_index = desired_corridor_start_box_index
        except Exception:
            pass
    if hasattr(env, "corridor_length"):
        try:
            env.corridor_length = desired_corridor_length
        except Exception:
            pass
    if hasattr(env, "success_reward"):
        try:
            env.success_reward = desired_success_reward
        except Exception:
            pass
    if hasattr(env, "dt"):
        try:
            env.dt = 0.5
        except Exception:
            pass
    if hasattr(env, "V_MAX"):
        try:
            env.V_MAX = 3.0
        except Exception:
            pass
    if hasattr(env, "_action_bound"):
        try:
            env._action_bound = 0.7
            env.action_bound = np.array([-0.7, 0.7], dtype=np.float32)
        except Exception:
            pass
    if hasattr(env, "max_steps"):
        try:
            env.max_steps = 220
        except Exception:
            pass

    return env


def main():
    set_seed(SEED)

    env = make_env()
    normalizer = StateNormalization(
        v_max=env.V_MAX,
        width=env.WIDTH,
        height=env.HEIGHT,
        depth=env.DEPTH
    )

    N_S = env.state_dim
    a_dim = env.action_dim
    A_BOUND = env.action_bound[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = SNNActor(N_S, a_dim, A_BOUND).to(device)
    critic = SNNCritic(N_S).to(device)

    optA = optim.Adam(actor.parameters(), lr=LR_A)
    optC = optim.Adam(critic.parameters(), lr=LR_C)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"uav_results_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    ep_reward_list = []
    ep_success_list = []
    ep_steps_list = []
    ep_passed_windows = []
    log_lines = []

    plt.ion()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    t_start = time.time()

    for ep in range(MAX_EPISODE):
        w, h = curriculum_window_size(ep, base=20.0, start=60.0, warmup=800)
        env.set_window_size(w, h)

        s = env.reset()
        done = False
        success = False

        states, actions, logps, rewards, dones = [], [], [], [], []
        xs, ys, zs = [], [], []

        ax.cla()
        ax.set_xlim(0, env.WIDTH)
        ax.set_ylim(0, env.HEIGHT)
        ax.set_zlim(0, env.DEPTH)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Episode {ep+1}/{MAX_EPISODE}   corridor=({env.window_w:.1f},{env.window_h:.1f})"
        )

        draw_box_planes_from_third(ax, env.BOX_X, env.window_y, env.window_z, env.window_w, env.window_h)

        colors = ["orange", "red", "purple"]
        for i, wx in enumerate(env.window_planes):
            draw_rectangle(
                ax,
                wx,
                env.window_y,
                env.window_z,
                env.window_w,
                env.window_h,
                f"Target Window {i+1}",
                colors[i % len(colors)]
            )

        traj_line, = ax.plot([], [], [], linestyle='-', marker='o', markersize=4, label='UAV Path')
        uav_point, = ax.plot([], [], [], 'bo', markersize=6, label='UAV')
        ax.legend(loc="upper left")

        total_r = 0.0
        steps = 0

        while not done and steps < env.max_steps:
            target_idx = min(env.current_target_idx, len(env.window_planes) - 1)
            target_x = env.window_planes[target_idx]

            s_norm = normalizer.state_normal(
                s,
                target_x=target_x,
                window_y=env.window_y,
                window_z=env.window_z
            )
            st = torch.from_numpy(s_norm).float().unsqueeze(0).to(device)

            dist = actor.dist(st)
            raw_a = dist.rsample()
            a = torch.tanh(raw_a) * A_BOUND
            logp = dist.log_prob(raw_a).sum(dim=1).detach().cpu().numpy()[0]

            a_np = a.detach().cpu().numpy().reshape(a_dim,)
            s2, r, done, info = env.step(a_np)

            r += _progress_reward_toward_current_target(env, s, s2)
            r += _yz_alignment_reward(env, s2)
            r += _stability_penalty(s2)

            if _inside_corridor(env, s2):
                r += INSIDE_CORRIDOR_REWARD

            xs.append(s2[0])
            ys.append(s2[1])
            zs.append(s2[2])

            if steps % 3 == 0 or done:
                traj_line.set_data(xs, ys)
                traj_line.set_3d_properties(zs)
                uav_point.set_data([s2[0]], [s2[1]])
                uav_point.set_3d_properties([s2[2]])
                fig.canvas.draw_idle()
                plt.pause(0.001)

            if info.get("event") == "final_success":
                success = True

            states.append(s_norm)
            actions.append(a_np)
            logps.append(logp)
            rewards.append(float(r))
            dones.append(bool(done))

            s = s2
            total_r += r
            steps += 1

        ep_reward_list.append(float(total_r))
        ep_success_list.append(1 if success else 0)
        ep_steps_list.append(int(steps))
        ep_passed_windows.append(int(env.passed_windows))

        episode_log = (
            f"Episode {ep+1:4d}   Steps: {steps:3d}   Reward: {total_r:8.2f}   "
            f"Passed Windows: {env.passed_windows}   Success: {'Yes' if success else 'No'}   "
            f"Window Size: ({env.window_w:.1f}, {env.window_h:.1f})"
        )
        print(episode_log)
        log_lines.append(episode_log)

        if len(states) < 2:
            skip_msg = "  (skip update: trajectory too short)"
            print(skip_msg)
            log_lines.append(skip_msg)
            plt.pause(0.001)
            continue

        returns = compute_returns(rewards, dones, GAMMA)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_logps_t = torch.tensor(np.array(logps), dtype=torch.float32, device=device).unsqueeze(1)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            values = critic(states_t)
            adv = returns_t - values

            adv_std = adv.std(unbiased=False)
            if torch.isnan(adv_std) or adv_std < 1e-6:
                adv = adv - adv.mean()
            else:
                adv = (adv - adv.mean()) / (adv_std + 1e-8)

        for _ in range(PPO_EPOCHS):
            dist = actor.dist(states_t)

            if torch.isnan(dist.loc).any() or torch.isnan(dist.scale).any():
                nan_msg = "  (skip update: NaN in actor outputs)"
                print(nan_msg)
                log_lines.append(nan_msg)
                break

            clipped_actions = torch.clamp(actions_t / A_BOUND, -0.999999, 0.999999)
            raw_actions = 0.5 * torch.log((1.0 + clipped_actions) / (1.0 - clipped_actions))

            new_logps = dist.log_prob(raw_actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(new_logps - old_logps_t)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
            actor_loss = -(torch.min(surr1, surr2)).mean()

            entropy = dist.entropy().sum(dim=1, keepdim=True).mean()
            value_pred = critic(states_t)
            value_loss = (returns_t - value_pred).pow(2).mean()

            loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_BETA * entropy

            optA.zero_grad()
            optC.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            optA.step()
            optC.step()

        plt.pause(0.001)

    elapsed = time.time() - t_start
    plt.ioff()
    plt.close(fig)

    total_episodes = len(ep_reward_list)
    total_successes = int(np.sum(ep_success_list))
    success_rate = (total_successes / total_episodes) * 100.0 if total_episodes > 0 else 0.0
    avg_reward = float(np.mean(ep_reward_list)) if total_episodes > 0 else 0.0
    avg_steps = float(np.mean(ep_steps_list)) if total_episodes > 0 else 0.0
    avg_windows = float(np.mean(ep_passed_windows)) if total_episodes > 0 else 0.0

    results_txt = os.path.join(out_dir, "results.txt")
    summary_lines = [
        f"Date/Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total episodes      : {total_episodes}",
        f"Total full successes: {total_successes}",
        f"Success rate        : {success_rate:.2f}%",
        f"Average reward      : {avg_reward:.2f}",
        f"Average steps/ep    : {avg_steps:.2f}",
        f"Average windows/ep  : {avg_windows:.2f}",
        f"Training time (s)   : {elapsed:.2f}",
        f"Output folder       : {out_dir}",
        "",
        "Reward shaping enabled:",
        f" - toward target: k={TOWARD_TARGET_REWARD_K}, clip={TOWARD_TARGET_CLIP}",
        f" - yz alignment max reward: {YZ_ALIGN_REWARD_MAX}",
        f" - lateral stability penalty k: {STABILITY_PENALTY_K}",
        f" - inside corridor step reward: {INSIDE_CORRIDOR_REWARD}",
        "",
        "Environment settings:",
        f" - dt: {env.dt}",
        f" - v_max: {env.V_MAX}",
        f" - action_bound: {env._action_bound}",
        f" - max_steps: {env.max_steps}",
        f" - target planes: {env.window_planes}",
    ]

    plot_rows = []

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(ep_reward_list)
    if total_episodes >= ROLLING_WINDOW:
        ma_reward = moving_average(ep_reward_list, ROLLING_WINDOW)
        plt.plot(range(ROLLING_WINDOW - 1, total_episodes), ma_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards over Training")
    plt.grid(True, alpha=0.3)

    rewards_png = os.path.join(out_dir, f"plot_rewards_{timestamp}.png")
    cap1 = (
        "Caption: Episode Reward Curve. "
        "Each point is the sum of rewards in one episode. "
        f"A rolling mean (window={ROLLING_WINDOW}) is overlaid when enough episodes exist."
    )
    save_plot_with_caption(fig1, rewards_png, cap1)
    plot_rows.append((os.path.basename(rewards_png), cap1))

    fig2 = plt.figure(figsize=(10, 5))
    plt.scatter(range(total_episodes), ep_success_list, s=12, alpha=0.7)
    if total_episodes >= ROLLING_WINDOW:
        ma_success = moving_average(ep_success_list, ROLLING_WINDOW) * 100.0
        plt.plot(range(ROLLING_WINDOW - 1, total_episodes), ma_success)
    plt.xlabel("Episode")
    plt.ylabel("Full Success (0/1) and Rolling %")
    plt.title("Final Success per Episode and Rolling Success Rate")
    plt.grid(True, alpha=0.3)

    success_png = os.path.join(out_dir, f"plot_success_{timestamp}.png")
    cap2 = (
        "Caption: Full Success Trace and Rolling Success Rate. "
        "Dots show final episode success (1) or failure (0). "
        f"The rolling curve is the moving average success rate (%) over {ROLLING_WINDOW} episodes."
    )
    save_plot_with_caption(fig2, success_png, cap2)
    plot_rows.append((os.path.basename(success_png), cap2))

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(ep_passed_windows)
    if total_episodes >= ROLLING_WINDOW:
        ma_windows = moving_average(ep_passed_windows, ROLLING_WINDOW)
        plt.plot(range(ROLLING_WINDOW - 1, total_episodes), ma_windows)
    plt.xlabel("Episode")
    plt.ylabel("Passed Windows")
    plt.title("Number of Windows Passed per Episode")
    plt.grid(True, alpha=0.3)

    windows_png = os.path.join(out_dir, f"plot_windows_{timestamp}.png")
    cap3 = (
        "Caption: Passed Windows per Episode. "
        "This shows how many target windows were crossed successfully in each episode, "
        "with rolling mean when enough episodes exist."
    )
    save_plot_with_caption(fig3, windows_png, cap3)
    plot_rows.append((os.path.basename(windows_png), cap3))

    plots_report_txt = os.path.join(out_dir, f"plots_report_{timestamp}.txt")
    run_title = "UAV Multi-Window Task - Plot Captions and Descriptions"
    write_plot_report(plots_report_txt, run_title, summary_lines, plot_rows)

    with open(results_txt, "w") as f:
        f.write("UAV Multi-Window Task - Full Results Log\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output folder: {out_dir}\n\n")

        f.write("EPISODE LOGS\n")
        f.write("------------\n")
        for line in log_lines:
            f.write(line + "\n")

        f.write("\nTRAINING SUMMARY\n")
        f.write("----------------\n")
        for line in summary_lines:
            f.write(line + "\n")

        f.write("\nSAVED PLOTS\n")
        f.write("-----------\n")
        for filename, desc in plot_rows:
            f.write(f"{filename}: {desc}\n")

        f.write("\nPLOT REPORT FILE\n")
        f.write("----------------\n")
        f.write(f"{plots_report_txt}\n")

    print("\nTraining completed.")
    print(f"Saved results folder: {out_dir}")
    print(f"Results log:          {results_txt}")
    print(f"Plot captions report: {plots_report_txt}")
    print("Saved plots:")
    print(f" - {rewards_png}")
    print(f" - {success_png}")
    print(f" - {windows_png}")


if __name__ == "__main__":
    main()