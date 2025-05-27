import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.filter_tuner import FilterTuner
from envs.signal_env import SignalFilterEnv
from agents.ppo_agent import PPOAgent
from utils.signal_utils import (
    generate_synthetic_signal,
    compute_snr
)

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO‐LSTM residual denoiser")
    p.add_argument("--config",        type=str,   help="YAML config file")
    p.add_argument("--signal_length", type=int,   default=2048)
    p.add_argument("--filter_order",  type=int,   default=5)
    p.add_argument("--window_size",   type=int,   help="defaults to 2*filter_order")
    p.add_argument("--stride",        type=int,   help="defaults to window_size//2")
    p.add_argument("--num_episodes",  type=int,   default=1000)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--update_epochs", type=int,   default=5)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--alpha",         type=float, default=0.5,
                   help="MSE weight in reward")
    p.add_argument("--beta",          type=float, default=0.1,
                   help="Smoothness weight")

    # 1) First parse whatever was passed on the CLI
    args = p.parse_args()

    # 2) Build a mapping from argument name → its `type` callable
    type_map = { action.dest: action.type
                 for action in p._actions
                 if action.type is not None }

    # 3) If user provided a --config, read it and override
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k in type_map:
                # cast v to the same type as the argparse expects
                setattr(args, k, type_map[k](v))
            else:
                # maybe it's window_size or stride that weren't in map:
                setattr(args, k, v)

    return args

    
def make_obs(noisy_win, aug_win):
    """Pad to W, then z-score normalize each window and concatenate."""
    W = noisy_win.shape[0]
    if len(noisy_win) < W:
        pad = W - len(noisy_win)
        noisy_win = np.pad(noisy_win, (0, pad))
        aug_win   = np.pad(aug_win,   (0, pad))
    μ = noisy_win.mean()
    σ = noisy_win.std() + 1e-8
    z_n = (noisy_win - μ) / σ
    z_a = (aug_win   - μ) / σ
    return np.concatenate([z_n, z_a]), μ, σ

def main():
    args = parse_args()
    reward_history = []

    # Derived defaults
    W = args.window_size or (2 * args.filter_order)
    S = args.stride      or (W // 2)

    # 1) Generate & normalize the full signal
    clean_signal, noisy_signal = generate_synthetic_signal(length=args.signal_length)
    gm, gs = noisy_signal.mean(), noisy_signal.std()
    clean_signal = (clean_signal - gm) / (gs + 1e-8)
    noisy_signal = (noisy_signal - gm) / (gs + 1e-8)

    # 2) Grid-search classical baselines via FilterTuner
    tuner = FilterTuner(filter_order=args.filter_order)
    tuned = tuner.tune_all(noisy_signal, clean_signal)
    # tuned: {'lms': (out, snr, params), 'rls':..., 'wiener':..., 'kalman':...}
    best_method, (aug_signal, best_snr, best_params) = max(
        tuned.items(), key=lambda kv: kv[1][1]
    )
    print(f"Using baseline = {best_method} (SNR={best_snr:.2f} dB), params={best_params}")

    # 3) Setup environment & PPO‐LSTM agent
    env   = SignalFilterEnv(noisy_signal, clean_signal,
                           filter_order=args.filter_order,
                           stride=S)
    obs_dim = 2 * W
    act_dim = W         # residual length
    agent   = PPOAgent(obs_dim, act_dim,
                       lr=args.lr,
                       gamma=args.gamma,
                       clip_eps=args.clip_eps,
                       update_epochs=args.update_epochs)

    # 4) Training loop
    for ep in range(1, args.num_episodes+1):
        ptr = 0
        done = False
        last_res = np.zeros(W, dtype=np.float32)

        # initial obs
        noisy_win = noisy_signal[ptr:ptr+W]
        aug_win   = aug_signal [ptr:ptr+W]
        obs, μ, σ = make_obs(noisy_win, aug_win)

        # buffers
        obs_buf, act_buf = [], []
        logp_buf, val_buf= [], []
        rew_buf, done_buf= [], []

        while not done:
            # get action from PPO (normalized residual)
            res_norm, logp, val = agent.get_action(obs)
            residual = res_norm * σ

            # apply residual on baseline window
            base = aug_signal[ptr:ptr+W]
            if len(base) < W:
                base = np.pad(base, (0, W-len(base)))
            denoised = base + residual

            # compute reward
            clean_win = clean_signal[ptr:ptr+W]
            noisy_win = noisy_signal[ptr:ptr+W]
            mse      = np.mean((clean_win - denoised)**2)
            snr_gain = compute_snr(clean_win, denoised) - compute_snr(clean_win, noisy_win)
            smooth   = np.mean((residual - last_res)**2)
            reward   = snr_gain - args.alpha*mse - args.beta*smooth

            # store
            obs_buf .append(obs)
            act_buf .append(res_norm)
            logp_buf.append(logp.detach())
            val_buf .append(val.detach())
            rew_buf .append(torch.tensor(reward, dtype=torch.float32))
            done_buf.append(torch.tensor(done, dtype=torch.float32))

            last_res = residual.copy()
            ptr += S
            done = ptr + W >= len(noisy_signal)
            if not done:
                noisy_win = noisy_signal[ptr:ptr+W]
                aug_win   = aug_signal [ptr:ptr+W]
                obs, μ, σ = make_obs(noisy_win, aug_win)

        # bootstrap last value
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            _, next_val = agent.policy(t_obs)

        # compute returns & advantages
        returns    = agent.compute_returns(rew_buf, val_buf, next_val.squeeze(0), done_buf)
        advantages = returns - torch.stack(val_buf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        agent.update(
            torch.tensor(np.array(obs_buf), dtype=torch.float32),
            torch.tensor(np.array(act_buf), dtype=torch.float32),
            torch.stack(logp_buf),
            returns,
            torch.stack(val_buf)
        )

        # log average per-step reward
        avg_r = sum(rew_buf).item() / len(rew_buf)
        reward_history.append(avg_r)
        if ep % 10 == 0:
            print(f"Episode {ep}/{args.num_episodes}  Avg reward/step: {avg_r:.2f}")

    # 5)Just before save
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(reward_history)+1), reward_history, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Step')
    plt.title('Training Progress: Avg Reward per Step vs Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

    # 6) Save final model
    model_name = f"ppo_model.pt"
    torch.save(agent.policy.state_dict(), model_name)
    print(f"Saved model to {model_name}")

if __name__ == "__main__":
    main()
