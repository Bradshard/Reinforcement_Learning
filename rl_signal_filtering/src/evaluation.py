import argparse
import yaml
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from agents.ppo_agent     import PPOAgent
from utils.filter_tuner import FilterTuner
from utils.stress_tester import StressTester
from utils.signal_utils import generate_synthetic_signal, compute_snr, lms_filter, rls_filter, sinusoidal_kalman_filter, wiener_filter

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PPO‐LSTM vs. classical filters across noise types"
    )
    p.add_argument("--config",       type=str,
                   help="Path to YAML config file")
    p.add_argument("--signal_length",type=int,   default=2048)
    p.add_argument("--filter_order", type=int,   default=5)
    p.add_argument("--window_size",  type=int,   help="Defaults to 2*filter_order")
    p.add_argument("--stride",       type=int,   help="Defaults to window_size//2")
    p.add_argument("--model_path",   type=str,   default="ppo_model.pt")
    p.add_argument("--noise_types",  nargs="+",
                   default=["gaussian","uniform","laplacian",
                            "impulse","pink","brown"])
    p.add_argument("--sigmas",       nargs="+", type=float,
                   default=[0.1, 0.5, 1.0, 2.0])
    return p.parse_args()

def main():
    args = parse_args()
    # loading YAML overrides
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k,v in cfg.items():
            setattr(args, k, v)

    # running defaults
    W = args.window_size or (2 * args.filter_order)
    S = args.stride      or (W // 2)

    # loading PPO‐LSTM agent
    obs_dim = 2 * W
    act_dim = W
    agent   = PPOAgent(obs_dim, act_dim)
    agent.policy.load_state_dict(torch.load(args.model_path))
    agent.policy.eval()

    # building the stress tester
    tester = StressTester(
        agent,
        filter_order  = args.filter_order,
        signal_length = args.signal_length,
        window_size   = W,
        stride        = S
    )

    # runAll
    results = tester.run(args.noise_types, args.sigmas)

    # Summary Table printing
    header = (
        "Noise","σ",
        "LMS_SNR","RLS_SNR","WIENER_SNR","KALMAN_SNR",
        "PPO_SNR","PPO_t(s)"
    )
    print(" ".join(f"{h:>10s}" for h in header))
    for r in results:
        bf = r.best_filters
        print(f"{r.noise:>10s}"
              f"{r.sigma:10.2f}"
              f"{bf['lms']['snr']:10.2f}"
              f"{bf['rls']['snr']:10.2f}"
              f"{bf['wiener']['snr']:10.2f}"
              f"{bf['kalman']['snr']:10.2f}"
              f"{r.ppo_snr:10.2f}"
              f"{r.ppo_time:10.4f}"
        )

    # 1) line plots of SNR vs σ for each noise type
    methods = ['lms','rls','wiener','kalman','ppo']
    for noise in args.noise_types:
        plt.figure(figsize=(6,4))
        for m in methods:
            if m == 'ppo':
                ys = [r.ppo_snr for r in results if r.noise==noise]
            else:
                ys = [r.best_filters[m]['snr'] for r in results if r.noise==noise]
            xs = [r.sigma for r in results if r.noise==noise]
            order = np.argsort(xs)
            plt.plot(np.array(xs)[order], np.array(ys)[order],
                     marker='o', label=m.upper())
        plt.title(f"SNR vs σ — {noise.capitalize()}")
        plt.xlabel("Noise σ")
        plt.ylabel("Output SNR (dB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # 2) bar chart of average inference time per method
    import time, collections
    method_times = collections.defaultdict(list)
    for r in results:
        # regen signal
        clean, noisy = generate_synthetic_signal(
            length=args.signal_length,
            noise_type=r.noise,
            noise_params={
                'std':   r.sigma,
                'range': r.sigma,
                'scale': r.sigma,
                'prob':  min(r.sigma,1.0),
                'amp':   5.0
            }
        )
        # classical filters
        bf = r.best_filters
        t0 = time.perf_counter()
        _ = lms_filter(noisy, clean,
                       mu=bf['lms']['params']['mu'],
                       filter_order=bf['lms']['params']['order'])
        method_times['LMS'].append(time.perf_counter()-t0)

        t0 = time.perf_counter()
        _ = rls_filter(noisy, clean,
                       delta=bf['rls']['params']['delta'],
                       lam=bf['rls']['params']['lam'],
                       filter_order=args.filter_order)
        method_times['RLS'].append(time.perf_counter()-t0)

        t0 = time.perf_counter()
        _ = wiener_filter(noisy, clean,
                          filter_order=bf['wiener']['params']['order'])
        method_times['Wiener'].append(time.perf_counter()-t0)

        t0 = time.perf_counter()
        _ = sinusoidal_kalman_filter(noisy, freqs=(5,10),
                                     dt=1.0,
                                     q=bf['kalman']['params']['q'],
                                     r=bf['kalman']['params']['r'])
        method_times['Kalman'].append(time.perf_counter()-t0)

        method_times['PPO'].append(r.ppo_time)

    avg_times = {m: np.mean(ts) for m,ts in method_times.items()}
    plt.figure(figsize=(6,4))
    plt.bar(avg_times.keys(), avg_times.values())
    plt.title("Average Inference Time per Method")
    plt.ylabel("Time (s)")
    plt.grid(axis='y')
    plt.tight_layout()

    plt.show()

if __name__=="__main__":
    main()
