import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.ppo_agent import PPOAgent
from utils.signal_utils import (
    generate_synthetic_signal,
    apply_fir_filter,
    compute_snr,
    plot_signals,
    lms_filter,
    rls_filter,
    kalman_filter
)

# Settings
SIGNAL_LENGTH = 1024
FILTER_ORDER = 5
OBS_DIM = SIGNAL_LENGTH
ACT_DIM = FILTER_ORDER + 1
MODEL_PATH = "ppo_filter.pt"

# Load PPO Agent
agent = PPOAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM)

try:
    agent.policy.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"No saved model found at {MODEL_PATH}, using random weights")

# Generate synthetic data
clean, noisy = generate_synthetic_signal(length=SIGNAL_LENGTH)
obs = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)

# PPO filter
agent.policy.eval()
with torch.no_grad():
    dist, _ = agent.policy(obs)
    action = dist.mean.squeeze(0).numpy()

coeffs = action[:FILTER_ORDER]
meta_param = action[-1]
ppo_filtered = apply_fir_filter(noisy, coeffs, meta_param)

# Classical filters
lms_filtered = lms_filter(noisy, clean, mu=0.01, filter_order=FILTER_ORDER)
rls_filtered = rls_filter(noisy, clean, delta=1.0, lam=0.99, filter_order=FILTER_ORDER)
kalman_filtered = kalman_filter(noisy, q=1e-5, r=0.5)

# Compute SNRs and MSEs
snrs = {
    "Noisy": compute_snr(clean, noisy),
    "PPO": compute_snr(clean, ppo_filtered),
    "LMS": compute_snr(clean, lms_filtered),
    "RLS": compute_snr(clean, rls_filtered),
    "Kalman": compute_snr(clean, kalman_filtered)
}

mses = {
    "PPO": np.mean((clean - ppo_filtered) ** 2),
    "LMS": np.mean((clean - lms_filtered) ** 2),
    "RLS": np.mean((clean - rls_filtered) ** 2),
    "Kalman": np.mean((clean - kalman_filtered) ** 2)
}

# Print metrics
for name, snr in snrs.items():
    print(f"{name} SNR: {snr:.2f} dB")
print()
for name, mse in mses.items():
    print(f"{name} MSE: {mse:.6f}")

# Plot all results
plot_signals(clean, noisy, ppo_filtered, title="PPO Filtering")
plot_signals(clean, noisy, lms_filtered, title="LMS Filtering")
plot_signals(clean, noisy, rls_filtered, title="RLS Filtering")
plot_signals(clean, noisy, kalman_filtered, title="Kalman Filtering")

# Compare SNR/MSE in a bar plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(snrs.keys(), snrs.values(), color='skyblue')
plt.title("SNR (dB)")
plt.ylabel("SNR")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(mses.keys(), mses.values(), color='salmon')
plt.title("MSE")
plt.ylabel("Mean Squared Error")
plt.grid(True)

plt.tight_layout()
plt.show()
