import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_signal(length=1024, freqs=(5, 10), noise_std=0.5):
    t = np.linspace(0, 1, length)
    clean = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    noise = np.random.normal(0, noise_std, length)
    return clean, clean + noise

def apply_fir_filter(signal, coeffs, meta_param):
    filtered = np.convolve(signal, coeffs, mode='same') * (1 + meta_param)
    return filtered

def compute_snr(clean, test):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - test) ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def plot_signals(clean, noisy, filtered, title='Signal Comparison'):
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label='Clean Signal', linewidth=2)
    plt.plot(noisy, label='Noisy Signal', alpha=0.6)
    plt.plot(filtered, label='Filtered Signal', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def lms_filter(noisy, desired, mu=0.01, filter_order=5):
    n = len(noisy)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)

    for i in range(filter_order, n):
        x = noisy[i - filter_order:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w += 2 * mu * e[i] * x

    return y

def rls_filter(noisy, desired, delta=1.0, lam=0.99, filter_order=5):
    n = len(noisy)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)
    P = np.eye(filter_order) / delta

    for i in range(filter_order, n):
        x = noisy[i - filter_order:i][::-1]
        x = x.reshape(-1, 1)
        y[i] = np.dot(w, x.flatten())
        e[i] = desired[i] - y[i]
        k = (P @ x) / (lam + x.T @ P @ x)
        w += (k.flatten() * e[i])
        P = (P - k @ x.T @ P) / lam

    return y

def kalman_filter(noisy, q=1e-5, r=0.5):
    n = len(noisy)
    x_est = np.zeros(n)
    p = 1.0  # initial error covariance
    x = 0.0  # initial estimate

    for i in range(n):
        # Prediction update
        p += q

        # Measurement update
        k = p / (p + r)
        x = x + k * (noisy[i] - x)
        p = (1 - k) * p

        x_est[i] = x

    return x_est
