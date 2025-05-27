import numpy as np

# 1) Synthetic signal generators
def time_function(func, *args, **kwargs):
    """
    Run func(*args, **kwargs), return (output, elapsed_time_sec).
    """
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0

def generate_dataset(trials=100, length=1024, freqs=(5,10),
                     noise_type='gaussian', noise_params=None):
    """
    Generate a list of (clean, noisy) pairs, delegating to generate_synthetic_signal.
    """
    dataset = []
    for _ in range(trials):
        clean, noisy = generate_synthetic_signal(length, freqs,
                                                 noise_type, noise_params)
        dataset.append((clean, noisy))
    return dataset


def generate_synthetic_signal(length=1024,
                              freqs=(5, 10),
                              noise_type='gaussian',
                              noise_params=None):
    """
    Returns a single (clean, noisy) signal pair with specified noise.
    Supported noise_types: 'gaussian', 'uniform', 'laplacian',
    'impulse', 'pink', 'brown'.
    """
    t = np.linspace(0, 1, length)
    clean = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    params = noise_params or {}

    if noise_type == 'gaussian':
        std = params.get('std', 0.5)
        noise = np.random.normal(0, std, length)

    elif noise_type == 'uniform':
        r = params.get('range', 1.0)
        noise = np.random.uniform(-r, r, length)

    elif noise_type == 'laplacian':
        # new case
        scale = params.get('scale', 0.5)
        noise = np.random.laplace(0, scale, length)

    elif noise_type == 'impulse':
        prob = params.get('prob', 0.01)
        amp  = params.get('amp', 5.0)
        noise = np.zeros(length)
        mask  = np.random.rand(length) < prob
        noise[mask] = amp * (2 * np.random.rand(mask.sum()) - 1)

    elif noise_type in ('pink', 'brown'):
        std = params.get('std', 0.5)
        white = np.random.normal(0, 1, length)
        f = np.fft.rfft(white)
        freqs_fft = np.fft.rfftfreq(length, d=1/length)
        if noise_type == 'pink':
            f /= np.sqrt(freqs_fft + 1e-6)
        else:
            f /= (freqs_fft + 1e-6)
        colored = np.fft.irfft(f, n=length)
        noise = (colored / (np.std(colored) + 1e-6)) * std

    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    return clean, clean + noise

# 2) FIR helper

def apply_fir_filter(signal, coeffs, meta_param):
    return np.convolve(signal, coeffs, mode='same') * (1 + meta_param)

# 3) Robust SNR computation

def compute_snr(clean, test):
    clean = np.asarray(clean, dtype=np.float64)
    test  = np.asarray(test,  dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        sp   = np.mean(clean**2, dtype=np.float64)
        npow = np.mean((clean - test)**2, dtype=np.float64)
        if npow <= 0 or not np.isfinite(npow):
            return np.inf
        return 10.0 * np.log10(sp / npow)

# 4) Plotting utility

def plot_signals(clean, noisy, filtered, title='Signal Comparison'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label='Clean Signal')
    plt.plot(noisy, label='Noisy Signal', alpha=0.6)
    plt.plot(filtered, label='Filtered Signal', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 5) Classical filters

def lms_filter(noisy, desired, mu=0.01, filter_order=5, w_max=1e3):
    n = len(noisy)
    y = np.zeros(n)
    w = np.zeros(filter_order)
    for i in range(filter_order, n):
        x = noisy[i-filter_order:i][::-1]
        y[i] = w.dot(x)
        e = desired[i] - y[i]
        upd = 2 * mu * e * x
        upd = np.nan_to_num(upd, nan=0.0, posinf=w_max, neginf=-w_max)
        w += upd
        w = np.clip(w, -w_max, w_max)
    return y


def rls_filter(noisy, desired, delta=1.0, lam=0.99, filter_order=5, p_max=1e3):
    n = len(noisy)
    y = np.zeros(n)
    w = np.zeros(filter_order)
    P = np.eye(filter_order) / delta
    for i in range(filter_order, n):
        x = noisy[i-filter_order:i][::-1].reshape(-1,1)
        y[i] = float(w.dot(x.flatten()))
        e = desired[i] - y[i]
        denom = lam + float(x.T.dot(P).dot(x))
        k = (P.dot(x)) / denom
        k = np.nan_to_num(k, nan=0.0, posinf=p_max, neginf=-p_max)
        upd = k.flatten() * e
        upd = np.nan_to_num(upd, nan=0.0, posinf=p_max, neginf=-p_max)
        w += upd
        w = np.clip(w, -p_max, p_max)
        P = (P - k.dot(x.T).dot(P)) / lam
        P = np.nan_to_num(P, nan=0.0, posinf=p_max, neginf=-p_max)
    return y


def kalman_filter(noisy, q=1e-5, r=0.5):
    n = len(noisy)
    x = 0.0
    p = 1.0
    x_est = np.zeros(n)
    for i in range(n):
        p += q
        k = p / (p + r)
        x = x + k * (noisy[i] - x)
        p = (1 - k) * p
        x_est[i] = x
    return x_est

# 6) New baselines

def wiener_filter(noisy, desired, filter_order=5):
    """
    Batch Wiener FIR filter solution (no scipy dependency).
    """
    n = len(noisy)
    # autocorrelation
    r_full = np.correlate(noisy, noisy, mode='full')
    r = r_full[n-1:n-1+filter_order]
    # Toeplitz
    R = np.empty((filter_order, filter_order), dtype=np.float64)
    for i in range(filter_order):
        for j in range(filter_order):
            R[i, j] = r[abs(i-j)]
    # cross-correlation
    p_full = np.correlate(desired, noisy, mode='full')
    p = p_full[n-1:n-1+filter_order]
    w = np.linalg.solve(R + 1e-6*np.eye(filter_order), p)
    return np.convolve(noisy, w, mode='same')


def sinusoidal_kalman_filter(noisy, freqs=(5,10), dt=1.0, q=1e-5, r=0.5):
    """
    Vector Kalman filter for sum-of-sinusoids state model.
    State dimension = 2*num_freqs (cos,sin per tone).
    """
    n = len(noisy)
    m = len(freqs)
    dim = 2*m
    A = np.zeros((dim, dim))
    H = np.zeros((1, dim))
    for i, f in enumerate(freqs):
        c = np.cos(2*np.pi*f*dt)
        s = np.sin(2*np.pi*f*dt)
        A[2*i:2*i+2,2*i:2*i+2] = [[c, -s],[s, c]]
        H[0,2*i] = 1.0
    x = np.zeros(dim)
    P = np.eye(dim)
    y = np.zeros(n)
    for k in range(n):
        x = A.dot(x)
        P = A.dot(P).dot(A.T) + q*np.eye(dim)
        z = noisy[k]
        S = H.dot(P).dot(H.T) + r
        K = P.dot(H.T) / S
        x = x + (K.flatten() * (z - H.dot(x)))
        P = (np.eye(dim) - K.dot(H)).dot(P)
        y[k] = H.dot(x)
    return y
