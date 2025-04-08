import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SignalFilterEnv(gym.Env):
    def __init__(self, signal_length=1024, filter_order=5):
        super(SignalFilterEnv, self).__init__()

        self.signal_length = signal_length
        self.filter_order = filter_order

        # Observation space: past N noisy samples + filter state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(signal_length,), dtype=np.float32
        )

        # Action space: filter coefficients + meta parameter (learning rate)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(filter_order + 1,), dtype=np.float32
        )

        self.reset()

    def generate_signal(self):
        t = np.linspace(0, 1, self.signal_length)
        clean = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
        noise = np.random.normal(0, 0.5, self.signal_length)
        return clean, clean + noise

    def step(self, action):
        coeffs = action[:self.filter_order]
        meta_param = action[-1]
        filtered = np.convolve(self.noisy_signal, coeffs, mode='same') * (1 + meta_param)
        reward = -np.mean((filtered - self.clean_signal) ** 2)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return self.noisy_signal.astype(np.float32), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.clean_signal, self.noisy_signal = self.generate_signal()
        self.current_step = 0
        self.max_steps = 1
        info = {}
        return self.noisy_signal.astype(np.float32), info

    def render(self, mode='human'):
        pass

