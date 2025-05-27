import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.signal_utils import compute_snr

class SignalFilterEnv(gym.Env):
    def __init__(self, signal, clean, filter_order=5, stride=64, imitation_weight=0.0):
        super().__init__()
        self.signal = signal
        self.clean = clean
        self.filter_order = filter_order
        self.stride = stride
        self.ptr = 0
        self.window_size = filter_order * 2
        self.max_steps = (len(signal) - self.window_size) // stride + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(filter_order,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.ptr = 0
        return self._get_obs(), {}

    def _get_obs(self):
        end = min(self.ptr + self.window_size, len(self.signal))
        obs = self.signal[self.ptr:end]
        if len(obs) < self.window_size:
            obs = np.pad(obs, (0, self.window_size - len(obs)), mode='constant')
        return obs.astype(np.float32)

    def step(self, coeffs):
        noisy_seg = self.signal[self.ptr:self.ptr + self.window_size]
        clean_seg = self.clean[self.ptr:self.ptr + self.window_size]

        # Apply filter
        filtered = np.convolve(noisy_seg, coeffs, mode="same")
        filtered = filtered[:len(clean_seg)]

        # Dense reward: negative MSE between filtered output and clean signal
        reward = -np.mean((clean_seg - filtered) ** 2)

        self.ptr += self.stride
        done = self.ptr + self.window_size >= len(self.signal)
        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass
