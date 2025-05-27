import numpy as np
import time
from typing import List, Tuple, Dict, Any
from collections import namedtuple

from utils.signal_utils import compute_snr, generate_synthetic_signal
from utils.filter_tuner import FilterTuner
from agents.ppo_agent import PPOAgent

StressResult = namedtuple(
    'StressResult',
    ['noise', 'sigma',
     'best_filters',  # Dict[str, Dict[str,Any]] with snr & params
     'ppo_snr', 'ppo_time']
)

class StressTester:
    def __init__(self,
                 agent: PPOAgent,
                 filter_order: int,
                 signal_length: int = 2048,
                 window_size: int   = None,
                 stride: int        = None):
        self.agent        = agent
        self.tuner        = FilterTuner(filter_order)
        self.signal_length= signal_length
        self.window_size  = window_size or filter_order*2
        self.stride       = stride or self.window_size//2

    def _build_obs_batch(self,
                         noisy: np.ndarray,
                         aug:   np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, mus, sigmas = [], [], []
        ptr = 0
        W   = self.window_size
        S   = self.stride
        while ptr < len(noisy):
            win_n = noisy[ptr:ptr+W]
            win_a = aug  [ptr:ptr+W]
            if len(win_n) < W:
                pad = W - len(win_n)
                win_n = np.pad(win_n, (0, pad))
                win_a = np.pad(win_a, (0, pad))
            μ = win_n.mean()
            σ = win_n.std() + 1e-8
            obs.append(np.concatenate([(win_n-μ)/σ, (win_a-μ)/σ]))
            mus.append(μ); sigmas.append(σ)
            ptr += S
        return np.stack(obs,axis=0), np.array(mus), np.array(sigmas)

    def ppo_denoise(self,
                    noisy: np.ndarray,
                    aug:   np.ndarray
                   ) -> Tuple[np.ndarray, float]:
        obs_batch, mus, sigmas = self._build_obs_batch(noisy, aug)
        import torch
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).unsqueeze(1)
        t0 = time.perf_counter()
        with torch.no_grad():
            dist, _ = self.agent.policy(obs_tensor)
            res = dist.mean.squeeze(1).cpu().numpy()
        ppo_time = time.perf_counter() - t0

        # overlap-add
        denoised = np.zeros_like(noisy)
        weights  = np.zeros_like(noisy)
        ptr = 0
        for b in range(res.shape[0]):
            residual = res[b] * sigmas[b]
            base     = aug[ptr:ptr+self.window_size]
            if len(base)<self.window_size:
                base = np.pad(base,(0,self.window_size-len(base)))
            seg = base + residual
            end = min(ptr+self.window_size, len(denoised))
            denoised[ptr:end] += seg[:end-ptr]
            weights[ptr:end]+=1
            ptr += self.stride
        weights[weights==0]=1
        denoised /= weights
        return denoised, ppo_time

    def run(self,
            noise_types: List[str],
            sigmas:      List[float]
           ) -> List[StressResult]:

        results = []
        for noise in noise_types:
            for σ in sigmas:
                # generate test signal
                clean, noisy = generate_synthetic_signal(
                    length=self.signal_length,
                    noise_type=noise,
                    noise_params={'std':σ,'range':σ,'scale':σ,'prob':σ,'amp':5.0}
                )

                # tune all filters
                tuned = self.tuner.tune_all(noisy, clean)
                best_filters = {}
                for name, (out, snr, params) in tuned.items():
                    best_filters[name] = {'snr': snr, 'params': params}

                # run PPO residual
                # pick one baseline to augment (e.g. lms), or pass tuned[name][0]
                aug_signal = tuned['lms'][0]
                denoised, ppo_t = self.ppo_denoise(noisy, aug_signal)
                ppo_snr = compute_snr(clean, denoised)

                results.append(StressResult(noise, σ, best_filters,
                                            ppo_snr, ppo_t))
        return results

