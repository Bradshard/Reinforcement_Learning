import numpy as np
from typing import Dict, Any, Tuple
from utils.signal_utils import (
    lms_filter,
    rls_filter,
    wiener_filter,
    sinusoidal_kalman_filter,
    compute_snr,
)

FilterResult = Tuple[np.ndarray, float, Dict[str, Any]]
# (output_signal, best_snr, best_params)

class FilterTuner:
    def __init__(self, filter_order: int):
        self.filter_order = filter_order

    def tune_lms(self, noisy: np.ndarray, clean: np.ndarray) -> FilterResult:
        best_snr = -np.inf
        best_out = None
        best_params = {}
        for mu in [1e-4, 1e-3, 1e-2]:
            for order in [2, 5, 8]:
                out = lms_filter(noisy, clean, mu=mu, filter_order=order)
                snr = compute_snr(clean, out)
                if snr > best_snr:
                    best_snr = snr
                    best_out = out
                    best_params = {'mu': mu, 'order': order}
        return best_out, best_snr, best_params

    def tune_rls(self, noisy: np.ndarray, clean: np.ndarray) -> FilterResult:
        best_snr = -np.inf
        best_out = None
        best_params = {}
        for delta in [0.01, 1.0]:
            for lam in [0.9, 0.99]:
                out = rls_filter(noisy, clean,
                                 delta=delta, lam=lam,
                                 filter_order=self.filter_order)
                snr = compute_snr(clean, out)
                if snr > best_snr:
                    best_snr = snr
                    best_out = out
                    best_params = {'delta': delta, 'lam': lam}
        return best_out, best_snr, best_params

    def tune_wiener(self, noisy: np.ndarray, clean: np.ndarray) -> FilterResult:
        best_snr = -np.inf
        best_out = None
        best_params = {}
        for order in [2, 5, 8]:
            out = wiener_filter(noisy, clean, filter_order=order)
            snr = compute_snr(clean, out)
            if snr > best_snr:
                best_snr = snr
                best_out = out
                best_params = {'order': order}
        return best_out, best_snr, best_params

    def tune_kalman(self, noisy: np.ndarray, clean: np.ndarray
                   ) -> FilterResult:
        best_snr = -np.inf
        best_out = None
        best_params = {}
        for q in [1e-6, 1e-5, 1e-4, 1e-3]:
            for r in [0.01, 0.1, 0.5, 1.0]:
                out = sinusoidal_kalman_filter(noisy, freqs=(5, 10),
                                               dt=1.0, q=q, r=r)
                snr = compute_snr(clean, out)
                if snr > best_snr:
                    best_snr = snr
                    best_out = out
                    best_params = {'q': q, 'r': r}
        return best_out, best_snr, best_params

    def tune_all(self, noisy: np.ndarray, clean: np.ndarray
                ) -> Dict[str, FilterResult]:
        """
        Returns a dict mapping:
          'lms'    -> (out, snr, params)
          'rls'    -> ...
          'wiener' -> ...
          'kalman' -> ...
        """
        return {
            'lms':    self.tune_lms(noisy, clean),
            'rls':    self.tune_rls(noisy, clean),
            'wiener': self.tune_wiener(noisy, clean),
            'kalman': self.tune_kalman(noisy, clean),
        }

