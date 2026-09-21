from __future__ import annotations

import numpy as np


class BoundaryHandler:
    @staticmethod
    def adaptive_extension_ratio(signal: np.ndarray) -> float:
        N = len(signal)
        if N < 500:
            return 0.30
        if N < 2000:
            return 0.20
        return 0.15

    @staticmethod
    def auto_window_alpha(
        signal: np.ndarray, min_alpha: float = 0.01, max_alpha: float = 0.1
    ) -> float:
        s = np.asarray(signal, dtype=np.float64)
        if s.size < 8:
            return min_alpha
        deriv = np.diff(s)
        deriv_var = np.var(deriv)
        norm_var = deriv_var / (np.mean(np.abs(s)) ** 2 + 1e-12)
        smoothness = 1.0 / (1.0 + norm_var)
        return float(min_alpha + (max_alpha - min_alpha) * smoothness)

    @staticmethod
    def extend_signal(
        signal: np.ndarray, method: str, extension_ratio: float
    ) -> tuple[np.ndarray, int, int]:
        s = np.asarray(signal, dtype=np.float64)
        N = s.size
        ext_len = int(N * float(extension_ratio))
        if ext_len <= 0 or method == "none":
            return s, 0, 0

        if ext_len >= N - 2:
            ext_len = max(1, N // 4)

        if method == "mirror":
            left_ext = s[1 : ext_len + 1][::-1]
            right_ext = s[-(ext_len + 1) : -1][::-1]
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "reflect":
            left_val, right_val = s[0], s[-1]
            left_ext = 2 * left_val - s[1 : ext_len + 1][::-1]
            right_ext = 2 * right_val - s[-(ext_len + 1) : -1][::-1]
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "linear":
            left_slope = s[1] - s[0]
            right_slope = s[-1] - s[-2]
            left_ext = s[0] + left_slope * np.arange(-ext_len, 0)
            right_ext = s[-1] + right_slope * np.arange(1, ext_len + 1)
            out = np.concatenate([left_ext, s, right_ext])

        elif method == "constant":
            out = np.concatenate([np.full(ext_len, s[0]), s, np.full(ext_len, s[-1])])

        else:
            return s, 0, 0

        return out, ext_len, ext_len

    @staticmethod
    def smooth_edge_junction(
        extended: np.ndarray,
        original_len: int,
        ext_len: int,
        smooth_ratio: float = 0.02,
    ) -> np.ndarray:
        if ext_len <= 0:
            return extended
        taper_len = max(2, int(original_len * smooth_ratio))

        # FIX: clamp with ext_len
        taper_len = min(taper_len, ext_len)
        if taper_len < 1:
            return extended

        ramp_up = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_len)))
        ramp_down = ramp_up[::-1]

        out = extended.copy()
        out[ext_len - taper_len : ext_len] *= ramp_up
        end_idx = ext_len + original_len
        out[end_idx : end_idx + taper_len] *= ramp_down
        return out

    @staticmethod
    def taper_boundaries(
        modes: list[np.ndarray], taper_length: int
    ) -> list[np.ndarray]:
        out = []
        for m in modes:
            x = np.asarray(m, dtype=np.float64).copy()
            N = x.size
            tlen = int(min(max(2, taper_length), N // 4))
            if tlen <= 1:
                out.append(x)
                continue

            w = np.ones(N, dtype=np.float64)
            w[:tlen] = np.sin(np.linspace(0, np.pi / 2, tlen)) ** 2
            w[-tlen:] = np.cos(np.linspace(0, np.pi / 2, tlen)) ** 2
            out.append(x * w)
        return out
