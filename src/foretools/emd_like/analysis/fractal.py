from __future__ import annotations

import numpy as np


class FractalDimension:
    """
    Multiple methods for estimating fractal/complexity dimension of 1D signals.
    """

    @staticmethod
    def box_counting(
        signal: np.ndarray,
        n_scales: int = 12,
        min_box: int = 4,
        max_box_ratio: float = 0.25,
    ) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 64 or np.allclose(x, x[0], atol=1e-12):
            return 1.0

        x_centered = x - np.median(x)
        mad = np.median(np.abs(x_centered)) + 1e-12
        x_norm = x_centered / (6.0 * mad)
        x_norm = np.clip(x_norm, -1.0, 1.0)
        y = 0.5 * (x_norm + 1.0)

        max_box = max(min_box * 2, int(N * max_box_ratio))
        sizes = np.unique(
            np.logspace(np.log10(min_box), np.log10(max_box), n_scales).astype(int)
        )
        sizes = sizes[(sizes > 1) & (sizes < N // 2)]
        if len(sizes) < 4:
            return 1.0

        counts = []
        inv_sizes = []
        for box_size in sizes:
            n_boxes_x = int(np.ceil(N / box_size))
            n_boxes_y = max(2, int(np.ceil(N / box_size)))
            box_x = (np.arange(N) // box_size).astype(int)
            box_y = np.floor(y * n_boxes_y).astype(int)
            box_y = np.clip(box_y, 0, n_boxes_y - 1)

            occupied = set(zip(box_x.tolist(), box_y.tolist()))
            counts.append(len(occupied))
            inv_sizes.append(1.0 / box_size)

        if len(counts) < 4:
            return 1.0

        log_inv = np.log(np.array(inv_sizes))
        log_cnt = np.log(np.array(counts) + 1e-12)
        coeffs = np.polyfit(log_inv, log_cnt, deg=1)
        dim = coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 0.5, 2.5))

    @staticmethod
    def differential_box_counting(
        signal: np.ndarray,
        n_scales: int = 10,
        min_scale: int = 2,
        max_scale_ratio: float = 0.25,
    ) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 64:
            return 1.0

        G = 256
        x_min, x_max = np.min(x), np.max(x)
        if np.isclose(x_max, x_min):
            return 1.0
        x_norm = ((x - x_min) / (x_max - x_min) * (G - 1)).astype(int)

        max_scale = max(min_scale + 1, int(N * max_scale_ratio))
        scales = np.unique(
            np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        )
        scales = scales[(scales >= min_scale) & (scales <= N // 4)]
        if len(scales) < 4:
            return FractalDimension.box_counting(signal)

        counts = []
        log_scales = []
        for r in scales:
            s = max(1, G // r)
            n_columns = N // r
            if n_columns < 1:
                continue

            total_boxes = 0
            for col in range(n_columns):
                start_idx = col * r
                end_idx = min(start_idx + r, N)
                column_vals = x_norm[start_idx:end_idx]
                if len(column_vals) == 0:
                    continue
                col_min = np.min(column_vals)
                col_max = np.max(column_vals)
                k_min = col_min // s
                k_max = col_max // s
                total_boxes += k_max - k_min + 1

            if total_boxes > 0:
                counts.append(total_boxes)
                log_scales.append(np.log(1.0 / r))

        if len(counts) < 4:
            return FractalDimension.box_counting(signal)

        log_counts = np.log(np.array(counts) + 1e-12)
        log_scales = np.array(log_scales)
        coeffs = np.polyfit(log_scales, log_counts, deg=1)
        dim = coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def higuchi(signal: np.ndarray, k_max: int = 10) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 20:
            return 1.0

        k_max = min(k_max, N // 4)
        if k_max < 2:
            return 1.0

        L = []
        k_values = []
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, N, k)
                if len(idx) < 2:
                    continue
                n_segments = len(idx) - 1
                if n_segments < 1:
                    continue
                sub_series = x[idx]
                length = np.sum(np.abs(np.diff(sub_series)))
                norm_factor = (N - 1) / (k * n_segments * k)
                Lk.append(length * norm_factor)
            if len(Lk) > 0:
                L.append(np.mean(Lk))
                k_values.append(k)

        if len(L) < 3:
            return 1.0

        log_k = np.log(np.array(k_values))
        log_L = np.log(np.array(L) + 1e-12)
        coeffs = np.polyfit(log_k, log_L, deg=1)
        dim = -coeffs[0]
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def katz(signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 10:
            return 1.0

        L = np.sum(np.sqrt(1 + np.diff(x) ** 2))
        distances = np.sqrt(np.arange(N) ** 2 + (x - x[0]) ** 2)
        d = np.max(distances)
        if d < 1e-12 or L < 1e-12:
            return 1.0

        n = N - 1
        dim = np.log10(n) / (np.log10(n) + np.log10(d / L))
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def petrosian(signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=np.float64)
        N = x.size
        if N < 10:
            return 1.0

        dx = np.diff(x)
        sign_changes = np.sum(dx[:-1] * dx[1:] < 0)
        if sign_changes == 0:
            return 1.0

        dim = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes)))
        if not np.isfinite(dim):
            return 1.0
        return float(np.clip(dim, 1.0, 2.0))

    @staticmethod
    def estimate(signal: np.ndarray, method: str = "auto") -> float:
        if method == "higuchi":
            return FractalDimension.higuchi(signal)
        if method == "dbc":
            return FractalDimension.differential_box_counting(signal)
        if method == "box":
            return FractalDimension.box_counting(signal)
        if method == "katz":
            return FractalDimension.katz(signal)
        if method == "petrosian":
            return FractalDimension.petrosian(signal)
        if method == "auto":
            estimates = [
                FractalDimension.higuchi(signal),
                FractalDimension.differential_box_counting(signal),
                FractalDimension.katz(signal),
            ]
            return float(np.median(estimates))
        raise ValueError(f"Unknown FD method: {method}")


def fractal_dimension(signal: np.ndarray, **kwargs) -> float:
    """
    Composite fractal-dimension estimate.

    Despite the legacy name ``box_counting_dimension``, this function does not
    return a pure box-counting dimension. It returns the median of the Higuchi,
    differential box-counting, and Katz estimates for a more stable aggregate
    complexity score.
    """
    return FractalDimension.estimate(signal, method="auto")


box_counting_dimension = fractal_dimension
