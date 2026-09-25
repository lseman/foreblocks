"""Numba kernels for `Rocket`, `MiniRocket` and `MultiRocket`.

All kernels take a `[N, C, L]` float32 panel; univariate input is `C == 1`.
Multivariate kernels sum their convolution over a per-kernel channel subset,
as in the reference multivariate implementations.

Inner loops index pre-sliced views with a counter that starts at 0 (`dst[u]`,
`src[u]`) instead of offset indices like `x[t + shift]`. Numba must guard a
possibly-negative index with a wraparound check, and that check alone stops
LLVM from vectorizing the loop (roughly an order of magnitude per core).
"""

from __future__ import annotations

import numba
import numpy as np
from numba import prange

# The 84 MiniRocket kernels: positions of the three taps weighted +2 among
# nine taps otherwise weighted -1 (all C(9, 3) combinations, in order).
MINIROCKET_INDICES = np.array(
    [
        (i, j, k)
        for i in range(9)
        for j in range(i + 1, 9)
        for k in range(j + 1, 9)
    ],
    dtype=np.int32,
)
NUM_MINIROCKET_KERNELS = len(MINIROCKET_INDICES)


# --------------------------------------------------------------------------- ROCKET


@numba.njit(fastmath=True, cache=True)
def _apply_rocket_kernel(x, weights, length, bias, dilation, padding, channels, scratch):
    """Tap-major dilated convolution into `scratch`, then (PPV, max) pooling.
    Per output position the terms are summed in the same (channel, tap) order
    as a direct convolution.
    """
    input_length = x.shape[1]
    output_length = input_length + 2 * padding - (length - 1) * dilation
    out = scratch[:output_length]
    out[:] = bias
    for c in range(len(channels)):
        row = x[channels[c]]
        for j in range(length):
            w = weights[c * length + j]
            # out[o] (position o - padding) reads row[o + shift].
            shift = j * dilation - padding
            lo = max(0, -shift)
            hi = min(output_length, input_length - shift)
            if hi <= lo:
                continue
            dst = out[lo:hi]
            src = row[lo + shift : hi + shift]
            for u in range(hi - lo):
                dst[u] += w * src[u]
    ppv = 0
    maximum = out[0]
    for t in range(output_length):
        value = out[t]
        ppv += 1 if value > 0 else 0
        maximum = max(maximum, value)
    return ppv / output_length, maximum


@numba.njit(parallel=True, fastmath=True, cache=True)
def rocket_transform(
    x, weights, weight_offsets, lengths, biases, dilations, paddings, channel_offsets, channel_indices
):
    """Returns `[N, 2 * num_kernels]`: interleaved (PPV, max) per kernel."""
    n_series = x.shape[0]
    n_kernels = len(lengths)
    out = np.empty((n_series, 2 * n_kernels), dtype=np.float32)
    for n in prange(n_series):
        scratch = np.empty(x.shape[2] + 1, dtype=np.float32)
        for k in range(n_kernels):
            ppv, maximum = _apply_rocket_kernel(
                x[n],
                weights[weight_offsets[k] : weight_offsets[k + 1]],
                lengths[k],
                biases[k],
                dilations[k],
                paddings[k],
                channel_indices[channel_offsets[k] : channel_offsets[k + 1]],
                scratch,
            )
            out[n, 2 * k] = ppv
            out[n, 2 * k + 1] = maximum
    return out


# ----------------------------------------------------------- MiniRocket / MultiRocket


@numba.njit(fastmath=True, cache=True)
def _conv9(x, dilation, taps, channels, out):
    """`out[t] = sum_c sum_j w_j x[c, t + (j - 4) * dilation]`, zero padded, with
    `w_j = 2` for `j in taps` and `-1` otherwise.
    """
    length = x.shape[1]
    out[:] = 0.0
    for c in range(len(channels)):
        row = x[channels[c]]
        for j in range(9):
            w = 2.0 if (j == taps[0] or j == taps[1] or j == taps[2]) else -1.0
            shift = (j - 4) * dilation
            lo = max(0, -shift)
            hi = min(length, length - shift)
            if hi <= lo:
                continue
            dst = out[lo:hi]
            src = row[lo + shift : hi + shift]
            for u in range(hi - lo):
                dst[u] += w * src[u]


@numba.njit(cache=True)
def _quantile_sorted(values, q):
    position = q * (len(values) - 1)
    lo = int(np.floor(position))
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


@numba.njit(parallel=True, cache=True)
def minirocket_fit_biases(
    x,
    example_indices,
    dilations,
    features_per_dilation,
    quantiles,
    channel_offsets,
    channel_indices,
):
    """One random training example per (dilation, kernel) combination; biases
    are low-discrepancy quantiles of that example's convolution output.
    """
    length = x.shape[2]
    n_combinations = len(dilations) * NUM_MINIROCKET_KERNELS
    feature_offsets = np.zeros(n_combinations + 1, dtype=np.int64)
    for comb in range(n_combinations):
        count = features_per_dilation[comb // NUM_MINIROCKET_KERNELS]
        feature_offsets[comb + 1] = feature_offsets[comb] + count
    biases = np.empty(feature_offsets[-1], dtype=np.float32)
    for comb in prange(n_combinations):
        conv = np.empty(length, dtype=np.float64)
        _conv9(
            x[example_indices[comb]],
            dilations[comb // NUM_MINIROCKET_KERNELS],
            MINIROCKET_INDICES[comb % NUM_MINIROCKET_KERNELS],
            channel_indices[channel_offsets[comb] : channel_offsets[comb + 1]],
            conv,
        )
        conv.sort()
        for f in range(feature_offsets[comb], feature_offsets[comb + 1]):
            biases[f] = _quantile_sorted(conv, quantiles[f])
    return biases


@numba.njit(fastmath=True, cache=True)
def _fill_alpha_gamma(x, dilation, c_alpha, c_gamma):
    """Per channel: `c_alpha` = the all-(-1) kernel's output, `c_gamma[j]` =
    the input shifted to tap `j`, so each MiniRocket kernel is
    `c_alpha + 3 * sum(taps)`, computed as `c_alpha + c_gamma[i] + ...` with
    `c_gamma` pre-scaled by 3.
    """
    n_channels, length = x.shape
    for c in range(n_channels):
        row = x[c]
        alpha = c_alpha[c]
        center = c_gamma[4, c]
        for t in range(length):
            alpha[t] = -row[t]
            center[t] = 3.0 * row[t]
        for j in range(9):
            if j == 4:
                continue
            shift = (j - 4) * dilation
            gamma = c_gamma[j, c]
            gamma[:] = 0.0
            lo = max(0, -shift)
            hi = min(length, length - shift)
            if hi <= lo:
                continue
            a = alpha[lo:hi]
            g = gamma[lo:hi]
            src = row[lo + shift : hi + shift]
            for u in range(hi - lo):
                a[u] -= src[u]
                g[u] = 3.0 * src[u]


# Pooling operators, in output-block order. MiniRocket uses the first; MultiRocket
# the first four; SelF-Rocket may pick any of the five.
POOLING_OPERATORS = ("ppv", "mpv", "mipv", "lspv", "gmp")


@numba.njit(fastmath=True, cache=True)
def _pool(conv, lo, hi, bias, out, row, feature, stride, n_pool):
    """The first `n_pool` of (PPV, MPV, MIPV, LSPV, GMP) of `conv - bias` over
    `conv[lo:hi]`, written to `out[row, i * stride + feature]` for operator `i`.
    PPV/MPV/MIPV are one branch-free (vectorizable) pass; LSPV, a serial run
    length, and GMP get their own passes (LSPV is skipped when no value is
    positive).
    """
    seg = conv[lo:hi]
    n = hi - lo
    count = 0
    if n_pool == 1:
        for t in range(n):
            count += 1 if seg[t] - bias > 0 else 0
        out[row, feature] = count / n
        return
    total = 0.0
    index_total = 0
    for t in range(n):
        value = seg[t] - bias
        positive = value > 0
        count += 1 if positive else 0
        total += value if positive else 0.0
        index_total += t if positive else 0
    longest = 0
    if count > 0:
        last = -1  # index of the latest non-positive value
        for t in range(n):
            if not seg[t] - bias > 0:
                last = t
            longest = max(longest, t - last)
    out[row, feature] = count / n
    out[row, stride + feature] = total / count if count > 0 else 0.0
    out[row, 2 * stride + feature] = index_total / count if count > 0 else -1.0
    out[row, 3 * stride + feature] = longest
    if n_pool > 4:
        maximum = seg[0]
        for t in range(n):
            maximum = max(maximum, seg[t])
        out[row, 4 * stride + feature] = maximum - bias


@numba.njit(parallel=True, fastmath=True, cache=True)
def minirocket_transform(
    x,
    dilations,
    features_per_dilation,
    biases,
    channel_offsets,
    channel_indices,
    n_pool,
    active,
):
    """Returns `n_pool` blocks `[N, n_pool * F]` of the operators in
    `POOLING_OPERATORS` order, where `F` counts the features of the
    (dilation, kernel) combinations with `active[comb]` set; inactive (pruned)
    combinations are neither convolved nor pooled. Kernel/dilation pairs
    alternate between "same" padding and valid-only pooling, as in the
    reference.
    """
    n_series, n_channels, length = x.shape
    n_dilations = len(dilations)
    n_features = 0
    for comb in range(n_dilations * NUM_MINIROCKET_KERNELS):
        if active[comb]:
            n_features += features_per_dilation[comb // NUM_MINIROCKET_KERNELS]
    out = np.empty((n_series, n_pool * n_features), dtype=np.float32)
    for n in prange(n_series):
        c_alpha = np.empty((n_channels, length), dtype=np.float32)
        c_gamma = np.empty((9, n_channels, length), dtype=np.float32)
        conv = np.empty(length, dtype=np.float32)
        bias_index = 0
        feature = 0
        for d in range(n_dilations):
            dilation = dilations[d]
            padding = 4 * dilation
            count = features_per_dilation[d]
            first = d * NUM_MINIROCKET_KERNELS
            any_active = False
            for k in range(NUM_MINIROCKET_KERNELS):
                any_active = any_active or active[first + k]
            if not any_active:
                bias_index += count * NUM_MINIROCKET_KERNELS
                continue
            _fill_alpha_gamma(x[n], dilation, c_alpha, c_gamma)
            for k in range(NUM_MINIROCKET_KERNELS):
                comb = first + k
                if not active[comb]:
                    bias_index += count
                    continue
                i0 = MINIROCKET_INDICES[k, 0]
                i1 = MINIROCKET_INDICES[k, 1]
                i2 = MINIROCKET_INDICES[k, 2]
                conv[:] = 0.0
                for ci in range(channel_offsets[comb], channel_offsets[comb + 1]):
                    c = channel_indices[ci]
                    a = c_alpha[c]
                    g0 = c_gamma[i0, c]
                    g1 = c_gamma[i1, c]
                    g2 = c_gamma[i2, c]
                    for t in range(length):
                        conv[t] += a[t] + g0[t] + g1[t] + g2[t]
                if (d + k) % 2 == 0:
                    lo, hi = 0, length
                else:
                    lo, hi = padding, length - padding
                for _ in range(count):
                    _pool(conv, lo, hi, biases[bias_index], out, n, feature, n_features, n_pool)
                    bias_index += 1
                    feature += 1
    return out
