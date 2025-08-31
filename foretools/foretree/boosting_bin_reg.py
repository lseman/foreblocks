# bin_registry.py
# Unified binning registry with Numba-accelerated prebin and histogram builders.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

# ============================================================================
# Numba kernels: prebinning + histograms
# ============================================================================

@njit(cache=True, nogil=True, fastmath=False)
def _binary_search_bin(e: np.ndarray, x: float) -> int:
    """
    Given strictly increasing edges e (len = nb+1), return bin id k in [0..nb-1]
    such that e[k] <= x < e[k+1]. Endpoints clamp to [0, nb-1].
    """
    nb = e.shape[0] - 1
    if nb <= 0:
        return 0
    lo = 0
    hi = nb
    if x < e[0]:
        return 0
    if x >= e[nb]:
        return nb - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x >= e[mid]:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True, fastmath=False)
def _prebin_column_njit(col: np.ndarray, edges: np.ndarray, miss_id: int) -> np.ndarray:
    """
    Bin a single column (float64) using edges (float64). Non-finite -> miss_id.
    Returns int32 codes in [0..nb-1] or miss_id.
    """
    n = col.shape[0]
    out = np.empty(n, dtype=np.int32)
    nb = edges.shape[0] - 1
    for i in range(n):
        v = col[i]
        if not np.isfinite(v):
            out[i] = miss_id
        else:
            k = _binary_search_bin(edges, v)
            if k < 0:
                k = 0
            elif k >= nb:
                k = nb - 1
            out[i] = k
    return out


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _prebin_matrix_flat_njit(
    X_local: np.ndarray,           # (n, p) float64
    flat_edges: np.ndarray,        # concatenated edges (float64)
    edge_offsets: np.ndarray,      # start idx in flat_edges per feature (int32, len=p)
    edge_counts: np.ndarray,       # number of edges per feature (int32, len=p)
    miss_id: int,
) -> np.ndarray:
    """
    Prebin all columns using flattened edges (no per-node overrides).
    Returns int32 codes.
    """
    n, p = X_local.shape
    codes = np.empty((n, p), dtype=np.int32)
    for j in prange(p):
        off = edge_offsets[j]
        cnt = edge_counts[j]
        nb = cnt - 1
        e = flat_edges[off:off + cnt]
        for i in range(n):
            v = X_local[i, j]
            if not np.isfinite(v):
                codes[i, j] = miss_id
            else:
                # binary search in e
                lo, hi = 0, nb
                if v < e[0]:
                    codes[i, j] = 0
                    continue
                if v >= e[nb]:
                    codes[i, j] = nb - 1
                    continue
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if v >= e[mid]:
                        lo = mid
                    else:
                        hi = mid
                codes[i, j] = lo
    return codes


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _build_histograms_from_codes_njit(
    codes: np.ndarray,        # (n, p) int32 codes for this subset (includes miss_id)
    g_node: np.ndarray,       # (n,) float64
    h_node: np.ndarray,       # (n,) float64
    n_bins_total: int,        # finite bins + reserved missing bin
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (p, n_bins_total) histograms for gradients and hessians from codes.
    """
    n, p = codes.shape
    hg = np.zeros((p, n_bins_total), dtype=np.float64)
    hh = np.zeros((p, n_bins_total), dtype=np.float64)
    for j in prange(p):
        gj = hg[j]
        hj = hh[j]
        for i in range(n):
            b = codes[i, j]
            if b < 0:
                continue
            if b >= n_bins_total:
                b = n_bins_total - 1
            gj[b] += g_node[i]
            hj[b] += h_node[i]
    return hg, hh


# ============================================================================
# Helpers for flattened edges
# ============================================================================

def _flatten_edges_list(edges_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten a list of edges arrays into a single vector with offsets and counts.
    - edges_list: list of arrays (strictly increasing, len>=2)
    Returns:
      flat_edges (float64), edge_offsets (int32), edge_counts (int32)
    """
    p = len(edges_list)
    counts = np.empty(p, dtype=np.int32)
    total = 0
    for j, e in enumerate(edges_list):
        c = 2 if (e is None or e.size < 2) else int(e.size)
        counts[j] = c
        total += c

    flat = np.empty(total, dtype=np.float64)
    offsets = np.empty(p, dtype=np.int32)
    pos = 0
    for j, e in enumerate(edges_list):
        offsets[j] = pos
        if e is None or e.size < 2:
            flat[pos:pos + 2] = np.array([0.0, 1.0], dtype=np.float64)
            pos += 2
        else:
            m = e.size
            flat[pos:pos + m] = e.astype(np.float64, copy=False)
            pos += m
    return flat, offsets, counts


# ============================================================================
# Bin registry data structures
# ============================================================================

BinMode = str  # e.g. "hist" | "approx" | "adaptive"


@dataclass
class BinLayout:
    """
    Compact layout used by a binning mode:
      - actual_max_bins: number of finite bins (excludes missing)
      - missing_bin_id: reserved id for missing (== actual_max_bins)
      - out_dtype: dtype for codes (np.uint8/np.uint16/np.int32, etc.)
    """
    actual_max_bins: int
    missing_bin_id: int
    out_dtype: np.dtype
    mode: Optional[BinMode] = None  # optional; not required by the tree


@dataclass
class BinSpec:
    """
    Stores edges for each global feature in a given mode, plus optional node-level overrides.
      - edges_list: list of per-feature edges aligned to local feature order
      - edges: dict {global_feature_index -> edges ndarray}
      - node_overrides: {(node_id, gfi) -> refined edges ndarray}
    """
    mode: str
    feature_indices: np.ndarray
    layout: BinLayout
    edges_list: List[np.ndarray] = field(default_factory=list)
    edges: Dict[int, np.ndarray] = field(default_factory=dict)
    node_overrides: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    cached_full_codes: Optional[np.ndarray] = None
    _row_prebin_cache: Dict[str, Tuple[np.ndarray, int]] = field(default_factory=dict)

    def get_edges(self, gfi: int, node_id: Optional[int] = None) -> Optional[np.ndarray]:
        if node_id is not None:
            e = self.node_overrides.get((int(node_id), int(gfi)))
            if e is not None and e.size >= 2:
                return e
        # exact map first
        e = self.edges.get(int(gfi))
        if e is not None:
            return e
        # fallback via local list (slower path)
        fi = self.feature_indices
        pos = np.where(fi == int(gfi))[0]
        if pos.size == 0:
            return None
        lfi = int(pos[0])
        if 0 <= lfi < len(self.edges_list):
            return self.edges_list[lfi]
        return None

    def get_edges_list(self, node_id: Optional[int] = None) -> List[np.ndarray]:
        """
        Returns per-feature edges aligned to local order; if node_id is provided
        and there are overrides, returns a list that includes overrides where present.
        """
        if node_id is None or not self.node_overrides:
            return self.edges_list
        out = []
        nid = int(node_id)
        for local_pos, gfi in enumerate(self.feature_indices):
            e = self.node_overrides.get((nid, int(gfi)))
            if e is None:
                e = self.edges_list[local_pos]
            out.append(e)
        return out


class BinRegistry:
    """
    Registry of bin specs per mode ("hist", "approx", "adaptive", ...).
    Provides consistent prebinning with reserved missing id and supports per-node overrides.

    Usage:
      bins = BinRegistry(feature_indices)
      bins.register_global_edges(mode="approx", edges_list_in_feature_index_order=edges_list,
                                 actual_max_bins=255, out_dtype=np.uint8)
      codes, miss_id = bins.prebin_matrix(X_local, mode="approx")  # X columns aligned to feature_indices
      # Optional per-node override:
      bins.set_node_override("adaptive", node_id, gfi, refined_edges)
      codes_node, miss_id = bins.prebin_matrix(X_node_local, mode="adaptive", node_id=node_id)
    """

    def __init__(self, feature_indices: np.ndarray, feature_types: Optional[Dict[int, str]] = None):
        self.feature_indices = np.asarray(feature_indices, dtype=np.int32)
        self._specs: Dict[BinMode, BinSpec] = {}
        self._default_mode: Optional[BinMode] = None
        # cache: (id(X), node_id or -1, mode_hash, shape) -> codes ndarray
        self._prebin_cache: Dict[Tuple[int, int, int, Tuple[int, int]], np.ndarray] = {}
        # global->local feature map aligned to feature_indices order
        self.feature_map: Dict[int, int] = {int(f): i for i, f in enumerate(self.feature_indices)}
        self.feature_types: Dict[int, str] = feature_types or {}
    # ----------------- registration -----------------

    def set_feature_types(self, feature_types: Dict[int, str]) -> None:
        """Optional: declare categorical vs numeric features."""
        self.feature_types = {int(k): ("cat" if v == "cat" else "num") for k, v in feature_types.items()}
    
    def is_categorical(self, gfi: int) -> bool:
        return self.feature_types.get(int(gfi), "num") == "cat"

    def register_global_edges(
        self,
        *,
        mode: str,
        edges_list_in_feature_index_order: List[np.ndarray],
        actual_max_bins: int,
        out_dtype: np.dtype,
    ) -> None:
        """
        Register the edges for all features (aligned with feature_indices order) in a mode.
        The reserved missing bin id is always 'actual_max_bins'.
        """
        layout = BinLayout(
            actual_max_bins=int(actual_max_bins),
            missing_bin_id=int(actual_max_bins),
            out_dtype=np.dtype(out_dtype),
            mode=mode,
        )
        spec = BinSpec(
            mode=mode,
            feature_indices=self.feature_indices.copy(),
            layout=layout,
        )
        # Normalize edges; store both positional list and dict by global feature id
        norm = [np.asarray(e, dtype=np.float64) for e in edges_list_in_feature_index_order]
        spec.edges_list = norm
        spec.edges = {int(gfi): norm[i] for i, gfi in enumerate(self.feature_indices)}

        # reset per-mode caches/overrides
        spec.node_overrides = {}
        spec.cached_full_codes = None
        spec._row_prebin_cache = {}

        self._specs[mode] = spec
        if self._default_mode is None:
            self._default_mode = mode

    def set_node_override(self, mode: BinMode, node_id: int, gfi: int, edges: np.ndarray) -> None:
        spec = self._specs.get(mode)
        if spec is None:
            raise KeyError(f"Mode '{mode}' not registered")
        arr = np.asarray(edges, dtype=np.float64)
        if arr.size < 2:
            raise ValueError("Override edges must have at least 2 values (>=1 bin).")
        nb = arr.size - 1
        # ✅ Enforce capacity defined by this mode’s layout
        if nb > spec.layout.actual_max_bins:
            raise ValueError(
                f"Override adds {nb} finite bins but layout allows only {spec.layout.actual_max_bins}. "
                "Increase layout (re-register mode) or reduce override bins."
            )
        spec.node_overrides[(int(node_id), int(gfi))] = arr

    # ----------------- getters -----------------

    def get_layout(self, *, mode: Optional[BinMode] = None) -> BinLayout:
        mode = mode or self._default_mode
        if mode is None or mode not in self._specs:
            raise KeyError("No bin mode registered in BinRegistry.")
        return self._specs[mode].layout

    def get_edges(self, gfi: int, *, mode: Optional[BinMode] = None, node_id: Optional[int] = None) -> Optional[np.ndarray]:
        mode = mode or self._default_mode
        spec = self._specs.get(mode)
        if spec is None:
            return None
        return spec.get_edges(int(gfi), node_id=node_id)

    def has_node_override(self, mode: str, node_id: int, gfi: int) -> bool:
        spec = self._specs.get(mode)
        if spec is None:
            return False
        return (int(node_id), int(gfi)) in spec.node_overrides

    def node_has_any_override(self, mode: str, node_id: int) -> bool:
        spec = self._specs.get(mode)
        if spec is None or not spec.node_overrides:
            return False
        nid = int(node_id)
        for (nid_k, _gfi) in spec.node_overrides.keys():
            if nid_k == nid:
                return True
        return False

    def get_codes_view(self, mode: str) -> Optional[np.ndarray]:
        spec = self._specs.get(mode)
        if spec is None:
            return None
        return spec.cached_full_codes

    # ----------------- prebin API -----------------

    def _has_overrides(self, mode: BinMode) -> bool:
        spec = self._specs.get(mode)
        return (spec is not None) and (len(spec.node_overrides) > 0)

    def prebin_matrix(
        self,
        X_local: np.ndarray,
        *,
        mode: Optional[BinMode] = None,
        node_id: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Prebin X_local whose columns are aligned to self.feature_indices order.
        Returns (codes, missing_bin_id).
        """
        mode = mode or self._default_mode
        if mode not in self._specs:
            raise KeyError(f"Mode '{mode}' not registered")
        spec = self._specs[mode]
        layout = spec.layout

        # Cache only when no overrides and node_id is None
        use_cache = (node_id is None) and (not self._has_overrides(mode))
        shape = X_local.shape
        key_hash = hash(mode) if cache_key is None else hash(cache_key)

        if use_cache:
            ck = (id(X_local), -1, key_hash, shape)
            cached = self._prebin_cache.get(ck)
            if cached is not None and cached.shape == shape:
                return cached, layout.missing_bin_id

        # Fast path (no overrides) -> numba flattened
        if node_id is None and not self._has_overrides(mode):
            edges_list = spec.edges_list if spec.edges_list else [
                spec.get_edges(int(gfi)) for gfi in self.feature_indices
            ]
            flat, offs, cnts = _flatten_edges_list(edges_list)
            Xf = X_local.astype(np.float64, copy=False)
            codes_i32 = _prebin_matrix_flat_njit(
                Xf,
                flat,
                offs.astype(np.int32),
                cnts.astype(np.int32),
                layout.missing_bin_id,
            )
            codes = codes_i32.astype(layout.out_dtype, copy=False)
            if use_cache:
                self._prebin_cache[(id(X_local), -1, key_hash, shape)] = codes
            return codes, layout.missing_bin_id

        # Override-aware fallback: per column (still numba for the inner loop)
        n, p = shape
        out = np.empty((n, p), dtype=layout.out_dtype)
        for local_pos, gfi in enumerate(self.feature_indices):
            e = spec.get_edges(int(gfi), node_id=node_id)
            if e is None or e.size < 2:
                e = np.array([0.0, 1.0], dtype=np.float64)
            col_codes = _prebin_column_njit(
                X_local[:, local_pos].astype(np.float64, copy=False),
                e.astype(np.float64, copy=False),
                layout.missing_bin_id,
            )
            out[:, local_pos] = col_codes.astype(layout.out_dtype, copy=False)
        # cache row-level result if a key is provided
        if cache_key is not None:
            spec._row_prebin_cache[cache_key] = (out, layout.missing_bin_id)
        return out, layout.missing_bin_id

    # ----------------- histograms API -----------------

    @staticmethod
    def build_histograms_from_codes(
        codes_sub: np.ndarray,    # (n_sub, p) int codes for a node subset
        g_node: np.ndarray,       # (n_sub,) float64 gradients
        h_node: np.ndarray,       # (n_sub,) float64 hessians
        n_bins_total: int,        # finite bins + reserved missing
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba-accelerated histogram builder for a node subset.
        Returns (hist_g, hist_h) with shape (p, n_bins_total).
        """
        codes_i32 = codes_sub.astype(np.int32, copy=False)
        g64 = g_node.astype(np.float64, copy=False)
        h64 = h_node.astype(np.float64, copy=False)
        return _build_histograms_from_codes_njit(codes_i32, g64, h64, int(n_bins_total))
