# bin_registry.py
# Unified binning registry with Numba-accelerated prebin and histogram builders.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

# ============================================================================
# Utilities for edge hygiene and metadata
# ============================================================================

def _strict_increasing(e: np.ndarray) -> np.ndarray:
    """Ensure strictly increasing edges (float64) with nextafter tie-breaking."""
    e = np.asarray(e, dtype=np.float64)
    if e.size < 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    out = e.copy()
    for i in range(1, out.size):
        if not (out[i] > out[i - 1]):
            out[i] = np.nextafter(out[i - 1], np.inf)
    return out


@njit(cache=True, nogil=True, fastmath=False)
def _uniform_params_njit(e: np.ndarray, tol: float = 1e-9) -> Tuple[bool, float, float, int]:
    """
    Detect near-uniform bins: return (is_uniform, lo, inv_w, nb).
    Uses relative deviation of step sizes from the mean.
    """
    nb = e.shape[0] - 1
    if nb <= 0:
        return False, 0.0, 1.0, 0
    
    # Compute diffs and mean in one pass
    total = 0.0
    for i in range(nb):
        total += e[i + 1] - e[i]
    m = total / nb
    
    if m <= 0.0:
        return False, 0.0, 1.0, nb
    
    # Check uniformity
    max_dev = 0.0
    for i in range(nb):
        diff = e[i + 1] - e[i]
        dev = abs(diff - m)
        if dev > max_dev:
            max_dev = dev
    
    threshold = tol * max(1.0, abs(m))
    if max_dev <= threshold:
        return True, float(e[0]), float(1.0 / m), int(nb)
    return False, 0.0, 1.0, nb


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


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _prebin_matrix_flat_njit(
    X_local: np.ndarray,           # (n, p) float64
    flat_edges: np.ndarray,        # concatenated edges (float64)
    edge_offsets: np.ndarray,      # start idx in flat_edges per feature (int32, len=p)
    edge_counts: np.ndarray,       # number of edges per feature (int32, len=p)
    uniform_flags: np.ndarray,     # uint8 - which features are uniform
    uniform_lo: np.ndarray,        # float64 - uniform feature low values
    uniform_invw: np.ndarray,      # float64 - uniform feature inverse widths
    uniform_nb: np.ndarray,        # int32 - uniform feature bin counts
    miss_id: int,
) -> np.ndarray:
    """
    Prebin all columns using flattened edges with uniform optimization.
    Returns int32 codes.
    """
    n, p = X_local.shape
    codes = np.empty((n, p), dtype=np.int32)
    
    for j in prange(p):
        off = edge_offsets[j]
        cnt = edge_counts[j]
        nb = cnt - 1
        is_uniform = uniform_flags[j] == 1
        
        if is_uniform and nb > 0:
            # Uniform binning: O(1) per sample
            lo = uniform_lo[j]
            invw = uniform_invw[j]
            for i in range(n):
                v = X_local[i, j]
                if not np.isfinite(v):
                    codes[i, j] = miss_id
                else:
                    k = int((v - lo) * invw)
                    if k < 0:
                        k = 0
                    elif k >= nb:
                        k = nb - 1
                    codes[i, j] = k
        else:
            # Binary search path
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
            if 0 <= b < n_bins_total:  # Combined bounds check
                gj[b] += g_node[i]
                hj[b] += h_node[i]
    return hg, hh


@njit(cache=True, nogil=True, fastmath=False, parallel=True)
def _fused_hist_uniform_mixed_njit(
    X: np.ndarray,               # (n,p) float64, contiguous
    flat_edges: np.ndarray,      # float64
    edge_offsets: np.ndarray,    # int32
    edge_counts: np.ndarray,     # int32
    uniform_flags: np.ndarray,   # uint8
    uniform_lo: np.ndarray,      # float64
    uniform_invw: np.ndarray,    # float64
    uniform_nb: np.ndarray,      # int32
    g: np.ndarray,               # (n,) float64
    h: np.ndarray,               # (n,) float64
    missing_id: int,
    n_bins_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fused kernel: directly builds (p, n_bins_total) histograms from X,g,h and edges.
    Uses O(1) arithmetic binning for uniform columns, binary search otherwise.
    """
    n, p = X.shape
    hg = np.zeros((p, n_bins_total), dtype=np.float64)
    hh = np.zeros((p, n_bins_total), dtype=np.float64)

    for j in prange(p):
        off = edge_offsets[j]
        cnt = edge_counts[j]
        nb = cnt - 1
        is_uniform = uniform_flags[j] == 1 and nb > 0 and uniform_nb[j] == nb

        gj = hg[j]
        hj = hh[j]

        if is_uniform:
            # uniform: k = floor((v - lo)*invw), clamped to [0, nb-1]
            lo = uniform_lo[j]
            invw = uniform_invw[j]
            for i in range(n):
                v = X[i, j]
                if not np.isfinite(v):
                    b = missing_id
                else:
                    k = int((v - lo) * invw)
                    if k < 0:
                        k = 0
                    elif k >= nb:
                        k = nb - 1
                    b = k
                gj[b] += g[i]
                hj[b] += h[i]
        else:
            # binary search
            e = flat_edges[off:off+cnt]
            for i in range(n):
                v = X[i, j]
                if not np.isfinite(v):
                    b = missing_id
                else:
                    if v < e[0]:
                        b = 0
                    elif v >= e[nb]:
                        b = nb - 1
                    else:
                        lo_i, hi_i = 0, nb
                        while hi_i - lo_i > 1:
                            mid = (lo_i + hi_i) // 2
                            if v >= e[mid]:
                                lo_i = mid
                            else:
                                hi_i = mid
                        b = lo_i
                gj[b] += g[i]
                hj[b] += h[i]
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
    if p == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    # Pre-allocate based on total size
    total = sum(2 if (e is None or e.size < 2) else e.size for e in edges_list)
    flat = np.empty(total, dtype=np.float64)
    offsets = np.empty(p, dtype=np.int32)
    counts = np.empty(p, dtype=np.int32)
    
    pos = 0
    for j, e in enumerate(edges_list):
        offsets[j] = pos
        if e is None or e.size < 2:
            flat[pos:pos + 2] = [0.0, 1.0]
            counts[j] = 2
            pos += 2
        else:
            m = e.size
            flat[pos:pos + m] = e
            counts[j] = m
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
    """
    mode: str
    feature_indices: np.ndarray
    layout: BinLayout
    edges_list: List[np.ndarray] = field(default_factory=list)
    edges: Dict[int, np.ndarray] = field(default_factory=dict)
    node_overrides: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    # caches
    cached_full_codes: Optional[np.ndarray] = None
    _row_prebin_cache: Dict[str, Tuple[np.ndarray, int]] = field(default_factory=dict)
    _hist_cache: Dict[Tuple[str, int, int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # flattened edges for fast kernels (global)
    _flat_edges: Optional[np.ndarray] = None
    _edge_offsets: Optional[np.ndarray] = None
    _edge_counts: Optional[np.ndarray] = None

    # uniform bin metadata per feature (global)
    _uniform_flags: Optional[np.ndarray] = None   # uint8
    _uniform_lo: Optional[np.ndarray] = None      # float64
    _uniform_invw: Optional[np.ndarray] = None    # float64
    _uniform_nb: Optional[np.ndarray] = None      # int32

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
    """

    def __init__(self, feature_indices: np.ndarray, feature_types: Optional[Dict[int, str]] = None):
        self.feature_indices = np.asarray(feature_indices, dtype=np.int32)
        self._specs: Dict[BinMode, BinSpec] = {}
        self._default_mode: Optional[BinMode] = None
        # Cache using array id and shape - avoids hashing issues
        self._prebin_cache: Dict[Tuple[int, Tuple[int, int], str], Tuple[np.ndarray, int]] = {}
        # global->local feature map aligned to feature_indices order
        self.feature_map: Dict[int, int] = {int(f): i for i, f in enumerate(self.feature_indices)}
        self.feature_types: Dict[int, str] = feature_types or {}
        self._prebin_cache: Dict[Tuple[int, Tuple[int, int], str], Tuple[np.ndarray, int]] = {}

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
        norm = [_strict_increasing(e) for e in edges_list_in_feature_index_order]
        spec.edges_list = norm
        spec.edges = {int(gfi): norm[i] for i, gfi in enumerate(self.feature_indices)}

        # reset per-mode caches/overrides
        spec.node_overrides = {}
        spec.cached_full_codes = None
        spec._row_prebin_cache = {}

        # Precompute flattened edges & per-feature metadata for fast kernels
        flat, offs, cnts = _flatten_edges_list(norm)
        spec._flat_edges = flat.astype(np.float64, copy=False)
        spec._edge_offsets = offs.astype(np.int32, copy=False)
        spec._edge_counts = cnts.astype(np.int32, copy=False)

        # Uniform per-feature flags/params (now using Numba)
        p = len(norm)
        uni_flags = np.zeros(p, dtype=np.uint8)
        uni_lo = np.zeros(p, dtype=np.float64)
        uni_invw = np.ones(p, dtype=np.float64)
        uni_nb = np.zeros(p, dtype=np.int32)
        for j, e in enumerate(norm):
            is_u, lo, invw, nb = _uniform_params_njit(e)
            uni_flags[j] = 1 if is_u else 0
            uni_lo[j] = lo
            uni_invw[j] = invw
            uni_nb[j] = nb
        spec._uniform_flags = uni_flags
        spec._uniform_lo = uni_lo
        spec._uniform_invw = uni_invw
        spec._uniform_nb = uni_nb

        self._specs[mode] = spec
        if self._default_mode is None:
            self._default_mode = mode

        # Clear any cached results for this mode
        self._prebin_cache.clear()

    def set_node_override(self, mode: BinMode, node_id: int, gfi: int, edges: np.ndarray) -> None:
        spec = self._specs.get(mode)
        if spec is None:
            raise KeyError(f"Mode '{mode}' not registered")
        arr = _strict_increasing(np.asarray(edges, dtype=np.float64))
        if arr.size < 2:
            raise ValueError("Override edges must have at least 2 values (>=1 bin).")
        nb = arr.size - 1
        # ✅ Enforce capacity defined by this mode's layout
        if nb > spec.layout.actual_max_bins:
            raise ValueError(
                f"Override adds {nb} finite bins but layout allows only {spec.layout.actual_max_bins}. "
                "Increase layout (re-register mode) or reduce override bins."
            )
        spec.node_overrides[(int(node_id), int(gfi))] = arr
        # Overrides make any global precomputed codes stale
        spec.cached_full_codes = None

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
        return any(nid_k == nid for (nid_k, _gfi) in spec.node_overrides.keys())

    def get_codes_view(self, mode: str) -> Optional[np.ndarray]:
        spec = self._specs.get(mode)
        if spec is None:
            return None
        return spec.cached_full_codes

    def clear_cached_codes(self, mode: Optional[str] = None) -> None:
        """Explicitly drop cached full codes for a mode (or all modes if None)."""
        if mode is None:
            for sp in self._specs.values():
                sp.cached_full_codes = None
            self._prebin_cache.clear()
        else:
            spec = self._specs.get(mode)
            if spec is not None:
                spec.cached_full_codes = None

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

        Xf = np.ascontiguousarray(X_local.astype(np.float64, copy=False))
        shape = Xf.shape

        # Improved caching strategy using array id
        use_cache = (node_id is None) and (not self._has_overrides(mode))
        cache_key_tuple = (id(X_local), shape, mode)
        if use_cache:
            cached = self._prebin_cache.get(cache_key_tuple)
            if cached is not None:
                return cached

        # Fast path: no overrides, use uniform optimization
        if node_id is None and not self._has_overrides(mode):
            codes_i32 = _prebin_matrix_flat_njit(
                Xf,
                spec._flat_edges,
                spec._edge_offsets,
                spec._edge_counts,
                spec._uniform_flags,
                spec._uniform_lo,
                spec._uniform_invw,
                spec._uniform_nb,
                layout.missing_bin_id,
            )
            codes = codes_i32.astype(layout.out_dtype, copy=False)
            if use_cache:
                self._prebin_cache[cache_key_tuple] = (codes, layout.missing_bin_id)
                spec.cached_full_codes = codes
            return codes, layout.missing_bin_id

        # Override-aware path: build per-node edges list then flatten once
        edges_list = spec.get_edges_list(node_id=node_id)
        flat_node, offs_node, cnts_node = _flatten_edges_list(edges_list)
        
        # Recompute uniform flags for this node's edges
        p = len(edges_list)
        uni_flags_node = np.zeros(p, dtype=np.uint8)
        uni_lo_node = np.zeros(p, dtype=np.float64)
        uni_invw_node = np.ones(p, dtype=np.float64)
        uni_nb_node = np.zeros(p, dtype=np.int32)
        for j, e in enumerate(edges_list):
            is_u, lo, invw, nb = _uniform_params_njit(e)
            uni_flags_node[j] = 1 if is_u else 0
            uni_lo_node[j] = lo
            uni_invw_node[j] = invw
            uni_nb_node[j] = nb
        
        codes_i32 = _prebin_matrix_flat_njit(
            Xf,
            flat_node,
            offs_node.astype(np.int32, copy=False),
            cnts_node.astype(np.int32, copy=False),
            uni_flags_node,
            uni_lo_node,
            uni_invw_node,
            uni_nb_node,
            layout.missing_bin_id,
        )
        out = codes_i32.astype(layout.out_dtype, copy=False)

        # cache row-level result if a key is provided (per-mode, per-node cache)
        if cache_key is not None:
            spec._row_prebin_cache[cache_key] = (out, layout.missing_bin_id)
            
        # update cached_full_codes
        if node_id is None and not self._has_overrides(mode):
            spec.cached_full_codes = out
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
        Numba-accelerated histogram builder for a node subset (from codes).
        Returns (hist_g, hist_h) with shape (p, n_bins_total).
        """
        codes_i32 = codes_sub.astype(np.int32, copy=False)
        g64 = g_node.astype(np.float64, copy=False)
        h64 = h_node.astype(np.float64, copy=False)
        return _build_histograms_from_codes_njit(codes_i32, g64, h64, int(n_bins_total))

    def build_histograms_from_X(
        self,
        X_sub: np.ndarray,       # (n_sub, p) columns aligned to feature_indices
        g_sub: np.ndarray,       # (n_sub,)
        h_sub: np.ndarray,       # (n_sub,)
        *,
        mode: BinMode,
        node_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Fused 'prebin+hist' builder: directly accumulates histograms from X,g,h.
        Uses uniform-bin O(1) binning where detected, binary-search otherwise.
        """
        spec = self._specs.get(mode)
        if spec is None:
            raise KeyError(f"Mode '{mode}' not registered")
        layout = spec.layout
        n_bins_total = int(layout.actual_max_bins) + 1

        Xf = np.ascontiguousarray(X_sub.astype(np.float64, copy=False))
        g64 = g_sub.astype(np.float64, copy=False)
        h64 = h_sub.astype(np.float64, copy=False)

        # When no overrides, re-use precomputed flattened arrays + uniform metadata
        if node_id is None and not self._has_overrides(mode):
            hg, hh = _fused_hist_uniform_mixed_njit(
                Xf,
                spec._flat_edges,
                spec._edge_offsets,
                spec._edge_counts,
                spec._uniform_flags,
                spec._uniform_lo,
                spec._uniform_invw,
                spec._uniform_nb,
                g64,
                h64,
                layout.missing_bin_id,
                n_bins_total,
            )
            return hg, hh, n_bins_total

        # With overrides: construct per-node flattened list with uniform detection
        edges_list = spec.get_edges_list(node_id=node_id)
        flat_node, offs_node, cnts_node = _flatten_edges_list(edges_list)
        
        # Recompute uniform metadata for node-specific edges
        p = len(edges_list)
        uni_flags_node = np.zeros(p, dtype=np.uint8)
        uni_lo_node = np.zeros(p, dtype=np.float64)
        uni_invw_node = np.ones(p, dtype=np.float64)
        uni_nb_node = np.zeros(p, dtype=np.int32)
        for j, e in enumerate(edges_list):
            is_u, lo, invw, nb = _uniform_params_njit(e)
            uni_flags_node[j] = 1 if is_u else 0
            uni_lo_node[j] = lo
            uni_invw_node[j] = invw
            uni_nb_node[j] = nb
        
        hg, hh = _fused_hist_uniform_mixed_njit(
            Xf,
            flat_node.astype(np.float64, copy=False),
            offs_node.astype(np.int32, copy=False),
            cnts_node.astype(np.int32, copy=False),
            uni_flags_node,
            uni_lo_node,
            uni_invw_node,
            uni_nb_node,
            g64, h64,
            layout.missing_bin_id,
            n_bins_total,
        )
        return hg, hh, n_bins_total

    def clone(self) -> BinRegistry:
        """Create a shallow copy of this registry (shares specs)."""
        new = BinRegistry(self.feature_indices.copy(), self.feature_types.copy())
        new._specs = self._specs.copy()
        new._default_mode = self._default_mode
        # Don't copy caches - they are instance-specific
        return new
