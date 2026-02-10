# conformal.py — Complete State-of-the-Art Conformal Prediction Engine (Corrected v4)
# =================================================================================
# Fixes (vs v3):
# - Sequential ACI update for rolling/agaci (true point-by-point alpha adaptation)
# - Batch vs sequential mode toggle for performance/correctness tradeoff
# - Improved documentation and type hints
#
# Previous fixes (v3):
# - CQR ("quantile") calibration corrected
# - Robust shape handling: supports preds/y as [N,H], [N,H,D], etc.
# - "local_window" enforced during calibrate()
# - ENBPI update can use ensemble (consistent with calibration)
# - Proper CV+ (Jackknife+) intervals when CV models are provided

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Optional, Literal, Callable, Sequence, Dict, Any, Tuple
import pickle


# ============================================================
# Helpers
# ============================================================

def _to_tensor(x, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _ensure_3d_pred(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize predictions to [N,H,D].
    Accepts:
      [N]         -> [N,1,1]
      [N,H]       -> [N,H,1]
      [N,H,D]     -> unchanged
    """
    if x.ndim == 1:
        return x[:, None, None]
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected preds with ndim in {{1,2,3}}, got {x.ndim} and shape {tuple(x.shape)}")


def _ensure_3d_y(y: torch.Tensor, like_pred: torch.Tensor) -> torch.Tensor:
    """
    Normalize y to [N,H,D] compatible with like_pred [N,H,D].
    Accepts:
      [N]         -> [N,1,1]
      [N,H]       -> [N,H,1]
      [N,H,D]     -> unchanged
    """
    y3 = _ensure_3d_pred(y)
    # Broadcast D if needed and safe:
    if y3.shape[2] == 1 and like_pred.shape[2] > 1:
        y3 = y3.expand(y3.shape[0], y3.shape[1], like_pred.shape[2])
    return y3


def _softmax_if_logits(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected [N,K] state/features, got {tuple(z.shape)}")
    row_sum = z.sum(dim=1, keepdim=True)
    if (z.min() < 0) or (z.max() > 1) or (torch.mean(torch.abs(row_sum - 1.0)) > 1e-2):
        return torch.softmax(z, dim=1)
    return z


def _conformal_quantile_level(n: int, alpha: float) -> float:
    """
    Finite-sample split conformal quantile level:
      q = min(1.0, ceil((n+1)*(1-alpha)) / n)
    """
    return min(1.0, float(np.ceil((n + 1) * (1 - alpha)) / n))


def _finite_q_level(n: int, q: float) -> float:
    """
    Finite-sample quantile level for general q in [0,1]:
      q_fs = min(1.0, ceil((n+1)*q) / n)
    """
    return min(1.0, float(np.ceil((n + 1) * q) / n))


def _exact_quantile_higher(values: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """
    Quantile using 'higher' interpolation via sorting:
    returns the smallest value v such that CDF(v) >= q.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    v = values
    if dim != 0:
        v = v.transpose(0, dim)
    n = v.shape[0]
    idx = int(np.ceil(q * n) - 1)
    idx = max(0, min(n - 1, idx))
    v_sorted, _ = torch.sort(v, dim=0)
    out = v_sorted[idx]
    return out


def weighted_quantile(values: torch.Tensor, q: float, weights: Optional[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Weighted quantile along `dim` (non-differentiable; suitable for conformal).
    If weights is None -> torch.quantile (linear interpolation).
    If weights provided -> weighted CDF with mid-point correction.
    """
    if weights is None:
        return torch.quantile(values, q, dim=dim)

    v = values.transpose(0, dim)
    w = weights.transpose(0, dim)
    K = v.shape[0]
    batch = int(np.prod(v.shape[1:])) if v.ndim > 1 else 1

    v2 = v.reshape(K, batch)
    w2 = w.reshape(K, batch).clamp_min(0.0)

    idx = torch.argsort(v2, dim=0)
    v_sorted = torch.gather(v2, 0, idx)
    w_sorted = torch.gather(w2, 0, idx)

    w_sum = w_sorted.sum(dim=0, keepdim=True).clamp_min(1e-12)
    w_norm = w_sorted / w_sum

    cdf = torch.cumsum(w_norm, dim=0) - 0.5 * w_norm

    ge = (cdf >= q)
    pos = ge.float().argmax(dim=0)
    pos = torch.where(ge.any(dim=0), pos, torch.full_like(pos, K - 1))

    out = v_sorted[pos, torch.arange(batch, device=v_sorted.device)]
    out = out.reshape(v.shape[1:])
    return out


def _pairwise_attention_weights(attn: nn.Module, z_test: torch.Tensor, z_cal: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    B, Fdim = z_test.shape
    T = z_cal.shape[0]
    zt = z_test[:, None, :].expand(B, T, Fdim)
    zc = z_cal[None, :, :].expand(B, T, Fdim)
    pair = torch.cat([zt, zc], dim=-1)
    logits = attn(pair).squeeze(-1)
    logits = logits / max(tau, 1e-8)
    return torch.softmax(logits, dim=1)


def _make_oob_mask_from_boot_indices(boot_indices: torch.Tensor, N: int) -> torch.Tensor:
    B = boot_indices.shape[0]
    oob = torch.ones((B, N), dtype=torch.bool, device=boot_indices.device)
    for b in range(B):
        oob[b, boot_indices[b]] = False
    return oob


def _rolling_clip(t: torch.Tensor, maxlen: int) -> torch.Tensor:
    if t.shape[0] <= maxlen:
        return t
    return t[-maxlen:]


# ============================================================
# Default Feature Extractor
# ============================================================

class DefaultFeatureExtractor(nn.Module):
    """
    Feature extractor for time series with temporal awareness.
    x: [N, seq_len, features] -> [N, feature_dim]
       otherwise flattened MLP -> [N, feature_dim]
    """
    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        seq_len: Optional[int] = None,
        n_features: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.n_features = n_features

        if seq_len is not None and n_features is not None:
            self.temporal = nn.Sequential(
                nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, feature_dim),
                nn.LayerNorm(feature_dim),
            )
            self.use_temporal = True
        else:
            self.use_temporal = False
            layers = []
            d_in = input_dim
            if depth < 2:
                depth = 2
            for _ in range(depth - 1):
                layers.extend([
                    nn.Linear(d_in, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                d_in = hidden
            layers.extend([
                nn.Linear(hidden, feature_dim),
                nn.LayerNorm(feature_dim),
            ])
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_temporal and x.ndim == 3:
            x = x.transpose(1, 2)       # [N, F, T]
            x = self.temporal(x)        # [N, hidden, 1]
            x = x.squeeze(-1)           # [N, hidden]
            return self.head(x)         # [N, feature_dim]
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


# ============================================================
# Main Engine
# ============================================================

class ConformalPredictionEngine:
    """
    State-of-the-art Conformal Prediction Engine supporting multiple methods.
    
    Methods:
        - split: Standard split conformal (global radius)
        - local: KNN-based local conformal
        - jackknife: CV+/Jackknife+ with leave-one-out residuals
        - quantile: Conformalized Quantile Regression (CQR)
        - tsp: Time-Series Partition with exponential weighting
        - rolling: Rolling conformal with ACI (Adaptive Conformal Inference)
        - agaci: Aggregated Adaptive CI with multiple learning rates
        - enbpi: Ensemble Batch Prediction Intervals
        - cptc: Conformal Prediction under Temporal Covariate shift
        - afocp: Attention-based Feature-weighted Online Conformal Prediction
    
    Key Features:
        - Sequential vs batch update modes for ACI methods
        - Proper finite-sample quantile correction
        - Multi-horizon support [N, H, D]
    """
    
    def __init__(
        self,
        method: Literal[
            "split", "local", "jackknife", "quantile",
            "tsp", "rolling", "agaci", "enbpi", "cptc", "afocp"
        ] = "split",
        quantile: float = 0.9,
        knn_k: int = 50,
        local_window: int = 5000,
        rolling_alpha: float = 0.05,
        aci_gamma: float = 0.01,
        agaci_gammas: Optional[Sequence[float]] = None,
        enbpi_B: int = 20,
        enbpi_window: int = 500,
        cptc_window: int = 500,
        cptc_tau: float = 1.0,
        cptc_hard_state_filter: bool = False,
        afocp_feature_dim: int = 128,
        afocp_attn_hidden: int = 64,
        afocp_window: int = 500,
        afocp_tau: float = 1.0,
        afocp_internal_feat_hidden: int = 256,
        afocp_internal_feat_depth: int = 3,
        afocp_internal_feat_dropout: float = 0.1,
        afocp_online_lr: float = 0.0,
        afocp_online_steps: int = 1,
        tsp_lambda: float = 0.01,
        tsp_window: int = 5000,
    ):
        self.method = method
        self.q = float(quantile)
        self.alpha = 1.0 - self.q

        self.knn_k = int(knn_k)
        self.local_window = int(local_window)

        self.rolling_alpha = float(rolling_alpha)
        self.aci_gamma = float(aci_gamma)

        self.enbpi_B = int(enbpi_B)
        self.enbpi_window = int(enbpi_window)

        self.cptc_window = int(cptc_window)
        self.cptc_tau = float(cptc_tau)
        self.cptc_hard_state_filter = bool(cptc_hard_state_filter)

        self.afocp_feature_dim = int(afocp_feature_dim)
        self.afocp_attn_hidden = int(afocp_attn_hidden)
        self.afocp_window = int(afocp_window)
        self.afocp_tau = float(afocp_tau)

        self.afocp_internal_feat_hidden = int(afocp_internal_feat_hidden)
        self.afocp_internal_feat_depth = int(afocp_internal_feat_depth)
        self.afocp_internal_feat_dropout = float(afocp_internal_feat_dropout)

        self.afocp_online_lr = float(afocp_online_lr)
        self.afocp_online_steps = int(afocp_online_steps)

        self.tsp_lambda = float(tsp_lambda)
        self.tsp_window = int(tsp_window)

        if agaci_gammas is None:
            agaci_gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        self.agaci_gammas = list(agaci_gammas)

        # State variables
        self.radii: Optional[np.ndarray] = None

        # local
        self.cal_X_feat: Optional[np.ndarray] = None
        self.local_residuals: Optional[np.ndarray] = None

        # rolling/agaci
        self.residuals_buffer: Optional[torch.Tensor] = None
        self.agaci_experts: Optional[list[dict]] = None

        # enbpi
        self.enbpi_oob_residuals: Optional[torch.Tensor] = None
        self._enbpi_member_models: Optional[Sequence[nn.Module]] = None

        # cptc
        self.cptc_state_model: Optional[Callable] = None
        self.cptc_states_buffer: Optional[torch.Tensor] = None
        self.cptc_residuals_buffer: Optional[torch.Tensor] = None

        # afocp
        self.afocp_feature_extractor: Optional[nn.Module] = None
        self.afocp_attn: Optional[nn.Module] = None
        self.afocp_feats_buffer: Optional[torch.Tensor] = None
        self.afocp_residuals_buffer: Optional[torch.Tensor] = None
        self._afocp_opt: Optional[torch.optim.Optimizer] = None
        self._afocp_input_dim: Optional[int] = None

        # cqr
        self.cqr_correction: Optional[np.ndarray] = None

        # jackknife+
        self.jackknife_residuals: Optional[torch.Tensor] = None
        self._jackknife_cv_models: Optional[Sequence[nn.Module]] = None
        self._jackknife_cv_indices: Optional[Sequence[np.ndarray]] = None

        # tsp
        self.tsp_residuals: Optional[torch.Tensor] = None
        self.tsp_weights: Optional[torch.Tensor] = None

    # ======================================================================
    # Batched forward pass
    # ======================================================================
    @torch.no_grad()
    def _forward(self, model: nn.Module, X: torch.Tensor, device: str = "cpu", batch_size: int = 256) -> torch.Tensor:
        model.eval()
        X = X.to(device)
        preds = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            out = model(xb)
            out = out[0] if isinstance(out, tuple) else out
            preds.append(out.detach().cpu())
        return torch.cat(preds, dim=0)

    # ======================================================================
    # Calibration
    # ======================================================================
    @torch.no_grad()
    def calibrate(
        self,
        model: nn.Module,
        X_cal: np.ndarray | torch.Tensor,
        y_cal: np.ndarray | torch.Tensor,
        device: str = "cpu",
        batch_size: int = 256,
        state_model: Optional[Callable] = None,
        feature_extractor: Optional[nn.Module] = None,
        enbpi_member_models: Optional[Sequence[nn.Module]] = None,
        enbpi_boot_indices: Optional[np.ndarray | torch.Tensor] = None,
        jackknife_cv_models: Optional[Sequence[nn.Module]] = None,
        jackknife_cv_indices: Optional[Sequence[np.ndarray]] = None,
    ):
        print(f"Calibrating ConformalPredictionEngine [{self.method}] (quantile={self.q})...")

        X_t = _to_tensor(X_cal, device=device)
        y_t = _to_tensor(y_cal, device=device)

        preds_raw = self._forward(model, X_t, device=device, batch_size=batch_size)

        # ---------------- quantile (CQR)
        if self.method == "quantile":
            if preds_raw.ndim < 2:
                raise ValueError("CQR requires model to output quantiles, got too few dims.")
            if preds_raw.ndim != 3 or preds_raw.shape[-1] != 2:
                raise ValueError(f"CQR expects preds shape [N,H,2], got {tuple(preds_raw.shape)}")

            lower_pred = preds_raw[..., 0]
            upper_pred = preds_raw[..., 1]

            y2 = y_t
            if y2.ndim == 3 and y2.shape[-1] == 1:
                y2 = y2[..., 0]
            if y2.ndim == 1:
                y2 = y2[:, None]
            if y2.ndim != 2:
                raise ValueError(f"CQR expects y as [N,H] (or [N,H,1]), got {tuple(y_t.shape)}")

            scores = torch.maximum(lower_pred - y2.cpu(), y2.cpu() - upper_pred)
            N = scores.shape[0]
            q_level = _conformal_quantile_level(N, self.alpha)
            self.cqr_correction = torch.quantile(scores, q_level, dim=0).cpu().numpy()
            self.radii = self.cqr_correction
            print(f"Calibration complete. Radii shape: {self.radii.shape}")
            return

        # For all other methods: normalize preds/y to [N,H,D]
        preds = _ensure_3d_pred(preds_raw)
        y3 = _ensure_3d_y(y_t.cpu(), preds)
        residuals = (preds - y3).abs()

        N, H, D = residuals.shape
        q_level = _conformal_quantile_level(N, self.alpha)

        # ---------------- split
        if self.method == "split":
            self.radii = torch.quantile(residuals, q_level, dim=0).numpy()

        # ---------------- local
        elif self.method == "local":
            cal_X_feat = X_t.reshape(X_t.shape[0], -1).detach().cpu().numpy()
            local_res = residuals.numpy()

            if cal_X_feat.shape[0] > self.local_window:
                cal_X_feat = cal_X_feat[-self.local_window:]
                local_res = local_res[-self.local_window:]
                N2 = local_res.shape[0]
                q_level2 = _conformal_quantile_level(N2, self.alpha)
                self.radii = np.quantile(local_res, q_level2, axis=0)
            else:
                self.radii = torch.quantile(residuals, q_level, dim=0).numpy()

            self.cal_X_feat = cal_X_feat
            self.local_residuals = local_res

        # ---------------- jackknife+ (CV+)
        elif self.method == "jackknife":
            if jackknife_cv_models is None or jackknife_cv_indices is None:
                warnings.warn(
                    "Jackknife+ (CV+) requires `jackknife_cv_models` and `jackknife_cv_indices`. "
                    "Falling back to split conformal."
                )
                self.radii = torch.quantile(residuals, q_level, dim=0).numpy()
                self.jackknife_residuals = residuals.clone()
            else:
                K = len(jackknife_cv_models)
                if K != len(jackknife_cv_indices):
                    raise ValueError("Number of CV models must match number of CV index arrays")

                loo_residuals = torch.zeros((N, H, D), dtype=torch.float32)
                covered = np.zeros(N, dtype=bool)

                for cv_model, val_idx in zip(jackknife_cv_models, jackknife_cv_indices):
                    val_idx = np.asarray(val_idx)
                    X_val = X_t[val_idx]
                    y_val = _ensure_3d_y(y_t[val_idx].cpu(), preds[val_idx])

                    cv_preds_raw = self._forward(cv_model, X_val, device=device, batch_size=batch_size)
                    cv_preds = _ensure_3d_pred(cv_preds_raw)
                    cv_res = (cv_preds - y_val).abs()
                    loo_residuals[val_idx] = cv_res
                    covered[val_idx] = True

                if not covered.all():
                    uncovered = ~covered
                    warnings.warn(f"{uncovered.sum()} samples not covered by CV folds; using base-model residuals there.")
                    loo_residuals[uncovered] = residuals[uncovered]

                self.jackknife_residuals = loo_residuals.clone()
                self._jackknife_cv_models = list(jackknife_cv_models)
                self._jackknife_cv_indices = list(jackknife_cv_indices)

                self.radii = torch.quantile(loo_residuals, q_level, dim=0).numpy()

        # ---------------- tsp
        elif self.method == "tsp":
            time_indices = torch.arange(N, dtype=torch.float32)
            weights = torch.exp(self.tsp_lambda * (time_indices - (N - 1)))
            weights = weights / weights.sum()

            self.tsp_residuals = residuals.clone()
            self.tsp_weights = weights.clone()

            radii = torch.zeros(H, D)
            for h in range(H):
                for d in range(D):
                    vals = residuals[:, h, d]
                    radii[h, d] = weighted_quantile(vals, q_level, weights, dim=0)

            self.radii = radii.numpy()

        # ---------------- rolling/agaci
        elif self.method in ["rolling", "agaci"]:
            self.residuals_buffer = residuals.clone()
            self.radii = torch.quantile(residuals, q_level, dim=0).numpy()
            if self.method == "agaci":
                self.agaci_experts = [
                    {"gamma": g, "alpha": self.alpha, "buffer": residuals.clone()}
                    for g in self.agaci_gammas
                ]

        # ---------------- enbpi
        elif self.method == "enbpi":
            if enbpi_member_models is None:
                warnings.warn(
                    "EnbPI requires `enbpi_member_models` + `enbpi_boot_indices` for true OOB residuals. "
                    "Falling back to approximate bootstrap."
                )
                boot_res = []
                for _ in range(self.enbpi_B):
                    idx = np.random.choice(N, N, replace=True)
                    boot_pred = preds[idx].mean(dim=0, keepdim=True).expand_as(preds)
                    boot_res.append((preds - boot_pred).abs())
                flat = torch.stack(boot_res, dim=0).reshape(self.enbpi_B * N, H, D)
                q_level2 = _conformal_quantile_level(flat.shape[0], self.alpha)
                self.radii = torch.quantile(flat, q_level2, dim=0).numpy()
                self.enbpi_oob_residuals = residuals[-min(self.enbpi_window, N):].clone()
            else:
                if enbpi_boot_indices is None:
                    raise ValueError("EnbPI with member models requires `enbpi_boot_indices`.")

                self._enbpi_member_models = list(enbpi_member_models)

                member_preds = []
                for m in enbpi_member_models:
                    p_raw = self._forward(m, X_t, device=device, batch_size=batch_size)
                    member_preds.append(_ensure_3d_pred(p_raw))
                member_preds = torch.stack(member_preds, dim=0)

                boot_idx = _to_tensor(enbpi_boot_indices, device="cpu").long()
                oob_mask = _make_oob_mask_from_boot_indices(boot_idx, N)
                mask = oob_mask[:, :, None, None]

                denom = mask.sum(dim=0).clamp_min(1.0)
                oob_mean = (member_preds * mask).sum(dim=0) / denom
                full_mean = member_preds.mean(dim=0)
                denom0 = (mask.sum(dim=0) < 0.5).expand_as(oob_mean)
                oob_mean = torch.where(denom0, full_mean, oob_mean)

                y_cpu = _ensure_3d_y(y_t.cpu(), oob_mean)
                oob_res = (oob_mean - y_cpu).abs()
                self.enbpi_oob_residuals = oob_res[-min(self.enbpi_window, N):].clone()

                q_level2 = _conformal_quantile_level(self.enbpi_oob_residuals.shape[0], self.alpha)
                self.radii = torch.quantile(self.enbpi_oob_residuals, q_level2, dim=0).numpy()

        # ---------------- cptc
        elif self.method == "cptc":
            if state_model is None:
                raise ValueError("CPTC requires a state_model.")
            self.cptc_state_model = state_model

            z = state_model(X_t.to(device))
            if isinstance(z, tuple):
                z = z[0]
            z = z.detach().cpu()
            z = _softmax_if_logits(z) if z.shape[1] <= 256 else z

            take = min(self.cptc_window, N)
            self.cptc_states_buffer = z[-take:].clone()
            self.cptc_residuals_buffer = residuals[-take:].clone()
            q_level2 = _conformal_quantile_level(take, self.alpha)
            self.radii = torch.quantile(self.cptc_residuals_buffer, q_level2, dim=0).numpy()

        # ---------------- afocp
        elif self.method == "afocp":
            if feature_extractor is None:
                input_dim = int(X_t.reshape(X_t.shape[0], -1).shape[1])
                self._afocp_input_dim = input_dim

                seq_len = X_t.shape[1] if X_t.ndim == 3 else None
                n_features = X_t.shape[2] if X_t.ndim == 3 else None

                feature_extractor = DefaultFeatureExtractor(
                    input_dim=input_dim,
                    feature_dim=self.afocp_feature_dim,
                    hidden=self.afocp_internal_feat_hidden,
                    depth=self.afocp_internal_feat_depth,
                    dropout=self.afocp_internal_feat_dropout,
                    seq_len=seq_len,
                    n_features=n_features,
                )
                warnings.warn(
                    "AFOCP: Using internal DefaultFeatureExtractor. "
                    "For best results, pass a pretrained feature_extractor."
                )

            self.afocp_feature_extractor = feature_extractor.to(device).eval()

            self.afocp_attn = nn.Sequential(
                nn.Linear(self.afocp_feature_dim * 2, self.afocp_attn_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.afocp_attn_hidden, 1),
            ).to(device).eval()

            z = self.afocp_feature_extractor(X_t.to(device))
            if isinstance(z, tuple):
                z = z[0]
            z = z.detach().cpu()

            take = min(self.afocp_window, N)
            self.afocp_feats_buffer = z[-take:].clone()
            self.afocp_residuals_buffer = residuals[-take:].clone()
            q_level2 = _conformal_quantile_level(take, self.alpha)
            self.radii = torch.quantile(self.afocp_residuals_buffer, q_level2, dim=0).numpy()

            if self.afocp_online_lr > 0.0:
                params = list(self.afocp_attn.parameters())
                params += list(self.afocp_feature_extractor.parameters())
                self._afocp_opt = torch.optim.Adam(params, lr=self.afocp_online_lr)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        print(f"Calibration complete. Radii shape: {self.radii.shape}")

    # ======================================================================
    # Prediction
    # ======================================================================
    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        X: np.ndarray | torch.Tensor,
        device: str = "cpu",
        batch_size: int = 256,
        jackknife_cv_models: Optional[Sequence[nn.Module]] = None,
        jackknife_use_stored: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.radii is None:
            raise RuntimeError("Engine must be calibrated before calling predict().")

        X_t = _to_tensor(X, device=device)
        preds_raw = self._forward(model, X_t, device=device, batch_size=batch_size)

        # CQR special case
        if self.method == "quantile":
            if preds_raw.ndim != 3 or preds_raw.shape[-1] != 2:
                raise ValueError(f"CQR expects preds shape [N,H,2], got {tuple(preds_raw.shape)}")
            lower_pred = preds_raw[..., 0].numpy()
            upper_pred = preds_raw[..., 1].numpy()
            correction = self.cqr_correction
            if correction is None:
                raise RuntimeError("CQR correction missing; calibrate() must be called first.")
            lower = lower_pred - correction[None, :]
            upper = upper_pred + correction[None, :]
            preds = (lower_pred + upper_pred) / 2.0
            preds = preds[:, :, None]
            lower = lower[:, :, None]
            upper = upper[:, :, None]
            return preds, lower, upper

        preds = _ensure_3d_pred(preds_raw)
        N, H, D = preds.shape

        # Proper Jackknife+ / CV+ intervals
        if self.method == "jackknife":
            if self.jackknife_residuals is None:
                raise RuntimeError("Jackknife residuals missing; calibrate() must be called first.")

            cv_models = None
            if jackknife_cv_models is not None:
                cv_models = jackknife_cv_models
            elif jackknife_use_stored and self._jackknife_cv_models is not None:
                cv_models = self._jackknife_cv_models

            if cv_models is None:
                warnings.warn(
                    "Jackknife+ requires CV models at predict time. "
                    "Falling back to split-style radii."
                )
                radii = np.broadcast_to(self.radii[None, :, :], (N, H, D))
                p = preds.numpy()
                return p, p - radii, p + radii

            fold_preds = []
            for m in cv_models:
                pr = self._forward(m, X_t, device=device, batch_size=batch_size)
                fold_preds.append(_ensure_3d_pred(pr))
            fold_preds = torch.stack(fold_preds, dim=0)
            pred_cv_mean = fold_preds.mean(dim=0)

            r = self.jackknife_residuals
            Ncal = r.shape[0]

            base = pred_cv_mean[:, None, :, :].expand(N, Ncal, H, D)
            r_exp = r[None, :, :, :].expand(N, Ncal, H, D)

            lower_set = base - r_exp
            upper_set = base + r_exp

            q_low = _finite_q_level(Ncal, self.alpha)
            q_high = _finite_q_level(Ncal, 1.0 - self.alpha)

            lower_t = _exact_quantile_higher(lower_set, q_low, dim=1)
            upper_t = _exact_quantile_higher(upper_set, q_high, dim=1)

            p = preds.numpy()
            return p, lower_t.numpy(), upper_t.numpy()

        # Per-sample adaptive radii
        if self.method == "local":
            radii = self._compute_local_radii(X_t.cpu())

        elif self.method == "cptc":
            radii = self._compute_cptc_radii(X_t, device)

        elif self.method == "afocp":
            radii = self._compute_afocp_radii(X_t, device)

        else:
            radii = np.broadcast_to(self.radii[None, :, :], (N, H, D))

        p = preds.numpy()
        lower = p - radii
        upper = p + radii
        return p, lower, upper

    def _compute_local_radii(self, X: torch.Tensor) -> np.ndarray:
        if self.local_residuals is None or self.cal_X_feat is None:
            raise RuntimeError("Local method not calibrated.")

        X_flat = X.reshape(len(X), -1).numpy()
        Nq = X_flat.shape[0]
        N_cal = self.cal_X_feat.shape[0]
        H, D = self.local_residuals.shape[1], self.local_residuals.shape[2]
        radii = np.zeros((Nq, H, D), dtype=np.float32)

        k = min(self.knn_k, N_cal)

        for i in range(Nq):
            dists = np.linalg.norm(self.cal_X_feat - X_flat[i], axis=1)
            knn_idx = np.argsort(dists)[:k]
            local_res = self.local_residuals[knn_idx]
            q_level = _conformal_quantile_level(k, self.alpha)
            radii[i] = np.quantile(local_res, q_level, axis=0)

        return radii

    def _compute_cptc_radii(self, X_t: torch.Tensor, device: str) -> np.ndarray:
        if self.cptc_state_model is None or self.cptc_states_buffer is None or self.cptc_residuals_buffer is None:
            raise RuntimeError("CPTC not calibrated (missing buffers/state_model).")

        zt = self.cptc_state_model(X_t.to(device))
        if isinstance(zt, tuple):
            zt = zt[0]
        zt = zt.detach().cpu()
        zt = _softmax_if_logits(zt) if zt.shape[1] <= 256 else zt

        cal_states = self.cptc_states_buffer
        cal_res = self.cptc_residuals_buffer
        N, T = zt.shape[0], cal_res.shape[0]
        q_level = _conformal_quantile_level(T, self.alpha)

        if self.cptc_hard_state_filter and zt.shape[1] <= 256:
            cs = cal_states.argmax(dim=1)
            ts = zt.argmax(dim=1)
            mask = (ts[:, None] == cs[None, :])
            any_mask = mask.any(dim=1, keepdim=True)
            mask = torch.where(any_mask, mask, torch.ones_like(mask))
            w = mask.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            sim = (zt[:, None, :] * cal_states[None, :, :]).sum(dim=-1)
            sim = sim / max(self.cptc_tau, 1e-8)
            w = torch.softmax(sim, dim=1)

        vals = cal_res.unsqueeze(0).expand(N, T, -1, -1)
        ww = w[:, :, None, None].expand_as(vals)
        radii_t = weighted_quantile(vals, q=q_level, weights=ww, dim=1)
        return radii_t.numpy()

    def _compute_afocp_radii(self, X_t: torch.Tensor, device: str) -> np.ndarray:
        if self.afocp_feature_extractor is None or self.afocp_attn is None:
            raise RuntimeError("AFOCP not calibrated (missing feature_extractor/attn).")
        if self.afocp_feats_buffer is None or self.afocp_residuals_buffer is None:
            raise RuntimeError("AFOCP not calibrated (missing buffers).")

        zt = self.afocp_feature_extractor(X_t.to(device))
        if isinstance(zt, tuple):
            zt = zt[0]
        zt = zt.detach().cpu()

        cal_z = self.afocp_feats_buffer
        cal_res = self.afocp_residuals_buffer
        N, T = zt.shape[0], cal_res.shape[0]
        q_level = _conformal_quantile_level(T, self.alpha)

        self.afocp_attn.eval()
        w = _pairwise_attention_weights(
            self.afocp_attn.to(device), zt.to(device), cal_z.to(device), tau=self.afocp_tau
        ).cpu()

        vals = cal_res.unsqueeze(0).expand(N, T, -1, -1)
        ww = w[:, :, None, None].expand_as(vals)
        radii_t = weighted_quantile(vals, q=q_level, weights=ww, dim=1)
        return radii_t.numpy()

    # ======================================================================
    # Online Update
    # ======================================================================
    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        X_new: np.ndarray | torch.Tensor,
        y_new: np.ndarray | torch.Tensor,
        device: str = "cpu",
        batch_size: int = 256,
        sequential: Optional[bool] = None,
        enbpi_member_models: Optional[Sequence[nn.Module]] = None,
        state_model: Optional[Callable] = None,
        feature_extractor: Optional[nn.Module] = None,
    ):
        """
        Online update for adaptive conformal methods.
        
        Args:
            model: The forecasting model
            X_new: New input features [B, ...]
            y_new: New ground truth labels [B, H] or [B, H, D]
            device: Computation device
            batch_size: Batch size for model inference
            sequential: If True, update point-by-point (required for exact ACI).
                       If None, auto-enables for rolling/agaci methods.
                       If False, use batch update (faster but approximate for ACI).
            enbpi_member_models: Ensemble models for EnbPI-consistent update
            state_model: State model for CPTC (uses stored if None)
            feature_extractor: Feature extractor for AFOCP (uses stored if None)
        """
        supported = ["rolling", "agaci", "enbpi", "cptc", "afocp", "local", "tsp"]
        if self.method not in supported:
            warnings.warn(f"Online update not supported for '{self.method}'.")
            return

        if self.radii is None:
            raise RuntimeError("Engine must be calibrated before calling update().")

        # Auto-enable sequential mode for ACI methods
        if sequential is None:
            sequential = self.method in ("rolling", "agaci")

        X_t = _to_tensor(X_new, device=device)
        y_t = _to_tensor(y_new, device=device)

        preds_raw = self._forward(model, X_t, device=device, batch_size=batch_size)
        preds = _ensure_3d_pred(preds_raw)
        y3 = _ensure_3d_y(y_t.cpu(), preds)
        new_residuals = (preds - y3).abs()

        B, H, D = new_residuals.shape

        # Sequential ACI update for rolling/agaci
        if sequential and self.method in ("rolling", "agaci"):
            for i in range(B):
                self._update_single_aci(
                    model=model,
                    X_single=X_t[i:i+1],
                    y_single=y3[i:i+1],
                    residual_single=new_residuals[i:i+1],
                    device=device,
                    batch_size=1,
                )
            return

        # Batch update for non-ACI methods or when sequential=False
        self._update_batch(
            model=model,
            X_t=X_t,
            y3=y3,
            new_residuals=new_residuals,
            device=device,
            batch_size=batch_size,
            enbpi_member_models=enbpi_member_models,
            state_model=state_model,
            feature_extractor=feature_extractor,
        )

    def _update_single_aci(
        self,
        model: nn.Module,
        X_single: torch.Tensor,
        y_single: torch.Tensor,
        residual_single: torch.Tensor,
        device: str,
        batch_size: int,
    ):
        """
        Single-point ACI update for rolling/agaci methods.
        
        Implements the true sequential ACI update:
            α_{t+1} = α_t + γ * (α_target - err_t)
        
        where α_target = 1 - q is the fixed target miscoverage rate.
        
        When we miss (err_t=1), α decreases, which increases the quantile level (1-α),
        giving larger radii and wider intervals to reduce future misses.
        
        Each sample uses its own α to compute intervals, then updates α.
        """
        # Get interval with CURRENT alpha
        _, lower, upper = self.predict(model, X_single.cpu().numpy(), device, batch_size)
        y_np = y_single.numpy()
        
        # Check coverage (any dimension/horizon miss counts as a miss)
        missed = float(np.any((y_np < lower) | (y_np > upper)))
        
        # Fixed target miscoverage rate (e.g., 0.1 for 90% coverage)
        target_alpha = 1.0 - self.q
        
        if self.method == "rolling":
            # Correct ACI update: α += γ * (target - err)
            # When miss (err=1), α decreases → (1-α) increases → larger quantile → wider intervals
            self.alpha += self.aci_gamma * (target_alpha - missed)
            self.alpha = float(np.clip(self.alpha, 0.01, 0.99))
            
            # Update buffer
            if self.residuals_buffer is None:
                self.residuals_buffer = residual_single.clone()
            else:
                self.residuals_buffer = torch.cat([self.residuals_buffer, residual_single], dim=0)
            self.residuals_buffer = _rolling_clip(self.residuals_buffer, maxlen=5000)
            
            # Recompute radii with updated alpha
            q_level = _conformal_quantile_level(self.residuals_buffer.shape[0], self.alpha)
            self.radii = torch.quantile(self.residuals_buffer, q_level, dim=0).numpy()
            
        elif self.method == "agaci":
            if self.agaci_experts is None:
                raise RuntimeError("AGACI not calibrated.")
            
            # Update each expert independently
            for expert in self.agaci_experts:
                # Correct ACI update for each expert
                expert["alpha"] += expert["gamma"] * (target_alpha - missed)
                expert["alpha"] = float(np.clip(expert["alpha"], 0.01, 0.99))
                
                # Update buffer
                expert["buffer"] = torch.cat([expert["buffer"], residual_single], dim=0)
                expert["buffer"] = _rolling_clip(expert["buffer"], maxlen=5000)
            
            # Aggregate expert radii
            expert_radii = []
            for e in self.agaci_experts:
                q_level_e = _conformal_quantile_level(e["buffer"].shape[0], e["alpha"])
                expert_radii.append(torch.quantile(e["buffer"], q_level_e, dim=0).numpy())
            self.radii = np.mean(expert_radii, axis=0)

    def _update_batch(
        self,
        model: nn.Module,
        X_t: torch.Tensor,
        y3: torch.Tensor,
        new_residuals: torch.Tensor,
        device: str,
        batch_size: int,
        enbpi_member_models: Optional[Sequence[nn.Module]] = None,
        state_model: Optional[Callable] = None,
        feature_extractor: Optional[Callable] = None,
    ):
        """
        Batch update for non-ACI methods or when sequential=False.
        """
        B, H, D = new_residuals.shape

        # Coverage tracking for batch ACI (approximate)
        if self.method in ("rolling", "agaci"):
            _, current_lower, current_upper = self.predict(model, X_t.cpu().numpy(), device=device, batch_size=batch_size)
            y_np = y3.numpy()
            miscoverage_rate = float(np.mean((y_np < current_lower) | (y_np > current_upper)))
            # Use FIXED target miscoverage, not the adaptive alpha
            target_miscoverage = 1.0 - self.q

        if self.method == "local":
            if self.cal_X_feat is None or self.local_residuals is None:
                raise RuntimeError("Local method not calibrated.")
            new_feat = X_t.reshape(B, -1).cpu().numpy()
            self.cal_X_feat = np.concatenate([self.cal_X_feat, new_feat], axis=0)
            self.local_residuals = np.concatenate([self.local_residuals, new_residuals.numpy()], axis=0)

            if len(self.cal_X_feat) > self.local_window:
                self.cal_X_feat = self.cal_X_feat[-self.local_window:]
                self.local_residuals = self.local_residuals[-self.local_window:]

            q_level = _conformal_quantile_level(len(self.local_residuals), self.alpha)
            self.radii = np.quantile(self.local_residuals, q_level, axis=0)

        elif self.method == "tsp":
            if self.tsp_residuals is None:
                self.tsp_residuals = new_residuals.clone()
            else:
                self.tsp_residuals = torch.cat([self.tsp_residuals, new_residuals], dim=0)
            self.tsp_residuals = _rolling_clip(self.tsp_residuals, self.tsp_window)

            N = self.tsp_residuals.shape[0]
            time_indices = torch.arange(N, dtype=torch.float32)
            self.tsp_weights = torch.exp(self.tsp_lambda * (time_indices - (N - 1)))
            self.tsp_weights = self.tsp_weights / self.tsp_weights.sum()

            q_level = _conformal_quantile_level(N, self.alpha)
            radii = torch.zeros(H, D)
            for h in range(H):
                for d in range(D):
                    vals = self.tsp_residuals[:, h, d]
                    radii[h, d] = weighted_quantile(vals, q_level, self.tsp_weights, dim=0)
            self.radii = radii.numpy()

        elif self.method == "rolling":
            # Batch ACI (approximate): α += γ * (target - miscoverage)
            # When miscoverage > target, α decreases → wider intervals
            self.alpha += self.aci_gamma * (target_miscoverage - miscoverage_rate)
            self.alpha = float(np.clip(self.alpha, 0.01, 0.99))

            if self.residuals_buffer is None:
                self.residuals_buffer = new_residuals.clone()
            else:
                self.residuals_buffer = torch.cat([self.residuals_buffer, new_residuals], dim=0)
            self.residuals_buffer = _rolling_clip(self.residuals_buffer, maxlen=5000)

            q_level = _conformal_quantile_level(self.residuals_buffer.shape[0], self.alpha)
            self.radii = torch.quantile(self.residuals_buffer, q_level, dim=0).numpy()

        elif self.method == "agaci":
            # Batch AgACI (approximate)
            if self.agaci_experts is None:
                raise RuntimeError("AGACI not calibrated.")
            for expert in self.agaci_experts:
                expert["buffer"] = torch.cat([expert["buffer"], new_residuals], dim=0)
                expert["buffer"] = _rolling_clip(expert["buffer"], maxlen=5000)
                # Correct update: α += γ * (target - miscoverage)
                expert["alpha"] += expert["gamma"] * (target_miscoverage - miscoverage_rate)
                expert["alpha"] = float(np.clip(expert["alpha"], 0.01, 0.99))

            expert_radii = []
            for e in self.agaci_experts:
                q_level_e = _conformal_quantile_level(e["buffer"].shape[0], e["alpha"])
                expert_radii.append(torch.quantile(e["buffer"], q_level_e, dim=0).numpy())
            self.radii = np.mean(expert_radii, axis=0)

        elif self.method == "enbpi":
            members = enbpi_member_models or self._enbpi_member_models
            if members is None:
                warnings.warn(
                    "EnbPI update without member models is inconsistent with OOB calibration; "
                    "updating using base-model residuals as an approximation."
                )
                oob_like = new_residuals
            else:
                member_preds = []
                for m in members:
                    pr = self._forward(m, X_t, device=device, batch_size=batch_size)
                    member_preds.append(_ensure_3d_pred(pr))
                member_preds = torch.stack(member_preds, dim=0)
                mean_pred = member_preds.mean(dim=0)
                oob_like = (mean_pred - y3).abs()

            if self.enbpi_oob_residuals is None:
                self.enbpi_oob_residuals = oob_like.clone()
            else:
                self.enbpi_oob_residuals = torch.cat([self.enbpi_oob_residuals, oob_like], dim=0)
            self.enbpi_oob_residuals = _rolling_clip(self.enbpi_oob_residuals, self.enbpi_window)

            q_level = _conformal_quantile_level(self.enbpi_oob_residuals.shape[0], self.alpha)
            self.radii = torch.quantile(self.enbpi_oob_residuals, q_level, dim=0).numpy()

        elif self.method == "cptc":
            sm = state_model or self.cptc_state_model
            if sm is None:
                raise RuntimeError("CPTC not calibrated (missing state_model).")
            
            z = sm(X_t.to(device))
            if isinstance(z, tuple):
                z = z[0]
            z = z.detach().cpu()
            z = _softmax_if_logits(z) if z.shape[1] <= 256 else z

            if self.cptc_states_buffer is None or self.cptc_residuals_buffer is None:
                raise RuntimeError("CPTC not calibrated (missing buffers).")

            self.cptc_states_buffer = torch.cat([self.cptc_states_buffer, z], dim=0)
            self.cptc_residuals_buffer = torch.cat([self.cptc_residuals_buffer, new_residuals], dim=0)
            self.cptc_states_buffer = _rolling_clip(self.cptc_states_buffer, self.cptc_window)
            self.cptc_residuals_buffer = _rolling_clip(self.cptc_residuals_buffer, self.cptc_window)

            q_level = _conformal_quantile_level(self.cptc_residuals_buffer.shape[0], self.alpha)
            self.radii = torch.quantile(self.cptc_residuals_buffer, q_level, dim=0).numpy()

        elif self.method == "afocp":
            fe = feature_extractor or self.afocp_feature_extractor
            if fe is None:
                raise RuntimeError("AFOCP not calibrated (missing feature extractor).")
            
            z = fe(X_t.to(device))
            if isinstance(z, tuple):
                z = z[0]
            z = z.detach().cpu()

            if self.afocp_feats_buffer is None or self.afocp_residuals_buffer is None:
                raise RuntimeError("AFOCP not calibrated (missing buffers).")

            self.afocp_feats_buffer = torch.cat([self.afocp_feats_buffer, z], dim=0)
            self.afocp_residuals_buffer = torch.cat([self.afocp_residuals_buffer, new_residuals], dim=0)
            self.afocp_feats_buffer = _rolling_clip(self.afocp_feats_buffer, self.afocp_window)
            self.afocp_residuals_buffer = _rolling_clip(self.afocp_residuals_buffer, self.afocp_window)

            q_level = _conformal_quantile_level(self.afocp_residuals_buffer.shape[0], self.alpha)
            self.radii = torch.quantile(self.afocp_residuals_buffer, q_level, dim=0).numpy()

            if self._afocp_opt is not None and self.afocp_online_steps > 0:
                self._afocp_online_train_step(X_t, y3, new_residuals, device)

    def _afocp_online_train_step(
        self,
        X_batch: torch.Tensor,
        y_batch_3d: torch.Tensor,
        actual_residuals: torch.Tensor,
        device: str
    ):
        B = X_batch.shape[0]
        if self.afocp_feats_buffer is None or self.afocp_residuals_buffer is None:
            return
        if self.afocp_feats_buffer.shape[0] <= B + 8:
            return

        cal_z = self.afocp_feats_buffer[:-B].to(device)
        cal_res = self.afocp_residuals_buffer[:-B].to(device)
        if cal_z.shape[0] < 8:
            return

        self.afocp_attn.train()
        self.afocp_feature_extractor.train()

        actual_res = actual_residuals.to(device)
        q_level = 1.0 - self.alpha

        with torch.enable_grad():
            for _ in range(self.afocp_online_steps):
                self._afocp_opt.zero_grad(set_to_none=True)
                zt = self.afocp_feature_extractor(X_batch.to(device))
                w = _pairwise_attention_weights(self.afocp_attn, zt, cal_z, tau=self.afocp_tau)

                vals = cal_res.unsqueeze(0).expand(B, -1, -1, -1)
                ww = w[:, :, None, None].expand_as(vals)
                r = weighted_quantile(vals, q=q_level, weights=ww, dim=1)

                diff = actual_res - r
                loss = torch.where(diff > 0, q_level * diff, (1 - q_level) * (-diff)).mean()
                loss.backward()
                self._afocp_opt.step()

        self.afocp_attn.eval()
        self.afocp_feature_extractor.eval()

    # ======================================================================
    # Utility Methods
    # ======================================================================
    def compute_coverage(
        self,
        model: nn.Module,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> Dict[str, float]:
        preds, lower, upper = self.predict(model, X, device, batch_size)
        y_t = _to_tensor(y, device="cpu")
        y3 = _ensure_3d_y(y_t, torch.from_numpy(preds))
        y_np = y3.numpy()

        covered = (y_np >= lower) & (y_np <= upper)
        widths = upper - lower

        return {
            "coverage": float(covered.mean()),
            "target_coverage": float(self.q),
            "coverage_gap": float(covered.mean() - self.q),
            "mean_width": float(widths.mean()),
            "std_width": float(widths.std()),
            "min_width": float(widths.min()),
            "max_width": float(widths.max()),
            "width_cv": float(widths.std() / (widths.mean() + 1e-8)),
        }

    def get_interval_widths(
        self,
        model: nn.Module,
        X: np.ndarray | torch.Tensor,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> np.ndarray:
        _, lower, upper = self.predict(model, X, device, batch_size)
        return upper - lower

    def get_current_alpha(self) -> float:
        """Get the current miscoverage level (useful for ACI methods)."""
        return self.alpha

    def get_expert_alphas(self) -> Optional[Dict[float, float]]:
        """Get current alpha values for each AgACI expert."""
        if self.agaci_experts is None:
            return None
        return {e["gamma"]: e["alpha"] for e in self.agaci_experts}

    def save(self, path: str):
        state = {
            "method": self.method,
            "q": self.q,
            "alpha": self.alpha,
            "knn_k": self.knn_k,
            "local_window": self.local_window,
            "aci_gamma": self.aci_gamma,
            "radii": self.radii,
            "cal_X_feat": self.cal_X_feat,
            "local_residuals": self.local_residuals,
            "cqr_correction": self.cqr_correction,
        }

        if self.residuals_buffer is not None:
            state["residuals_buffer"] = self.residuals_buffer.numpy()
        if self.enbpi_oob_residuals is not None:
            state["enbpi_oob_residuals"] = self.enbpi_oob_residuals.numpy()
        if self.cptc_states_buffer is not None:
            state["cptc_states_buffer"] = self.cptc_states_buffer.numpy()
        if self.cptc_residuals_buffer is not None:
            state["cptc_residuals_buffer"] = self.cptc_residuals_buffer.numpy()
        if self.afocp_feats_buffer is not None:
            state["afocp_feats_buffer"] = self.afocp_feats_buffer.numpy()
        if self.afocp_residuals_buffer is not None:
            state["afocp_residuals_buffer"] = self.afocp_residuals_buffer.numpy()
        if self.tsp_residuals is not None:
            state["tsp_residuals"] = self.tsp_residuals.numpy()
        if self.tsp_weights is not None:
            state["tsp_weights"] = self.tsp_weights.numpy()
        if self.agaci_experts is not None:
            state["agaci_experts"] = [
                {"gamma": e["gamma"], "alpha": e["alpha"], "buffer": e["buffer"].numpy()}
                for e in self.agaci_experts
            ]
        if self.jackknife_residuals is not None:
            state["jackknife_residuals"] = self.jackknife_residuals.numpy()

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.method = state["method"]
        self.q = state["q"]
        self.alpha = state["alpha"]
        self.knn_k = state.get("knn_k", self.knn_k)
        self.local_window = state.get("local_window", self.local_window)
        self.aci_gamma = state.get("aci_gamma", self.aci_gamma)

        self.radii = state["radii"]
        self.cal_X_feat = state.get("cal_X_feat")
        self.local_residuals = state.get("local_residuals")
        self.cqr_correction = state.get("cqr_correction")

        if "residuals_buffer" in state:
            self.residuals_buffer = torch.from_numpy(state["residuals_buffer"])
        if "enbpi_oob_residuals" in state:
            self.enbpi_oob_residuals = torch.from_numpy(state["enbpi_oob_residuals"])
        if "cptc_states_buffer" in state:
            self.cptc_states_buffer = torch.from_numpy(state["cptc_states_buffer"])
        if "cptc_residuals_buffer" in state:
            self.cptc_residuals_buffer = torch.from_numpy(state["cptc_residuals_buffer"])
        if "afocp_feats_buffer" in state:
            self.afocp_feats_buffer = torch.from_numpy(state["afocp_feats_buffer"])
        if "afocp_residuals_buffer" in state:
            self.afocp_residuals_buffer = torch.from_numpy(state["afocp_residuals_buffer"])
        if "tsp_residuals" in state:
            self.tsp_residuals = torch.from_numpy(state["tsp_residuals"])
        if "tsp_weights" in state:
            self.tsp_weights = torch.from_numpy(state["tsp_weights"])
        if "agaci_experts" in state:
            self.agaci_experts = [
                {"gamma": e["gamma"], "alpha": e["alpha"], "buffer": torch.from_numpy(e["buffer"])}
                for e in state["agaci_experts"]
            ]
        if "jackknife_residuals" in state:
            self.jackknife_residuals = torch.from_numpy(state["jackknife_residuals"])

        # Non-serializable refs are cleared
        self._enbpi_member_models = None
        self._jackknife_cv_models = None
        self._jackknife_cv_indices = None
        self.cptc_state_model = None
        self.afocp_feature_extractor = None
        self.afocp_attn = None
        self._afocp_opt = None

    def reset(self):
        """Reset all state variables to allow recalibration."""
        self.radii = None
        self.cal_X_feat = None
        self.local_residuals = None
        self.residuals_buffer = None
        self.enbpi_oob_residuals = None
        self._enbpi_member_models = None
        self.cptc_state_model = None
        self.cptc_states_buffer = None
        self.cptc_residuals_buffer = None
        self.afocp_feature_extractor = None
        self.afocp_attn = None
        self.afocp_feats_buffer = None
        self.afocp_residuals_buffer = None
        self._afocp_opt = None
        self.agaci_experts = None
        self._afocp_input_dim = None
        self.cqr_correction = None
        self.jackknife_residuals = None
        self._jackknife_cv_models = None
        self._jackknife_cv_indices = None
        self.tsp_residuals = None
        self.tsp_weights = None
        # Reset alpha to original value
        self.alpha = 1.0 - self.q