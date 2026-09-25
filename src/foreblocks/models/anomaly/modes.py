"""Detection strategies: build backbones, compute losses, and score batches.

Block contracts and composition live in ``blocks``; numerical scorers live in
``scorers``. Mode classes adapt those implementations to the detector pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from foreblocks.models.anomaly.backbones import (
    DAGMM,
    MLPVAE,
    AnomalyTransformer,
    ContrastiveTransformerEncoder,
    OmniAnomaly,
    PatchMamba,
    TranAD,
    TransformerForecaster,
    TransformerVAE,
    association_discrepancy,
    iTransformer,
)
from foreblocks.models.anomaly.blocks import (
    AnomalyBlock,
    BaseAnomalyBlock,
    register_block,
)
from foreblocks.models.anomaly.scorers import (
    cusum_score,
    ebs_score,
    ewma_score,
    isolation_forest_score,
    lof_score,
    matrix_profile_score,
    pca_mahalanobis_score,
    seasonal_hybrid_score,
    stl_residual_score,
)
from foreblocks.models.anomaly.scorers.empirical import COPOD, ECOD, HBOS
from foreblocks.models.anomaly.scorers.native import NATIVE_MODELS


def beta_for_epoch(config, epoch: int) -> float:
    warmup = max(1, int(config.beta_warmup_epochs))
    return float(config.beta) * min(1.0, float(epoch + 1) / warmup)


def _vae_loss(
    model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
) -> torch.Tensor:
    out = model(batch)
    recon = F.mse_loss(out.reconstruction, batch)
    kl = -0.5 * torch.mean(1.0 + out.logvar - out.mu.pow(2) - out.logvar.exp())
    return recon + beta_for_epoch(config, epoch) * kl


def _vae_score(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    recon = model.reconstruct_mean(batch)
    score = (recon - batch).pow(2).mean(dim=1)
    return score.detach().cpu().numpy()


def _reconstruction_loss(
    model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
) -> torch.Tensor:
    if isinstance(model, DAGMM):
        return model.loss(
            batch,
            energy_weight=config.energy_weight,
            covariance_weight=config.covariance_weight,
        )

    out = model(batch)
    if hasattr(out, "series") and hasattr(out, "prior"):
        recon = F.mse_loss(out.reconstruction, batch)
        return (
            recon
            + float(config.association_weight) * association_discrepancy(out).mean()
        )

    return _vae_loss(model, batch, config, epoch)


def _reconstruction_score(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    if isinstance(model, DAGMM):
        recon = model.reconstruct_mean(batch)
        recon_score = (recon - batch).pow(2).mean(dim=(1, 2))
        score = recon_score + model.energy_score(batch)
        return score.detach().cpu().numpy()

    out = model(batch)
    if hasattr(out, "series") and hasattr(out, "prior"):
        recon_score = (out.reconstruction - batch).pow(2).mean(dim=1)
        discrepancy = association_discrepancy(out).unsqueeze(1)
        return (recon_score + discrepancy).detach().cpu().numpy()

    return _vae_score(model, batch)


@dataclass(frozen=True)
class ReconstructionMode(BaseAnomalyBlock):
    name: str = "reconstruction"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        if config.model_type == "mlp_vae":
            return MLPVAE(
                n_features=n_features,
                window_size=config.window_size,
                hidden_size=config.hidden_size,
                latent_size=config.latent_size,
                dropout=config.dropout,
            )
        if config.model_type == "omni_anomaly":
            return OmniAnomaly(
                n_features=n_features,
                window_size=config.window_size,
                hidden_size=config.hidden_size,
                latent_size=config.latent_size,
                n_layers=config.n_layers,
                dropout=config.dropout,
            )
        if config.model_type == "anomaly_transformer":
            return AnomalyTransformer(
                n_features=n_features,
                window_size=config.window_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
        if config.model_type == "dagmm":
            return DAGMM(
                n_features=n_features,
                window_size=config.window_size,
                hidden_size=config.hidden_size,
                latent_size=config.latent_size,
                n_components=config.gmm_components,
                dropout=config.dropout,
            )
        return TransformerVAE(
            n_features=n_features,
            window_size=config.window_size,
            d_model=config.d_model,
            latent_size=config.latent_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_attention_type=config.layer_attention_type,
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        return _reconstruction_loss(model, batch, config, epoch)

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        return _reconstruction_score(model, batch)


@dataclass(frozen=True)
class ForecastingMode(BaseAnomalyBlock):
    name: str = "forecasting"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        if config.model_type == "tranad":
            return TranAD(
                feats=n_features,
                window_size=config.window_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                dropout=config.dropout,
            )
        return TransformerForecaster(
            n_features=n_features,
            window_size=config.window_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_attention_type=config.layer_attention_type,
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        target = batch[:, -1:, :]
        pred = model(batch)
        if isinstance(pred, tuple):
            x1, x2 = pred
            return 0.5 * F.mse_loss(x1, target) + 0.5 * F.mse_loss(x2, target)
        return F.mse_loss(pred, target)

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        target = batch[:, -1:, :]
        pred = model(batch)
        if isinstance(pred, tuple):
            x1, x2 = pred
            score = 0.5 * (x1 - target).pow(2) + 0.5 * (x2 - target).pow(2)
        else:
            score = (pred - target).pow(2)
        return score.squeeze(1).detach().cpu().numpy()


@dataclass(frozen=True)
class RepresentationMode(BaseAnomalyBlock):
    name: str = "representation"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return ContrastiveTransformerEncoder(
            n_features=n_features,
            window_size=config.window_size,
            d_model=config.d_model,
            projection_size=config.projection_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_attention_type=config.layer_attention_type,
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        view_a = _augment(batch, noise_std=config.augmentation_noise_std)
        view_b = _augment(batch, noise_std=config.augmentation_noise_std)
        z1 = model(view_a)
        z2 = model(view_b)
        logits = z1 @ z2.T / max(float(config.contrastive_temperature), 1e-4)
        labels = torch.arange(batch.shape[0], device=batch.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        emb = model(batch)
        centroid = getattr(model, "centroid_", None)
        if centroid is None:
            centroid = emb.mean(dim=0, keepdim=True)
        score = (emb - centroid.to(emb.device)).pow(2).mean(dim=1, keepdim=True)
        return score.detach().cpu().numpy()


@dataclass(frozen=True)
class HybridMode(BaseAnomalyBlock):
    name: str = "hybrid"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return torch.nn.ModuleDict(
            {
                "reconstruction": ReconstructionMode().build_model(config, n_features),
                "forecasting": ForecastingMode().build_model(config, n_features),
                "representation": RepresentationMode().build_model(config, n_features),
            }
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        recon = ReconstructionMode().loss(model["reconstruction"], batch, config, epoch)
        forecast = ForecastingMode().loss(model["forecasting"], batch, config, epoch)
        rep = RepresentationMode().loss(model["representation"], batch, config, epoch)
        return (
            float(config.reconstruction_weight) * recon
            + float(config.forecasting_weight) * forecast
            + float(config.representation_weight) * rep
        )

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        scores = [
            _normalize_batch_scores(
                ReconstructionMode().score_batch(model["reconstruction"], batch, config)
            ),
            _normalize_batch_scores(
                ForecastingMode().score_batch(model["forecasting"], batch, config)
            ),
            _normalize_batch_scores(
                RepresentationMode().score_batch(model["representation"], batch, config)
            ),
        ]
        weights = np.array(
            [
                float(config.reconstruction_weight),
                float(config.forecasting_weight),
                float(config.representation_weight),
            ],
            dtype=np.float32,
        )
        weights = weights / max(float(weights.sum()), 1e-8)
        return weights[0] * scores[0] + weights[1] * scores[1] + weights[2] * scores[2]


# PatchMamba mode — SSM-based reconstruction
@dataclass(frozen=True)
class PatchMambaMode(BaseAnomalyBlock):
    name: str = "patch_mamba"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return PatchMamba(
            n_features=n_features,
            window_size=config.window_size,
            patch_size=config.get("patch_size", 4) if hasattr(config, "get") else 4,
            d_model=(
                config.get("d_model", 128) if hasattr(config, "get") else config.d_model
            ),
            n_layers=getattr(config, "n_layers", 4),
            d_state=config.get("d_state", 16) if hasattr(config, "get") else 16,
            dropout=(
                config.get("dropout", 0.1) if hasattr(config, "get") else config.dropout
            ),
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        return F.mse_loss(model(batch).reconstruction, batch)

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        with torch.no_grad():
            out = model(batch)
        return out.per_token_scores.detach().cpu().numpy()


# iTransformer mode — inverted attention over features
@dataclass(frozen=True)
class iTransformerMode(BaseAnomalyBlock):
    name: str = "i_transformer"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return iTransformer(
            n_features=n_features,
            window_size=config.window_size,
            d_model=(
                config.get("d_model", 128) if hasattr(config, "get") else config.d_model
            ),
            n_heads=getattr(config, "n_heads", None),
            n_layers=getattr(config, "n_layers", 2),
            dim_feedforward=(getattr(config, "dim_feedforward", None)),
            dropout=(
                config.get("dropout", 0.1) if hasattr(config, "get") else config.dropout
            ),
        )

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor:
        return F.mse_loss(model(batch).reconstruction, batch)

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray:
        with torch.no_grad():
            out = model(batch)
        return (out.reconstruction - batch).square().mean(dim=(1, 2)).cpu().numpy()


# ── Classical (non-deep) mode ──


@dataclass(frozen=True)
class ClassicalMode(BaseAnomalyBlock):
    name: str = "classical"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        model = torch.nn.Identity()
        scorers = {"ecod": ECOD, "copod": COPOD, "hbos": HBOS}
        if config.model_type in scorers:
            model.scorer = (
                HBOS(n_bins=config.hbos_bins, alpha=config.hbos_alpha)
                if config.model_type == "hbos"
                else scorers[config.model_type]()
            )
        return model

    def loss(self, model, batch, config, epoch):
        return 0.0  # No loss for classical methods

    def score_batch(self, model, batch, config) -> np.ndarray:
        windows = batch.detach().cpu().numpy()
        if hasattr(model, "scorer"):
            return model.scorer.decision_function(windows)
        if config.model_type == "isolation_forest":
            return isolation_forest_score(windows)
        if config.model_type == "lof":
            return lof_score(windows)
        if config.model_type == "pca_mahalanobis":
            return pca_mahalanobis_score(windows)
        if config.model_type == "matrix_profile":
            return matrix_profile_score(windows)
        # Default: isolation forest
        return isolation_forest_score(windows)


@dataclass(frozen=True)
class NativeMode(ClassicalMode):
    """Fitted native estimators own their numerical or neural training lifecycle."""

    name: str = "native"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        if config.model_type not in NATIVE_MODELS:
            raise ValueError(
                f"Unknown native model {config.model_type!r}; "
                f"choose from {tuple(NATIVE_MODELS)}"
            )
        kwargs = {"seed": config.seed, "standardize": False}
        kind = config.model_type
        if kind in {"autoencoder", "vae", "deep_svdd"}:
            kwargs.update(
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                weight_decay=config.weight_decay,
            )
        elif kind == "dif":
            kwargs["batch_size"] = config.batch_size
        kwargs.update(config.scorer_kwargs)
        model = torch.nn.Identity()
        model.scorer = NATIVE_MODELS[kind](**kwargs)
        return model


# ── Statistical mode ──


@dataclass(frozen=True)
class StatisticalMode(BaseAnomalyBlock):
    name: str = "statistical"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return torch.nn.Identity()  # No model needed; scoring done in score_batch

    def loss(self, model, batch, config, epoch):
        return 0.0  # No loss for statistical methods

    def score_batch(self, model, batch, config) -> np.ndarray:
        windows = batch.detach().cpu().numpy()
        if config.model_type == "ebs":
            return ebs_score(windows)
        if config.model_type == "cusum":
            return cusum_score(windows)
        if config.model_type == "ewma":
            return ewma_score(windows)
        if config.model_type == "seasonal_hybrid":
            return seasonal_hybrid_score(windows)
        if config.model_type == "stl_residual":
            return stl_residual_score(windows)
        # Default: EBS
        return ebs_score(windows)


# Register built-in blocks
register_block("reconstruction")(ReconstructionMode)
register_block("forecasting")(ForecastingMode)
register_block("representation")(RepresentationMode)
register_block("hybrid")(HybridMode)
register_block("patch_mamba")(PatchMambaMode)
register_block("i_transformer")(iTransformerMode)
register_block("classical")(ClassicalMode)
register_block("native")(NativeMode)
register_block("statistical")(StatisticalMode)


def resolve_mode(config) -> AnomalyBlock:
    mode = config.detection_mode
    if config.model_type in NATIVE_MODELS and mode not in {"auto", "native"}:
        raise ValueError("Native model types require detection_mode='auto' or 'native'.")
    if mode == "auto":
        if config.model_type in NATIVE_MODELS:
            mode = "native"
        elif config.model_type in {
            "ecod",
            "copod",
            "hbos",
            "isolation_forest",
            "lof",
            "pca_mahalanobis",
            "matrix_profile",
        }:
            mode = "classical"
        elif config.model_type in {
            "ebs",
            "cusum",
            "ewma",
            "seasonal_hybrid",
            "stl_residual",
        }:
            mode = "statistical"
        elif config.model_type in {"patch_mamba", "i_transformer"}:
            mode = config.model_type
        elif config.model_type == "tranad":
            mode = "forecasting"
        else:
            mode = "reconstruction"
    if mode == "forecasting":
        return ForecastingMode()
    if mode == "reconstruction":
        return ReconstructionMode()
    if mode == "representation":
        return RepresentationMode()
    if mode == "hybrid":
        return HybridMode()
    if mode == "patch_mamba":
        return PatchMambaMode()
    if mode == "i_transformer":
        return iTransformerMode()
    if mode == "native":
        return NativeMode()
    if mode == "classical":
        return ClassicalMode()
    if mode == "statistical":
        return StatisticalMode()
    raise ValueError(
        "detection_mode must be one of "
        "{'auto','forecasting','reconstruction','representation','hybrid',"
        "'patch_mamba','i_transformer','classical','native','statistical'}"
    )


def fit_mode_state(
    mode: AnomalyBlock, model: torch.nn.Module, windows: np.ndarray, detector
) -> None:
    if hasattr(model, "scorer"):
        model.scorer.fit(windows)
    if mode.name == "reconstruction" and isinstance(model, DAGMM):
        tensor = torch.from_numpy(windows.astype(np.float32, copy=False)).to(
            detector.device
        )
        model.fit_density(tensor, batch_size=detector.config.batch_size)
    if mode.name == "representation":
        _fit_representation_centroid(model, windows, detector)
    if mode.name == "hybrid":
        if isinstance(model["reconstruction"], DAGMM):
            tensor = torch.from_numpy(windows.astype(np.float32, copy=False)).to(
                detector.device
            )
            model["reconstruction"].fit_density(
                tensor,
                batch_size=detector.config.batch_size,
            )
        _fit_representation_centroid(model["representation"], windows, detector)


def _fit_representation_centroid(
    model: torch.nn.Module, windows: np.ndarray, detector
) -> None:
    reps = []
    model.eval()
    with torch.no_grad():
        for (batch,) in detector._loader(windows, shuffle=False):
            batch = batch.to(detector.device, non_blocking=True)
            reps.append(model(batch).detach().cpu())
    if reps:
        centroid = torch.cat(reps, dim=0).mean(dim=0, keepdim=True)
        model.register_buffer("centroid_", centroid, persistent=False)


def _augment(batch: torch.Tensor, *, noise_std: float) -> torch.Tensor:
    noise = float(noise_std) * torch.randn_like(batch)
    return batch + noise


def _normalize_batch_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True) + 1e-6
    return (arr - mean) / std
