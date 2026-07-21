"""foreblocks.anomaly.modes.

Anomaly detection modes and composable block composition.

Defines anomaly detection modes (reconstruction, forecasting, representation, hybrid,
patch_mamba, i_transformer) and the AnomalyBlock protocol for modular block composition.
Provides AnomalyBlockStack to train and combine multiple detection blocks with voting
strategies (majority, weighted, all, any).

Core API:
- AnomalyBlock: protocol for composable anomaly-detection blocks
- AnomalyBlockStack: compose multiple anomaly-detection blocks and combine decisions
- AnomalyDecisionResult: result from anomaly detection with per-block scores and voting
- AnomalyBlockSpec: declaration of one anomaly-detection block in a stack
- ReconstructionMode, ForecastingMode, RepresentationMode, HybridMode, PatchMambaMode, iTransformerMode: detection mode implementations
- resolve_mode, resolve_block, list_blocks, register_block: block registry utilities

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from foreblocks.anomaly.models import (
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

# ── AnomalyBlock protocol (foreblocks-style composable block) ──


@dataclass
class AnomalyDecisionResult:
    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    window_scores: np.ndarray
    block_scores: dict[str, np.ndarray] | None = None
    block_labels: dict[str, np.ndarray] | None = None
    voting_info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnomalyBlockSpec:
    block: str
    weights: dict[str, float] | None = None
    kwargs: dict = field(default_factory=dict)


class AnomalyBlock(Protocol):
    def build_model(self, config, n_features: int) -> torch.nn.Module: ...

    def loss(
        self, model: torch.nn.Module, batch: torch.Tensor, config, epoch: int
    ) -> torch.Tensor: ...

    def score_batch(
        self, model: torch.nn.Module, batch: torch.Tensor, config
    ) -> np.ndarray: ...

    def decide(self, scores: np.ndarray, contamination: float) -> np.ndarray: ...

    def block_type(self) -> str: ...


# ── BlockRegistry ──


_BLOCK_REGISTRY: dict[str, AnomalyBlock] = {}


def register_block(name: str):

    def decorator(cls: type) -> type:
        _BLOCK_REGISTRY[name] = cls()
        return cls

    return decorator


def resolve_block(name: str) -> AnomalyBlock:
    if name in _BLOCK_REGISTRY:
        return _BLOCK_REGISTRY[name]
    raise ValueError(f"unknown block '{name}'; valid: {list(_BLOCK_REGISTRY.keys())}")


def list_blocks() -> list[str]:
    return list(_BLOCK_REGISTRY.keys())


# ── AnomalyBlockStack (compose multiple blocks) ──


@dataclass(frozen=True)
class VotingConfig:
    strategy: str = "majority"  # majority | weighted | all
    weights: dict[str, float] | None = None  # block_name -> weight
    contamination: float = 0.01


@dataclass(frozen=True)
class DecisionConfig:
    strategy: str = "majority"  # majority | weighted | all | any
    weights: dict[str, float] | None = None  # block_name -> weight
    contamination: float = 0.01


class AnomalyBlockStack:
    def __init__(
        self,
        blocks: list[str] | list[AnomalyBlockSpec],
        decision: DecisionConfig | None = None,
    ) -> None:
        self._block_specs: list[tuple[str, AnomalyBlock, dict]] = []
        for b in blocks:
            if isinstance(b, AnomalyBlockSpec):
                spec = resolve_block(b.block)
                self._block_specs.append((b.block, spec, b.kwargs or {}))
            else:
                spec = resolve_block(b)
                self._block_specs.append((b, spec, {}))
        self.decision = decision or DecisionConfig()

    @property
    def block_names(self) -> list[str]:
        return [name for name, _, _ in self._block_specs]

    def build_models(self, config, n_features: int) -> dict[str, torch.nn.Module]:
        models = {}
        for name, block, extra_kwargs in self._block_specs:
            merged = {**config.__dict__, **extra_kwargs}
            merged_config = (
                type(config)(**merged) if hasattr(config, "__dict__") else config
            )
            models[name] = block.build_model(merged_config, n_features)
        return models

    def fit(
        self,
        models: dict[str, torch.nn.Module],
        windows: np.ndarray,
        config,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 5,
        seed: int | None = 42,
        device: torch.device | None = None,
        use_mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        num_workers: int = 0,
    ) -> AnomalyBlockStack:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = bool(use_mixed_precision) and device.type == "cuda"

        tensor = torch.from_numpy(windows.astype(np.float32, copy=False))
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(0, int(num_workers)),
            pin_memory=device.type == "cuda",
        )

        for block_name, block, extra_kwargs in self._block_specs:
            model = models[block_name].to(device)
            opt = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            merged = {**config.__dict__, **extra_kwargs}
            merged_config = (
                type(config)(**merged) if hasattr(config, "__dict__") else config
            )

            for epoch in range(epochs):
                model.train()
                for (batch,) in loader:
                    batch = batch.to(device, non_blocking=True)
                    if use_amp:
                        with torch.autocast(device.type):
                            loss = block.loss(model, batch, merged_config, epoch)
                    else:
                        loss = block.loss(model, batch, merged_config, epoch)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                    opt.step()

            models[block_name] = model

        return self

    def predict(
        self,
        models: dict[str, torch.nn.Module],
        windows: np.ndarray,
        config,
    ) -> AnomalyDecisionResult:
        _dev = next(iter(models.values()))
        device = next(_dev.parameters(), torch.tensor(0.0)).device
        tensor = torch.from_numpy(windows.astype(np.float32, copy=False))
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
        )

        block_scores: list[np.ndarray] = []
        block_labels: list[np.ndarray] = []

        for name, block, _ in self._block_specs:
            model = models[name].to(device)
            scores_list: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for (batch,) in loader:
                    batch = batch.to(device, non_blocking=True)
                    scores_list.append(block.score_batch(model, batch, config))
            raw_scores = np.concatenate(scores_list, axis=0)

            # Per-block decision
            labels = block.decide(raw_scores, self.decision.contamination)

            block_scores.append(raw_scores)
            block_labels.append(labels)

        # Combine decisions
        final_labels, combined_scores, voting_info = self._combine_decisions(
            block_labels, block_scores, self.decision
        )

        return AnomalyDecisionResult(
            scores=combined_scores,
            labels=final_labels,
            threshold=0.0,
            window_scores=(
                np.block(block_scores).T if block_scores else np.empty((0, 0))
            ),
            block_scores={
                name: scores
                for name, (_, scores) in zip(
                    self.block_names, zip(block_scores, block_labels)
                )
            },
            block_labels={
                name: lbl for name, lbl in zip(self.block_names, block_labels)
            },
            voting_info=voting_info,
        )

    def _combine_decisions(
        self,
        block_labels: list[np.ndarray],
        block_scores: list[np.ndarray],
        decision: DecisionConfig,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        n_samples = block_labels[0].shape[0]
        n_blocks = len(block_labels)
        matrix = np.column_stack(block_labels)  # (n_samples, n_blocks)

        if decision.strategy == "majority":
            votes_per_sample = matrix.sum(axis=1)
            threshold = n_blocks / 2
            final = (votes_per_sample > threshold).astype(np.int8)
            voting_info = {
                "method": "majority",
                "threshold": threshold,
                "n_blocks": n_blocks,
            }

        elif decision.strategy == "any":
            final = matrix.max(axis=1).astype(np.int8)
            voting_info = {
                "method": "any",
                "threshold": 1,
                "n_blocks": n_blocks,
            }

        elif decision.strategy == "weighted":
            weights = self._resolve_weights(n_blocks)
            weighted_votes = matrix @ weights
            threshold = weights.sum() / 2
            final = (weighted_votes > threshold).astype(np.int8)
            voting_info = {
                "method": "weighted",
                "threshold": float(threshold),
                "n_blocks": n_blocks,
                "weights": {
                    self.block_names[i]: float(w) for i, w in enumerate(weights)
                },
            }

        elif decision.strategy == "all":
            final = matrix.min(axis=1).astype(np.int8)
            voting_info = {
                "method": "all",
                "threshold": n_blocks,
                "n_blocks": n_blocks,
            }

        else:
            raise ValueError(f"unknown decision strategy: {decision.strategy}")

        # Combined scores: per-block scores concatenated
        combined = np.hstack(block_scores)

        return final, combined, voting_info

    def _resolve_weights(self, n: int) -> np.ndarray:
        weights = self.decision.weights
        if weights is None:
            return np.ones(n) / n
        name_to_weight = {name: w for name, w in weights.items()}
        return np.array([name_to_weight.get(name, 1.0) for name in self.block_names])


# ── Extend AnomalyBlock with decide method ──


def _default_decide(scores: np.ndarray, contamination: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    # Reduce 2D scores to 1D per sample via max
    if arr.ndim == 2:
        arr = arr.max(axis=1)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros(arr.shape[0], dtype=np.int8)
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    if mad <= 1e-8:
        mad = float(np.nanstd(finite)) + 1e-8
    # Use percentile-based threshold for contamination
    from foreblocks.anomaly.windows import robust_threshold

    thresh = robust_threshold(finite, contamination=contamination)
    return np.where(np.isfinite(arr) & (arr > thresh), 1, 0)


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
class ReconstructionMode:
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

    def decide(self, scores: np.ndarray, contamination: float) -> np.ndarray:
        return _default_decide(scores, contamination)

    def block_type(self) -> str:
        return "reconstruction"


@dataclass(frozen=True)
class ForecastingMode:
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

    def decide(self, scores: np.ndarray, contamination: float) -> np.ndarray:
        return _default_decide(scores, contamination)

    def block_type(self) -> str:
        return "forecasting"


@dataclass(frozen=True)
class RepresentationMode:
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

    def decide(self, scores: np.ndarray, contamination: float) -> np.ndarray:
        return _default_decide(scores, contamination)

    def block_type(self) -> str:
        return "representation"


@dataclass(frozen=True)
class HybridMode:
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


# ── Patch modes with decide() and block_type() for protocol compatibility ──


for _mode_cls in (ReconstructionMode, ForecastingMode, RepresentationMode, HybridMode):
    if not hasattr(_mode_cls, "decide"):
        _mode_cls.decide = staticmethod(_default_decide)  # type: ignore
    if not hasattr(_mode_cls, "block_type"):
        _mode_cls.block_type = lambda self=None: self.name if self else "unknown"  # type: ignore


# PatchMamba mode — SSM-based reconstruction
@dataclass(frozen=True)
class PatchMambaMode:
    name: str = "patch_mamba"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return PatchMamba(
            n_features=n_features,
            window_size=config.window_size,
            patch_size=config.get("patch_size", 4) if hasattr(config, "get") else 4,
            d_model=(
                config.get("d_model", 128) if hasattr(config, "get") else config.d_model
            ),
            n_layers=config.get("n_layers", 4) if hasattr(config, "get") else 4,
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

    def block_type(self) -> str:
        return "patch_mamba"


# iTransformer mode — inverted attention over features
@dataclass(frozen=True)
class iTransformerMode:
    name: str = "i_transformer"

    def build_model(self, config, n_features: int) -> torch.nn.Module:
        return iTransformer(
            n_features=n_features,
            window_size=config.window_size,
            d_model=(
                config.get("d_model", 128) if hasattr(config, "get") else config.d_model
            ),
            n_heads=config.get("n_heads") if hasattr(config, "get") else None,
            n_layers=config.get("n_layers", 2) if hasattr(config, "get") else 2,
            dim_feedforward=(
                config.get("dim_feedforward") if hasattr(config, "get") else None
            ),
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
        return out.feature_scores.detach().cpu().numpy()

    def block_type(self) -> str:
        return "i_transformer"


# Register built-in blocks
register_block("reconstruction")(ReconstructionMode)
register_block("forecasting")(ForecastingMode)
register_block("representation")(RepresentationMode)
register_block("hybrid")(HybridMode)
register_block("patch_mamba")(PatchMambaMode)
register_block("i_transformer")(iTransformerMode)

# Patch decide/block_type for new modes
for _mode_cls in (PatchMambaMode, iTransformerMode):
    if not hasattr(_mode_cls, "decide"):
        _mode_cls.decide = staticmethod(_default_decide)  # type: ignore
    if not hasattr(_mode_cls, "block_type"):
        _mode_cls.block_type = lambda self=None: self.name if self else "unknown"  # type: ignore


def resolve_mode(config) -> AnomalyBlock:
    mode = config.detection_mode
    if mode == "auto":
        if config.model_type == "tranad":
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
    raise ValueError(
        "detection_mode must be one of "
        "{'auto','forecasting','reconstruction','representation','hybrid'}"
    )


def fit_mode_state(
    mode: AnomalyBlock, model: torch.nn.Module, windows: np.ndarray, detector
) -> None:
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
