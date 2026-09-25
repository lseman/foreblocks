"""Anomaly block contracts, registry, composition, and voting decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from foreblocks.models.anomaly.windows import robust_threshold


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
class DecisionConfig:
    strategy: str = "majority"  # majority | weighted | all | any
    weights: dict[str, float] | None = None  # block_name -> weight
    contamination: float = 0.01


VotingConfig = DecisionConfig


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
            if hasattr(model, "scorer"):
                model.scorer.fit(windows)
                models[block_name] = model
                continue
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
            block_labels=dict(zip(self.block_names, block_labels)),
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
        name_to_weight = dict(weights)
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
    thresh = robust_threshold(finite, contamination=contamination)
    return np.where(np.isfinite(arr) & (arr > thresh), 1, 0)


class BaseAnomalyBlock:
    """Shared decision behavior for built-in detection modes."""

    name: str

    def decide(self, scores: np.ndarray, contamination: float) -> np.ndarray:
        return _default_decide(scores, contamination)

    def block_type(self) -> str:
        return self.name
