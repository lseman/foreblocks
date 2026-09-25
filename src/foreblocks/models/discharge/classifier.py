"""Two-branch discharge classifier with class-conditional novelty flagging.

Implements the model recommended in
`datasets/_paper_review/model_proposal.md`: a shared multiscale waveform
encoder over 10 ms frames, fused with a dilated envelope encoder, trained for
known-class classification; a rejection layer is fit on the trained
embeddings afterward to flag signals that resemble no known class well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from foreblocks.models.discharge.backbones import (
    EnvelopeEncoder,
    TemporalAttentionPool,
    WaveformEncoder,
    compute_envelope,
)
from foreblocks.models.discharge.calibration import fit_temperature
from foreblocks.models.discharge.config import DischargeClassifierConfig
from foreblocks.models.discharge.novelty import ClassConditionalNovelty
from foreblocks.models.discharge.preprocessing import normalize_context
from foreblocks.models.discharge.validation import (
    label_vector,
    split_training_indices,
    waveform_matrix,
)


@dataclass
class DischargeResult:
    predicted_class: np.ndarray
    class_probabilities: np.ndarray
    embedding: np.ndarray
    novelty_score: np.ndarray | None = None
    is_unfamiliar: np.ndarray | None = None
    novelty_pvalue: np.ndarray | None = None


# log-RMS, log-peak, log-crest-factor: computed once, pre-normalization.
_N_CONTEXT_FEATURES = 3


class _DischargeNet(nn.Module):
    def __init__(self, config: DischargeClassifierConfig, n_classes: int):
        super().__init__()
        self.n_frames = config.n_frames
        self.waveform = WaveformEncoder(
            feature_dim=config.waveform_feature_dim,
            kernels=config.waveform_kernels,
            n_blocks=config.waveform_blocks,
            dropout=config.dropout,
        )
        self.envelope = EnvelopeEncoder(
            hidden_dim=config.envelope_hidden_dim,
            n_levels=config.envelope_levels,
            kernel_size=config.envelope_kernel_size,
            dropout=config.dropout,
        )
        self.frame_pool = (
            TemporalAttentionPool(self.waveform.output_dim)
            if config.frame_pooling == "attention"
            else None
        )
        self.use_context_features = config.use_context_features
        waveform_dim = (
            self.frame_pool.output_dim
            if self.frame_pool is not None
            else self.waveform.output_dim
        )
        fused_in = waveform_dim + self.envelope.output_dim
        if self.use_context_features:
            fused_in += _N_CONTEXT_FEATURES
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.embedding_dim, n_classes)

    def forward(
        self, frames: torch.Tensor, envelope: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_frames, frame_size = frames.shape
        per_frame = self.waveform(frames.reshape(batch * n_frames, frame_size))
        sequence = per_frame.reshape(batch, n_frames, -1)
        waveform_embed = (
            self.frame_pool(sequence)
            if self.frame_pool is not None
            else sequence.mean(dim=1)
        )
        envelope_embed = self.envelope(envelope)
        branches = [waveform_embed, envelope_embed]
        if self.use_context_features:
            branches.append(context)
        embedding = self.fusion(torch.cat(branches, dim=-1))
        logits = self.head(embedding)
        return embedding, logits


def _context_features(x: np.ndarray) -> np.ndarray:
    """Pre-normalization scale/shape statistics (log-RMS, log-peak,
    log-crest-factor), preserved explicitly since `normalize_context`
    removes the scale cue from the waveform itself. Milestone 1 found the
    model failing to separate corona from surface_discharge, two classes
    whose amplitude ranges are close but not identical
    (`recording_audit.json`); this restores that cue as a direct input
    instead of relying on the CNN to recover it from normalized shape alone.
    """
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2, axis=-1))
    peak = np.max(np.abs(x), axis=-1)
    crest = peak / np.maximum(rms, 1e-6)
    features = np.stack([np.log1p(rms), np.log1p(peak), np.log1p(crest)], axis=-1)
    return features.astype(np.float32)


class DischargeClassifier:
    def __init__(
        self, config: DischargeClassifierConfig | None = None, **kwargs
    ) -> None:
        if config is None:
            config = DischargeClassifierConfig(**kwargs)
        elif kwargs:
            config = DischargeClassifierConfig(**{**config.__dict__, **kwargs})
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: _DischargeNet | None = None
        self.novelty_: ClassConditionalNovelty | None = None
        self.classes_: list | None = None
        self.temperature_ = 1.0
        self._fitted = False
        self.context_mean_ = np.zeros(_N_CONTEXT_FEATURES, dtype=np.float32)
        self.context_scale_ = np.ones(_N_CONTEXT_FEATURES, dtype=np.float32)

    def _prepare_inputs(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = waveform_matrix(x, context_size=self.config.context_size)
        context = (_context_features(x) - self.context_mean_) / self.context_scale_
        normalized = normalize_context(x)
        frames = normalized.reshape(-1, self.config.n_frames, self.config.frame_size)
        envelope = compute_envelope(normalized, self.config.fs, self.config.envelope_hz)
        return tuple(
            torch.from_numpy(a.astype(np.float32, copy=False))
            for a in (frames, envelope, context)
        )

    def _infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit before inference")
        x = waveform_matrix(x, context_size=self.config.context_size, allow_empty=True)
        if not len(x):
            return np.empty((0, self.config.embedding_dim), dtype=np.float32), np.empty(
                (0, len(self.classes_)), dtype=np.float32
            )
        self.model.eval()
        embeddings, logits = [], []
        with torch.no_grad():
            # Preprocess each batch, bounding Hilbert/FFT and frame memory at inference.
            for start in range(0, len(x), self.config.batch_size):
                inputs = self._prepare_inputs(x[start : start + self.config.batch_size])
                embedding, output = self.model(*(a.to(self.device) for a in inputs))
                embeddings.append(embedding.cpu().numpy())
                logits.append(output.cpu().numpy())
        return np.concatenate(embeddings), np.concatenate(logits)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
        *,
        groups: np.ndarray | None = None,
    ) -> DischargeClassifier:
        """Fit on contexts, optionally reserving entire recording groups for validation.

        Split recordings before constructing overlapping windows. ``groups``
        enforces recording separation for early stopping, not calibration.
        Supply separate reserved contexts to the calibration methods afterward.
        """
        self.config.__post_init__()
        x = waveform_matrix(x, context_size=self.config.context_size)
        y = label_vector(y, len(x))
        classes, y_idx = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("At least two known classes are required.")
        train_idx, val_idx = split_training_indices(
            y, validation_split, groups=groups, seed=self.config.seed
        )
        self._fitted = False
        self.novelty_ = None
        self.temperature_ = 1.0
        self.classes_ = classes.tolist()
        self.config.class_names = tuple(self.classes_)
        self.train_indices_, self.validation_indices_ = train_idx, val_idx
        context = _context_features(x[train_idx]).astype(np.float64)
        self.context_mean_ = context.mean(axis=0).astype(np.float32)
        scale = context.std(axis=0)
        self.context_scale_ = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        inputs = self._prepare_inputs(x)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            if self.config.seed is not None:
                torch.manual_seed(int(self.config.seed))
            self._train(
                inputs, torch.from_numpy(y_idx.astype(np.int64)), train_idx, val_idx
            )
        # References must exclude validation contexts used to select the checkpoint.
        embeddings, _ = self._infer(x[train_idx])
        self.novelty_ = ClassConditionalNovelty(method=self.config.novelty_method).fit(
            embeddings, y[train_idx]
        )
        self._fitted = True
        return self

    def _train(self, inputs, labels, train_idx, val_idx):
        counts = np.bincount(labels[train_idx].numpy(), minlength=len(self.classes_))
        weights = torch.from_numpy(1.0 / counts[labels[train_idx].numpy()]).double()
        # Whole-batch index gathers replace DataLoader's per-sample collation;
        # inputs live on the device when they fit, so batches never round-trip.
        inputs = self._staged(inputs)
        labels = labels.to(inputs[0].device)
        train_idx = torch.as_tensor(train_idx, device=labels.device)
        val_idx = torch.as_tensor(val_idx, device=labels.device)
        batch_size = self.config.batch_size

        def batches(index):
            for start in range(0, len(index), batch_size):
                rows = index[start : start + batch_size]
                yield (
                    *(a[rows].to(self.device, non_blocking=True) for a in inputs),
                    labels[rows].to(self.device, non_blocking=True),
                )

        self.model = _DischargeNet(self.config, len(self.classes_)).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        autocast = torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.mixed_precision and self.device.type == "cuda",
        )
        best_val, best_state, stall = float("inf"), None, 0
        self.history_ = []
        self.best_epoch_ = None
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            # Class-balanced sampling with replacement, as WeightedRandomSampler.
            order = torch.multinomial(weights, len(train_idx), replacement=True)
            for *batch, targets in batches(train_idx[order.to(train_idx.device)]):
                optimizer.zero_grad(set_to_none=True)
                with autocast:
                    _, logits = self.model(*batch)
                loss = criterion(logits.float(), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Training produced a nonfinite loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                optimizer.step()
                total_loss += loss.detach() * len(targets)
            val_loss = None
            total_loss = float(total_loss)
            if len(val_idx):
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for *batch, targets in batches(val_idx):
                        _, logits = self.model(*batch)
                        val_loss += nn.functional.cross_entropy(
                            logits, targets, reduction="sum"
                        )
                val_loss = float(val_loss) / len(val_idx)
                if not np.isfinite(val_loss):
                    raise ValueError("Validation produced a nonfinite loss.")
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self.best_epoch_ = epoch
                    stall = 0
                else:
                    stall += 1
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_idx),
                    "validation_loss": val_loss,
                }
            )
            if len(val_idx) and stall >= self.config.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _staged(self, inputs):
        """Move training inputs to the accelerator once when they fit in a
        quarter of its free memory; otherwise batches are gathered on the host."""
        if self.device.type != "cuda":
            return inputs
        size = sum(a.numel() * a.element_size() for a in inputs)
        free, _ = torch.cuda.mem_get_info(self.device)
        return tuple(a.to(self.device) for a in inputs) if size <= free // 4 else inputs

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit before inference or calibration")

    def embed(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._infer(x)[0]

    def calibrate_probabilities(
        self, x_calibration: np.ndarray, y_calibration: np.ndarray
    ) -> float:
        """Fit one temperature on reserved labeled contexts; preserve class argmax."""
        self._check_fitted()
        x = waveform_matrix(x_calibration, context_size=self.config.context_size)
        y = label_vector(y_calibration, len(x))
        index = {label: i for i, label in enumerate(self.classes_)}
        if any(label not in index for label in y):
            raise ValueError("Calibration labels must belong to the fitted classes.")
        _, logits = self._infer(x)
        self.temperature_ = fit_temperature(
            logits, np.array([index[label] for label in y])
        )
        return self.temperature_

    def calibrate_novelty(
        self, x_calibration: np.ndarray, quantile: float | None = None
    ) -> float:
        self._check_fitted()
        x = waveform_matrix(x_calibration, context_size=self.config.context_size)
        scores = self.novelty_.score(self.embed(x))
        return self.novelty_.calibrate_threshold(
            scores, self.config.novelty_quantile if quantile is None else quantile
        )

    def predict(self, x: np.ndarray) -> DischargeResult:
        self._check_fitted()
        embeddings, logits = self._infer(x)
        scaled = logits.astype(np.float64) / self.temperature_
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predicted = np.asarray(self.classes_)[probabilities.argmax(axis=1)]
        scores = self.novelty_.score(embeddings)
        calibrated = self.novelty_.threshold_ is not None
        return DischargeResult(
            predicted_class=predicted,
            class_probabilities=probabilities,
            embedding=embeddings,
            novelty_score=scores,
            is_unfamiliar=(scores > self.novelty_.threshold_) if calibrated else None,
            novelty_pvalue=self.novelty_.p_values(embeddings) if calibrated else None,
        )
