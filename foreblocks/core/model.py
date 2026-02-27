from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from foreblocks.ui.node_spec import node

# =============================================================================
# Graph utilities
# =============================================================================


def _ensure_batch_adj(adj: Optional[torch.Tensor], B: int) -> Optional[torch.Tensor]:
    """
    Normalize adjacency to [B, N, N] or return None.

    Accepts:
      - None
      - [N, N]
      - [1, N, N]  (broadcasted to B)
      - [B, N, N]

    Raises:
      ValueError if the shape is inconsistent or non-square.
    """
    if adj is None:
        return None

    if adj.dim() == 2:
        # [N, N] -> [B, N, N]
        N = adj.size(0)
        if adj.size(1) != N:
            raise ValueError(f"adj must be square, got [{N}, {adj.size(1)}]")
        return adj.unsqueeze(0).expand(B, N, N).contiguous()

    if adj.dim() == 3:
        b_adj, n1, n2 = adj.shape
        if n1 != n2:
            raise ValueError(f"adj must be square on last dims, got [{n1}, {n2}]")
        if b_adj == 1 and B > 1:
            # Broadcast [1, N, N] -> [B, N, N]
            return adj.expand(B, n1, n2).contiguous()
        if b_adj != B:
            raise ValueError(
                f"Batch mismatch for adj: got batch={b_adj}, expected B={B}"
            )
        return adj

    raise ValueError(
        f"adj must be shape [N, N] or [B, N, N] (or None), got {tuple(adj.shape)}"
    )


class TimewiseGraph(nn.Module):
    """
    Adapter: lift any graph block that consumes [B, N, F] (+ adj)
    to work on [B, T, N, F] by applying it independently per time step.
    If x is already [B, N, F], just forwards.
    """

    def __init__(self, gblock: nn.Module):
        super().__init__()
        self.gblock = gblock
        # mark capability to avoid re-wrapping
        self._accepts_time_dim = True

    def forward(
        self, x: torch.Tensor, adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 3:  # [B, N, F]
            return self.gblock(x, adj)
        assert x.dim() == 4, "TimewiseGraph expects [B, N, F] or [B, T, N, F]"
        B = x.shape[0]
        A = _ensure_batch_adj(adj, B)
        # loop over time with unbind (slightly cheaper indexing)
        ys = [self.gblock(xt, A).unsqueeze(1) for xt in x.unbind(dim=1)]
        return torch.cat(ys, dim=1)  # [B, T, N, F']


# =============================================================================
# Base head wrapper
# =============================================================================


class BaseHead(nn.Module):
    """Base class for all heads in the forecasting model.

    Wraps an nn.Module to standardize:
      - aux_loss handling
      - naming / inspection
      - attribute delegation
    """

    def __init__(self, module: nn.Module, name: Optional[str] = None) -> None:
        super().__init__()
        self.module = module
        self.name: str = name or self.__class__.__name__
        self._fallback_device: Optional[torch.device] = None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ---- aux loss handling ---------------------------------------------------
    def _infer_device(self) -> torch.device:
        """Infer and cache a reasonable device for scalar tensors."""
        if self._fallback_device is not None:
            return self._fallback_device

        for p in self.module.parameters():
            self._fallback_device = p.device
            return p.device
        for b in self.module.buffers():
            self._fallback_device = b.device
            return b.device

        self._fallback_device = torch.device("cpu")
        return self._fallback_device

    def get_aux_loss(self) -> torch.Tensor:
        """Return aux_loss from wrapped module if present, else 0 on correct device."""
        device = self._infer_device()
        if hasattr(self.module, "aux_loss"):
            aux = self.module.aux_loss
            if torch.is_tensor(aux):
                return aux.to(device)
            return torch.as_tensor(aux, device=device)
        return torch.zeros((), device=device)

    # ---- attribute delegation ------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# =============================================================================
# ForecastingModel
# =============================================================================


@node(
    type_id="forecasting_model",
    name="Forecasting Model",
    category="Models",
    outputs=["model"],
    infer=True,
    color="bg-gradient-to-br from-blue-700 to-blue-800",
)
class ForecastingModel(nn.Module):
    """
    Head-based modular forecasting model for time series tasks.

    Simplified:
      • No multi-encoder/decoder path.
      • No VAE handling.
      • For 'direct' strategy, uses self.head (not decoder).
      • Keeps 4D graph dispatch [B, T, N, F] -> per-node temporal processing.
    """

    VALID_STRATEGIES = ["seq2seq", "autoregressive", "direct", "transformer_seq2seq"]
    VALID_MODEL_TYPES = ["lstm", "transformer", "informer-like", "head_only"]

    def __init__(
        self,
        *,
        # Core modules
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        # Strategy & shape
        forecasting_strategy: str = "seq2seq",
        model_type: str = "lstm",
        target_len: int = 5,
        output_size: Optional[int] = None,
        hidden_size: int = 64,
        # Processing modules
        input_preprocessor: Optional[nn.Module] = None,
        output_postprocessor: Optional[nn.Module] = None,
        input_normalization: Optional[nn.Module] = None,
        output_normalization: Optional[nn.Module] = None,
        output_block: Optional[nn.Module] = None,
        input_skip_connection: bool = False,
        # Attention
        attention_module: Optional[nn.Module] = None,
        # Training
        teacher_forcing_ratio: float = 0.5,
        scheduled_sampling_fn: Optional[Callable[[Optional[int]], float]] = None,
        # Time embeddings
        time_feature_embedding_enc: Optional[nn.Module] = None,
        time_feature_embedding_dec: Optional[nn.Module] = None,
        # Transformer-style decoding prompt length
        label_len: Optional[int] = None,
        # Optional HeadComposer to own pre/inverse processing
        head_composer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # === Validate ===
        if forecasting_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid forecasting strategy: {forecasting_strategy}")
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type: {model_type}")

        # === Store Core Params ===
        self.strategy = forecasting_strategy
        self.model_type = model_type
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)
        self.scheduled_sampling_fn = scheduled_sampling_fn
        self.input_skip_connection = input_skip_connection

        # === Wrap processors in heads ===
        self.input_preprocessor = self._wrap_head(
            input_preprocessor or nn.Identity(), "input_preprocessor"
        )
        self.output_postprocessor = self._wrap_head(
            output_postprocessor or nn.Identity(), "output_postprocessor"
        )
        self.input_normalization = self._wrap_head(
            input_normalization or nn.Identity(), "input_normalization"
        )
        self.output_normalization = self._wrap_head(
            output_normalization or nn.Identity(), "output_normalization"
        )
        self.output_block = self._wrap_head(
            output_block or nn.Identity(), "output_block"
        )

        # === Time Embeddings ===
        self.time_feature_embedding_enc = time_feature_embedding_enc
        self.time_feature_embedding_dec = time_feature_embedding_dec

        # === Attention ===
        self.use_attention = attention_module is not None
        self.attention_module = attention_module

        # === Architecture ===
        self.encoder = encoder
        self.decoder = decoder
        self.head = head if head is not None else nn.Identity()

        # Infer sizes (only needed for non-direct strategies)
        self.input_size = getattr(encoder, "input_size", None) if encoder else None
        self.output_size = output_size or (
            getattr(decoder, "output_size", None) if decoder else None
        )
        if self.strategy != "direct" and self.output_size is None:
            raise ValueError(
                "For non-direct strategies, provide output_size or a decoder with .output_size"
            )

        # label_len is a prompt length for transformer decoding
        self.label_len = label_len if label_len is not None else target_len // 2

        # Aux (kept for backward compatibility)
        self._kl: Optional[torch.Tensor] = None
        self._mem_model_bridge: Optional[nn.Module] = None

        # Small cache for causal masks to avoid re-allocating every step
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        # === Decoder input/feedback projection bridges (if decoder exists) ===
        if self.decoder is not None:
            self._decoder_input_size = getattr(
                self.decoder, "input_size", self.input_size or self.output_size
            )
            if (
                self._decoder_input_size
                and self.input_size
                and self.input_size != self._decoder_input_size
            ):
                self.dec_init_proj = nn.Linear(
                    self.input_size, self._decoder_input_size
                )
            else:
                self.dec_init_proj = nn.Identity()
            if (
                self._decoder_input_size
                and self.output_size
                and self.output_size != self._decoder_input_size
            ):
                self.dec_feedback_proj = nn.Linear(
                    self.output_size, self._decoder_input_size
                )
            else:
                self.dec_feedback_proj = nn.Identity()
        else:
            self._decoder_input_size = None
            self.dec_init_proj = nn.Identity()
            self.dec_feedback_proj = nn.Identity()

        # === Output layers (seq2seq paths) ===
        self._setup_output_layers()

        # === Optional graph blocks registry ===
        self._graph_blocks: Dict[str, nn.Module] = {}

        # === Validate configuration ===
        self._validate_configuration()

        # === HeadComposer (optional) ===
        self.head_composer = head_composer
        if self.head_composer is not None:
            self._to_model_device_dtype(self.head_composer)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def pred_len(self) -> int:
        """Alias for target_len."""
        return self.target_len

    # -----------------------------
    # Composer helpers (kept minimal)
    # -----------------------------
    def set_head_composer(self, composer: nn.Module) -> "ForecastingModel":
        """Attach a HeadComposer post-hoc; moved to model device/dtype."""
        self.head_composer = composer
        self._to_model_device_dtype(self.head_composer)
        return self

    def clear_head_composer(self) -> "ForecastingModel":
        """Detach the HeadComposer."""
        self.head_composer = None
        return self

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def add_head(self, head: nn.Module, position: str = "input", name: str = None):
        """
        Add/replace a head at positions:
          'encoder', 'decoder', 'attention', 'input', 'output',
          'input_norm', 'output_norm', 'head'
        """
        if not isinstance(head, BaseHead):
            head = BaseHead(head, name or f"{position}_head")

        if position == "encoder":
            self.encoder = head.module
        elif position == "decoder":
            self.decoder = head.module
        elif position == "attention":
            self.attention_module = head.module
            self.use_attention = True
        elif position == "input":
            old = self.input_preprocessor.module
            self.input_preprocessor = (
                head
                if isinstance(old, nn.Identity)
                else BaseHead(
                    nn.Sequential(old, head.module), "input_preprocessor_chain"
                )
            )
        elif position == "output":
            old = self.output_postprocessor.module
            self.output_postprocessor = (
                head
                if isinstance(old, nn.Identity)
                else BaseHead(
                    nn.Sequential(old, head.module), "output_postprocessor_chain"
                )
            )
        elif position == "input_norm":
            self.input_normalization = head
        elif position == "output_norm":
            self.output_normalization = head
        elif position == "head":
            self.head = head.module
            self._setup_output_layers()
        else:
            raise ValueError(f"Invalid position: {position}")

        return self

    def remove_head(self, position: str) -> "ForecastingModel":
        """Remove head at specified position."""
        if position == "encoder":
            self.encoder = None
        elif position == "decoder":
            self.decoder = None
        elif position == "attention":
            self.attention_module = None
            self.use_attention = False
        elif position == "input":
            self.input_preprocessor = BaseHead(nn.Identity(), "input_preprocessor")
        elif position == "output":
            self.output_postprocessor = BaseHead(nn.Identity(), "output_postprocessor")
        elif position == "input_norm":
            self.input_normalization = BaseHead(nn.Identity(), "input_normalization")
        elif position == "output_norm":
            self.output_normalization = BaseHead(nn.Identity(), "output_normalization")
        elif position == "head":
            self.head = nn.Identity()
        else:
            raise ValueError(f"Invalid position: {position}")
        return self

    def list_heads(self) -> Dict[str, Any]:
        """List all registered heads in the model."""
        heads: Dict[str, Any] = {}
        if self.encoder:
            heads["encoder"] = type(self.encoder).__name__
        if self.decoder:
            heads["decoder"] = type(self.decoder).__name__
        if self.attention_module:
            heads["attention"] = type(self.attention_module).__name__

        for name in [
            "input_preprocessor",
            "output_postprocessor",
            "input_normalization",
            "output_normalization",
        ]:
            head = getattr(self, name, None)
            if isinstance(head, BaseHead):
                heads[name] = {
                    "name": head.name,
                    "type": type(head.module).__name__,
                    "is_identity": isinstance(head.module, nn.Identity),
                }

        if hasattr(self, "head"):
            heads["head"] = type(self.head).__name__
        return heads

    def add_graph_block(self, graph_block: nn.Module, where: str = "pre_encoder"):
        """
        Register a graph block; where ∈ {"pre_encoder","post_encoder","post_decoder"}.
        The block should accept (x, adj) with shapes:
          x: [B, T, N, F] or [B, N, F]
          adj: [N, N] or [B, N, N]
        """
        valid_positions = {"pre_encoder", "post_encoder", "post_decoder"}
        if where not in valid_positions:
            raise ValueError(
                f"Invalid position '{where}'. Must be one of {valid_positions}"
            )

        # Wrap once to operate timewise if needed
        if not getattr(graph_block, "_accepts_time_dim", False):
            graph_block = TimewiseGraph(graph_block)

        # Move to the same device/dtype as the model
        self._to_model_device_dtype(graph_block)

        self._graph_blocks[where] = graph_block
        return self

    def get_aux_loss(self) -> torch.Tensor:
        """Aggregate auxiliary losses from all components."""
        device = self._get_model_device_dtype()[0]
        loss = torch.zeros((), device=device)

        # Heads
        for head_name in [
            "input_preprocessor",
            "output_postprocessor",
            "input_normalization",
            "output_normalization",
        ]:
            head = getattr(self, head_name, None)
            if isinstance(head, BaseHead):
                loss = loss + head.get_aux_loss().to(device)

        # Encoder / decoder
        if hasattr(self, "encoder") and hasattr(self.encoder, "aux_loss"):
            aux = self.encoder.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        if hasattr(self, "decoder") and hasattr(self.decoder, "aux_loss"):
            aux = self.decoder.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        # Model-level aux_loss (if you ever add it)
        if hasattr(self, "aux_loss"):
            aux = self.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        # in get_aux_loss()
        if self.head_composer is not None and hasattr(
            self.head_composer, "get_aux_loss"
        ):
            loss = loss + self.head_composer.get_aux_loss().to(device)
        elif self.head_composer is not None and hasattr(self.head_composer, "aux_loss"):
            aux = self.head_composer.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        return loss

    def get_kl(self) -> Optional[torch.Tensor]:
        """Get KL divergence loss (kept for backward compatibility)."""
        return self._kl

    def get_model_size(self) -> Dict[str, Any]:
        """Calculate model size statistics."""
        params = sum(p.numel() for p in self.parameters())
        buffers = sum(b.numel() for b in self.buffers())
        size_mb = (params + buffers) * 4 / 1024**2
        return {
            "parameters": params,
            "buffers": buffers,
            "total_elements": params + buffers,
            "size_mb": size_mb,
            "is_quantized": False,
        }

    def benchmark_inference(
        self, input_tensor: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark inference time and throughput."""
        import time

        self.eval()
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        avg = (end - start) / num_runs
        return {
            "avg_inference_time_ms": avg * 1000.0,
            "throughput_samples_per_sec": 1.0 / avg if avg > 0 else float("inf"),
            "device": str(device),
        }

    def attribute_forward(
        self,
        src: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        output_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass for attribution analysis (gradients enabled)."""
        self.train()
        self._disable_dropout()
        src = src.requires_grad_()
        out = self.forward(
            src, targets=targets, time_features=time_features, epoch=epoch
        )
        return out[..., output_idx] if output_idx is not None else out

    # -------------------------------------------------------------------------
    # Private / setup
    # -------------------------------------------------------------------------
    def _wrap_head(self, module: nn.Module, name: str) -> BaseHead:
        """Wrap module in BaseHead if not already wrapped."""
        if isinstance(module, BaseHead):
            return module
        return BaseHead(module, name)

    def _get_model_device_dtype(self) -> Tuple[torch.device, Optional[torch.dtype]]:
        """Get device and dtype from model parameters."""
        ref_param = next(self.parameters(), None)
        if ref_param is None:
            return torch.device("cpu"), None
        return ref_param.device, ref_param.dtype

    def _to_model_device_dtype(self, module: nn.Module) -> None:
        """Move module to model's device/dtype."""
        device, dtype = self._get_model_device_dtype()
        if dtype is not None:
            module.to(device=device, dtype=dtype)
        else:
            module.to(device=device)

    def _validate_configuration(self) -> None:
        """Validate model configuration consistency."""
        if self.strategy != "direct" and self.decoder is None:
            raise ValueError(f"Strategy '{self.strategy}' requires a decoder")

        if self.strategy in ("seq2seq", "transformer_seq2seq"):
            if self.encoder is None or self.decoder is None:
                raise ValueError(
                    f"Strategy '{self.strategy}' requires both encoder and decoder"
                )

        if self.use_attention and self.attention_module is None:
            raise ValueError("use_attention=True but attention_module is None")

    def _setup_output_layers(self) -> None:
        """Prepare output_head and projection for seq2seq paths."""
        if self.decoder is None:
            self.output_head = nn.Identity()
            self.project_output = nn.Identity()
            return

        encoder_hidden = (
            getattr(self.encoder, "hidden_size", self.hidden_size)
            if self.encoder is not None
            else self.hidden_size
        )
        decoder_hidden = getattr(self.decoder, "hidden_size", self.hidden_size)
        if self.output_size is None:
            raise ValueError(
                "output_size must be provided (or decoder must have .output_size) for non-direct strategies."
            )

        out_dim = (
            (decoder_hidden + encoder_hidden) if self.use_attention else decoder_hidden
        )
        self.output_head = self._create_output_projection(out_dim, self.output_size)

        if self.input_size and self.input_size != self.output_size:
            self.project_output = nn.Linear(self.input_size, self.output_size)
        else:
            self.project_output = nn.Identity()

    def _create_output_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create output projection with consistent architecture."""
        hidden_dim = max(in_dim // 2, out_dim)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _apply_graph_block(
        self, x: torch.Tensor, adj: Optional[torch.Tensor], block_name: str
    ) -> torch.Tensor:
        """Apply registered graph block by name."""
        block = self._graph_blocks.get(block_name)
        if block is None:
            return x
        A = _ensure_batch_adj(adj, x.size(0))
        if A is not None:
            A = A.to(device=x.device, dtype=x.dtype)
        return block(x, A)

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _preprocess_input(self, src: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing with optional skip connection."""
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _finalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final normalization/postprocessing exactly once."""
        return self.output_postprocessor(self.output_normalization(x))

    def _get_strategy_fn(self):
        """Resolve active strategy to its forward implementation."""
        return {
            "direct": self._forward_direct,
            "autoregressive": self._forward_autoregressive,
            "seq2seq": self._forward_seq2seq,
            "transformer_seq2seq": self._forward_seq2seq,
        }[self.strategy]

    def _run_strategy(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor],
        time_features: Optional[torch.Tensor],
        epoch: Optional[int],
    ) -> torch.Tensor:
        """Run selected strategy and return raw predictions."""
        return self._get_strategy_fn()(src, targets, time_features, epoch)

    def _encode_memory(
        self,
        src: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source (with optional time features) and align to decoder dim."""
        if self.encoder is None:
            raise RuntimeError("Encoder is required for this strategy.")
        enc = (
            self.encoder(src, time_features=time_features)
            if time_features is not None
            else self.encoder(src)
        )
        enc_out = enc[0] if isinstance(enc, tuple) else enc
        return self._align_memory_to_decoder(enc_out)

    @staticmethod
    def _targets_to_horizon(
        targets: Optional[torch.Tensor],
        horizon: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Clip/pad targets to exactly `horizon` steps."""
        if targets is None:
            return None
        if targets.size(1) >= horizon:
            return targets[:, :horizon, :]
        pad = torch.zeros(
            batch_size,
            horizon - targets.size(1),
            targets.size(2),
            device=device,
            dtype=dtype,
        )
        return torch.cat([targets, pad], dim=1)

    def _next_decoder_input(
        self,
        output: torch.Tensor,
        targets: Optional[torch.Tensor],
        t: int,
        use_tf: bool,
    ) -> torch.Tensor:
        """Get next decoder input (teacher forcing or model output)."""
        if use_tf and targets is not None:
            return targets[:, t : t + 1, :]
        return output if output.dim() == 3 else output.unsqueeze(1)

    def _rollout_decoder(
        self,
        *,
        decoder_input: torch.Tensor,
        steps: int,
        targets: Optional[torch.Tensor],
        epoch: Optional[int],
        step_fn: Callable[[torch.Tensor, int], torch.Tensor],
        select_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        aggregate: str = "cat",
    ) -> torch.Tensor:
        """
        Generic autoregressive rollout helper used by decoder-based strategies.

        Args:
            decoder_input: initial decoder input [B, 1, Din] (or prompt slice).
            steps: number of decoding steps.
            targets: optional teacher-forcing targets [B, H, Dout].
            epoch: training epoch for scheduled sampling.
            step_fn: returns per-step raw output tensor.
            select_fn: optional projection/slicing before aggregation.
            aggregate: 'cat' for sequence tensors or 'stack' for step vectors.
        """
        use_tf = self._should_use_teacher_forcing(targets, epoch)
        outputs = []

        for t in range(steps):
            step_out = step_fn(decoder_input, t)
            selected = select_fn(step_out)
            outputs.append(selected)

            if t < steps - 1:
                next_in = self._next_decoder_input(selected, targets, t, use_tf)
                decoder_input = self.dec_feedback_proj(next_in)

        if aggregate == "cat":
            return torch.cat(outputs, dim=1)
        if aggregate == "stack":
            return torch.stack(outputs, dim=1)
        raise ValueError(f"Unknown aggregate mode: {aggregate}")

    def _should_use_teacher_forcing(
        self,
        targets: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        fallback_device: str = "cpu",
    ) -> bool:
        """Determine whether to use teacher forcing for current step."""
        if (not self.training) or (targets is None):
            return False
        # avoid randomness during tracing
        if self._is_fx_tracing() or torch.jit.is_tracing():
            return False
        if self.scheduled_sampling_fn is not None and epoch is not None:
            ratio = float(self.scheduled_sampling_fn(epoch))
        else:
            ratio = self.teacher_forcing_ratio
        ratio = float(max(0.0, min(1.0, ratio)))
        device = getattr(targets, "device", torch.device(fallback_device))
        return torch.rand((1,), device=device).item() < ratio

    def _prepare_decoder_hidden(self, encoder_hidden: Any) -> Any:
        """Prepare decoder hidden state (merge bidirectional if needed)."""
        if self.encoder is None or not getattr(self.encoder, "bidirectional", False):
            return encoder_hidden
        # Bidirectional RNN handling
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            return (self._merge_bidirectional(h_n), self._merge_bidirectional(c_n))
        return self._merge_bidirectional(encoder_hidden)

    def _merge_bidirectional(self, hidden: torch.Tensor) -> torch.Tensor:
        """Merge bidirectional hidden states."""
        assert hidden.size(0) % 2 == 0, (
            "Expected even number of layers for bidirectional RNN"
        )
        num_layers = hidden.size(0) // 2
        reshaped = hidden.reshape(num_layers, 2, *hidden.shape[1:])
        return reshaped.sum(dim=1)

    def _get_attention_query(
        self, decoder_output: torch.Tensor, decoder_hidden: Any
    ) -> torch.Tensor:
        """Extract attention query from decoder state."""
        if hasattr(self.decoder, "is_transformer") and getattr(
            self.decoder, "is_transformer"
        ):
            return decoder_hidden.permute(1, 0, 2)  # (batch, seq_len, hidden)
        return (
            decoder_hidden[0][-1]
            if isinstance(decoder_hidden, tuple)
            else decoder_hidden[-1]
        )

    @staticmethod
    def _is_fx_tracing() -> bool:
        """Best-effort check if currently in tracing mode."""
        try:
            import torch.fx  # noqa: F401

            # If this ever changes upstream, we still return False gracefully
            return False
        except Exception:
            return False

    def _disable_dropout(self) -> None:
        """Disable dropout for all layers (used in attribution)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    def _decoder_model_dim(self) -> Optional[int]:
        """Get model dimension from decoder if available."""
        if self.decoder is None:
            return None
        for name in ("d_model", "model_dim", "embed_dim", "dim", "hidden_dim"):
            if hasattr(self.decoder, name) and isinstance(
                getattr(self.decoder, name), int
            ):
                return getattr(self.decoder, name)
        return None

    def _align_memory_to_decoder(self, memory: torch.Tensor) -> torch.Tensor:
        """Align encoder memory dimension to decoder's expected dimension."""
        d_model = self._decoder_model_dim()
        if d_model is None or memory is None or memory.size(-1) == d_model:
            return memory
        if self._mem_model_bridge is None:
            self._mem_model_bridge = nn.Linear(memory.size(-1), d_model).to(
                memory.device
            )
        return self._mem_model_bridge(memory)

    def _supports_time(self, module: nn.Module) -> bool:
        """Check if module supports time features."""
        return (
            hasattr(module, "supports_time_features")
            or "time" in " ".join(dir(module)).lower()
        )

    # --- mask cache helpers ---------------------------------------------------
    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        True = masked (blocked), False = allowed.
        Cached per (L, device) for speed.
        """
        key = (L, device)
        mask = self._mask_cache.get(key)
        if mask is None or mask.device != device:
            mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
            )
            self._mask_cache[key] = mask
        return mask

    # -------------------------------------------------------------------------
    # Forward (Public) — finalize exactly once here
    # -------------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Main forward pass. Returns finalized predictions.

        Args:
            src: Input tensor [B, T, F] or [B, T, N, F]
            targets: Target tensor (for teacher forcing)
            time_features: Time-related features
            epoch: Current epoch (for scheduled sampling)

        Returns:
            Finalized predictions with same shape structure as targets.
        """
        # 4D path delegates composer internally after per-node flattening
        if src.dim() == 4:
            raw = self._forward_graph_4d(
                src, targets=targets, time_features=time_features, epoch=epoch
            )
            return self._finalize_output(raw)

        # 3D path
        if self.head_composer is not None:
            # Composer owns preprocessing; avoid legacy double-processing
            x_enc, comp_state = self.head_composer.forward_pre(src)  # [B, T, F’]
        else:
            x_enc = self._preprocess_input(src)
            comp_state = None

        # Strategy (RAW)
        raw = self._run_strategy(x_enc, targets, time_features, epoch)

        # Composer inverse BEFORE final postprocess/normalization
        if self.head_composer is not None:
            raw = self.head_composer.inverse_post(raw, comp_state)

        return self._finalize_output(raw)

    # -------------------------------------------------------------------------
    # 4D Graph Forward — returns RAW predictions
    # -------------------------------------------------------------------------
    def _forward_graph_4d(
        self,
        src: torch.Tensor,  # [B, T, N, F]
        targets: Optional[torch.Tensor] = None,  # [B, L, N, D_out]
        time_features: Optional[torch.Tensor] = None,  # [B, T, N, Ft]
        epoch: Optional[int] = None,
        *,
        adj: Optional[torch.Tensor] = None,
        mode: str = "temporal_per_node",
    ) -> torch.Tensor:
        """
        Forward pass for 4D spatiotemporal data.
        Returns RAW predictions (finalization happens in forward()).
        Composer is applied per node-stream after flattening to [B·N, T, F].
        """
        assert src.dim() == 4, "Expected [B, T, N, F]"
        B, T, N, F = src.shape
        if targets is not None and targets.dim() == 3:
            targets = targets.unsqueeze(-1)  # [B, L, N, 1]

        # Graph block BEFORE encoder
        src = self._apply_graph_block(src, adj, "pre_encoder")

        # Graph block POST encoder (kept to match original ordering)
        src = self._apply_graph_block(src, adj, "post_encoder")

        if mode != "temporal_per_node":
            raise NotImplementedError(
                f"mode={mode} not implemented. Use 'temporal_per_node'."
            )

        # Flatten nodes to batch
        src_3d = src.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B·N, T, F]

        # Targets/time features flattened the same way
        if targets is not None:
            L = targets.size(1)
            D_out = targets.size(-1)
            targets_3d = targets.permute(0, 2, 1, 3).reshape(B * N, L, D_out)
        else:
            targets_3d = None

        if time_features is not None:
            Ft = time_features.size(-1)
            time_3d = time_features.permute(0, 2, 1, 3).reshape(B * N, T, Ft)
        else:
            time_3d = None

        # Composer (if present) owns preprocessing here
        if self.head_composer is not None:
            x_enc, comp_state = self.head_composer.forward_pre(src_3d)  # [B·N, T, F’]
        else:
            x_enc = self._preprocess_input(src_3d)
            comp_state = None

        # Temporal strategy on flattened streams (RAW)
        out_3d = self._run_strategy(x_enc, targets_3d, time_3d, epoch)

        # Composer inverse (if present) BEFORE reshaping back
        if self.head_composer is not None:
            out_3d = self.head_composer.inverse_post(out_3d, comp_state)

        # Back to [B, L, N, D_out]
        L = out_3d.size(1)
        D_out = out_3d.size(-1)
        out_4d = out_3d.reshape(B, N, L, D_out).permute(0, 2, 1, 3).contiguous()

        # Graph block AFTER decoder
        out_4d = self._apply_graph_block(out_4d, adj, "post_decoder")

        return out_4d  # RAW — finalized once in forward()

    # -------------------------------------------------------------------------
    # Forward strategies — all return RAW predictions (no finalize inside)
    # -------------------------------------------------------------------------
    def _forward_direct(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Direct prediction strategy (no autoregression)."""
        return self.head(src)  # RAW

    def _forward_autoregressive(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive prediction strategy."""
        if self.decoder is None:
            raise RuntimeError("Autoregressive strategy requires a decoder.")
        decoder_input = self.dec_init_proj(src[:, -1:, :])

        def step_fn(x_in: torch.Tensor, _: int) -> torch.Tensor:
            out = self.decoder(x_in)
            return self.output_head(out)  # map to output dims; keep RAW

        return self._rollout_decoder(
            decoder_input=decoder_input,
            steps=self.target_len,
            targets=targets,
            epoch=epoch,
            step_fn=step_fn,
            aggregate="cat",
        )  # RAW

    def _forward_seq2seq(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Sequence-to-sequence prediction strategy."""
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Seq2seq strategy requires both encoder and decoder.")
        strategy = {
            "informer-like": self._forward_informer_style,
            "transformer": self._forward_transformer_style,
        }.get(self.model_type, self._forward_rnn_style)
        return strategy(src, targets, time_features, epoch)  # RAW

    def _forward_rnn_style(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """RNN-style seq2seq forward pass."""
        enc_out, enc_hidden = self.encoder(src)
        dec_hidden = self._prepare_decoder_hidden(enc_hidden)
        decoder_input = self.dec_init_proj(src[:, -1:, :])

        def step_fn(x_in: torch.Tensor, _: int) -> torch.Tensor:
            nonlocal dec_hidden
            dec_out, dec_hidden = self.decoder(x_in, dec_hidden)
            if self.use_attention:
                context, _ = self.attention_module(dec_hidden, enc_out)
                dec_out = torch.cat([dec_out, context], dim=-1)
            return self.output_head(dec_out)  # RAW mapping

        return self._rollout_decoder(
            decoder_input=decoder_input,
            steps=self.target_len,
            targets=targets,
            epoch=epoch,
            step_fn=step_fn,
            select_fn=lambda out: out.squeeze(1) if out.dim() == 3 else out,
            aggregate="stack",
        )  # RAW

    def _forward_transformer_style(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Transformer-style seq2seq forward pass."""
        device = src.device
        memory = self._encode_memory(src, time_features=time_features)

        decoder_input = (
            src[:, -self.label_len :, :]
            if (self.label_len and self.label_len > 0)
            else src[:, -1:, :]
        )
        decoder_input = self.dec_init_proj(decoder_input)
        prompt_len = decoder_input.size(1)

        outputs = []
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.pred_len):
            L_tgt = decoder_input.size(1)
            tgt_mask = self._causal_mask(L_tgt, device=device)
            mtp_base = None
            if self.training and (targets is not None):
                n_future = max(0, L_tgt - prompt_len)
                if n_future > 0:
                    gt_now = self.dec_feedback_proj(targets[:, :n_future, :])
                    mtp_base = torch.cat(
                        [decoder_input[:, :prompt_len, :], gt_now], dim=1
                    )
                else:
                    mtp_base = decoder_input
            try:
                out = self.decoder(
                    decoder_input,
                    memory,
                    tgt_mask=tgt_mask,
                    mtp_targets=mtp_base,
                )
            except TypeError:
                out = self.decoder(decoder_input, memory)
            out = self.output_head(out)  # RAW mapping
            outputs.append(out[:, -1:, :] if out.size(1) > 1 else out)

            if t < self.pred_len - 1:
                next_in = self._next_decoder_input(outputs[-1], targets, t, use_tf)
                next_in = self.dec_feedback_proj(next_in)
                decoder_input = torch.cat([decoder_input, next_in], dim=1)

        return torch.cat(outputs, dim=1)  # RAW

    def _forward_informer_style(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        *,
        teacher_forcing: str = "none",  # "none" | "prefix" | "scheduled"
        teacher_forcing_k: int = 0,  # used when teacher_forcing == "prefix"
    ) -> torch.Tensor:
        """
        Informer-style seq2seq forward pass with optional teacher forcing
        on the decoder inputs.

        Layout:
          decoder_input = concat( last label_len of src , horizon segment )
          - horizon segment is zeros by default
          - with teacher forcing (training only), selected horizon positions
            are replaced by the *ground truth at the same positions* (no
            extra shift), since the last label_len from src already precedes
            the horizon.

        Returns:
          dec_out: [B, pred_len, d_model] (or already-projected depending on
          your decoder/head).
        """
        B, L_enc, F_in = src.shape
        L_label = int(self.label_len)
        H = int(self.pred_len)
        device, dtype = src.device, src.dtype

        # Clamp to avoid slicing issues
        L_label = max(0, min(L_label, L_enc))
        L_dec = L_label + H

        # ----- Encoder -----
        enc_out = self._encode_memory(src, time_features=time_features)

        # ----- Build decoder features: [known label] + [future zeros] -----
        dec_known_feats = (
            src[:, -L_label:, :] if L_label > 0 else src[:, :0, :]
        )  # [B, L_label, F_in]
        dec_future_feats = torch.zeros(
            B, H, F_in, device=device, dtype=dtype
        )  # [B, H, F_in]

        # Project known part
        dec_known_emb = self.dec_init_proj(dec_known_feats)  # [B, L_label, d_model]

        # Prepare future segment default (zeros) in d_model
        zeros_emb = self.dec_init_proj(dec_future_feats)  # [B, H, d_model]

        # ----- Teacher forcing over the horizon (no last_src_step, direct fill) -----
        use_tf = self.training and teacher_forcing != "none" and targets is not None
        tgt_full = None
        if use_tf:
            # Ensure targets cover H steps (pad if shorter)
            tgt_full = self._targets_to_horizon(
                targets,
                H,
                batch_size=B,
                device=targets.device,
                dtype=targets.dtype,
            )

            # Project GT to d_model (if output features differ from input features,
            # provide a dedicated proj)
            gt_emb = self.dec_init_proj(tgt_full[:, :H, :])  # [B, H, d_model]

            # Choose which horizon positions to force
            if teacher_forcing == "prefix":
                k = max(0, min(int(teacher_forcing_k), H))
                force_mask = torch.zeros(B, H, dtype=torch.bool, device=device)
                if k > 0:
                    force_mask[:, :k] = True
            elif teacher_forcing == "scheduled":
                ratio = (
                    self.scheduled_sampling_fn(epoch)
                    if (self.scheduled_sampling_fn and epoch is not None)
                    else self.teacher_forcing_ratio
                )
                p = float(max(0.0, min(1.0, ratio)))
                force_mask = torch.rand(B, H, device=device) < p  # [B, H]
            else:
                force_mask = torch.zeros(B, H, dtype=torch.bool, device=device)

            # Mix: GT where forced, zeros elsewhere
            dec_future_emb = torch.where(
                force_mask.unsqueeze(-1), gt_emb, zeros_emb
            )  # [B, H, d_model]
        else:
            dec_future_emb = zeros_emb
            force_mask = torch.zeros(B, H, dtype=torch.bool, device=device)

        mtp_base = None
        if self.training and (targets is not None):
            if tgt_full is None:
                tgt_full = self._targets_to_horizon(
                    targets,
                    H,
                    batch_size=B,
                    device=targets.device,
                    dtype=targets.dtype,
                )
            gt_future_in = self.dec_feedback_proj(tgt_full[:, :H, :])  # [B, H, dec_in]
            mtp_base = torch.cat(
                [dec_known_emb, gt_future_in], dim=1
            )  # [B, L_dec, dec_in]

        # Final decoder input
        dec_input = torch.cat(
            [dec_known_emb, dec_future_emb], dim=1
        )  # [B, L_dec, d_model]

        # ----- Optional decoder time features -----
        dec_time = None
        if isinstance(time_features, dict) and "dec" in time_features:
            dec_time = time_features["dec"]  # [B, L_dec, ...]
        elif time_features is not None and not isinstance(time_features, dict):
            tf = time_features
            if tf.size(1) >= L_enc:
                tf_known = tf[:, -L_label:, ...] if L_label > 0 else tf[:, :0, ...]
                tf_zeros = torch.zeros(
                    B, H, *tf.shape[2:], device=device, dtype=tf.dtype
                )
                dec_time = torch.cat([tf_known, tf_zeros], dim=1)  # [B, L_dec, *]

        # ----- Masks -----
        # Causal mask over the entire decoder sequence (label + horizon)
        tgt_mask = self._causal_mask(L_dec, device=device)

        # K/V padding mask:
        #   - known label segment: valid (False)
        #   - horizon segment: mask only positions that remain zeros
        #     (i.e., NOT teacher-forced)
        if H > 0:
            horiz_placeholder = ~force_mask  # True -> mask K/V
            kpm_known = torch.zeros(B, L_label, dtype=torch.bool, device=device)
            tgt_key_padding_mask = torch.cat(
                [kpm_known, horiz_placeholder], dim=1
            )  # [B, L_dec]
        else:
            tgt_key_padding_mask = torch.zeros(
                B, L_dec, dtype=torch.bool, device=device
            )

        # ----- Decode -----
        try:
            dec_out_full = self.decoder(
                dec_input,
                enc_out,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None,
                time_features=dec_time,
                mtp_targets=mtp_base,
            )
        except TypeError:
            try:
                dec_out_full = self.decoder(
                    dec_input,
                    enc_out,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
            except TypeError:
                dec_out_full = self.decoder(dec_input, enc_out)

        # Keep only the forecast window
        dec_out = dec_out_full[:, -H:, :]  # [B, H, d_model]
        return dec_out
