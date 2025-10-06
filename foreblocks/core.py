from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

# =============================================================================
# Graph utilities
# =============================================================================

def _ensure_batch_adj(adj: Optional[torch.Tensor], B: int) -> Optional[torch.Tensor]:
    """
    adj: [N,N] or [B,N,N] or None  ->  [B,N,N] or None
    """
    if adj is None:
        return None
    if adj.dim() == 2:
        N = adj.size(0)
        return adj.unsqueeze(0).expand(B, N, N).contiguous()
    return adj  # already [B,N,N]


class TimewiseGraph(nn.Module):
    """
    Adapter: lift any graph block that consumes [B, N, F] (+ adj)
    to work on [B, T, N, F] by applying it independently per time step.
    If x is already [B,N,F], just forwards.
    """
    def __init__(self, gblock: nn.Module):
        super().__init__()
        self.gblock = gblock
        # mark capability to avoid re-wrapping
        self._accepts_time_dim = True

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3:  # [B,N,F]
            return self.gblock(x, adj)
        assert x.dim() == 4, "TimewiseGraph expects [B,N,F] or [B,T,N,F]"
        B, T, N, F = x.shape
        A = _ensure_batch_adj(adj, B)
        ys = []
        for t in range(T):
            xt = x[:, t, :, :]                  # [B,N,F]
            yt = self.gblock(xt, A)             # [B,N,F’]
            ys.append(yt.unsqueeze(1))
        return torch.cat(ys, dim=1)             # [B,T,N,F’]


def _apply_graph_block(x: torch.Tensor,
                       adj: Optional[torch.Tensor],
                       block: Optional[nn.Module]) -> torch.Tensor:
    """
    Apply a registered graph block that supports (x, adj)
    where x is [B,N,F] or [B,T,N,F], adj is [N,N] or [B,N,N] or None.
    Ensures device/dtype alignment.
    """
    if block is None:
        return x

    # align adj to batch/device/dtype of x
    B = x.size(0)
    if adj is not None:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, adj.size(0), adj.size(1)).contiguous()
        adj = adj.to(device=x.device, dtype=x.dtype)

    # align block to x
    params = list(block.parameters(recurse=True))
    if params and (params[0].device != x.device or params[0].dtype != x.dtype):
        block.to(device=x.device, dtype=x.dtype)

    return block(x, adj)
# -----------------------------
# Graph + Per-Node Core Wrapper
# -----------------------------
class BaseHead(nn.Module):
    """Base class for all heads in the forecasting model."""

    def __init__(self, module: nn.Module, name: str = None):
        super().__init__()
        self.module = module
        self.name = name or self.__class__.__name__

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_aux_loss(self) -> torch.Tensor:
        if hasattr(self.module, "aux_loss"):
            aux = self.module.aux_loss
            return aux if torch.is_tensor(aux) else torch.tensor(aux)
        return torch.zeros(())

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

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
    ):
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
        self.pred_len = target_len  # alias
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
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
        if self.strategy != "direct":
            if self.output_size is None:
                raise ValueError(
                    "For non-direct strategies, provide output_size or a decoder with .output_size"
                )

        # label_len is a prompt length for transformer decoding
        self.label_len = label_len

        # Aux
        self._kl = None  # kept for API compatibility (unused here)
        self._mem_model_bridge: Optional[nn.Module] = None

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

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def add_head(self, head, position: str = "input", name: str = None):
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

    def remove_head(self, position: str):
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
        heads = {}
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
        x: [B, T, N, F] or [B, N, F]; adj: [N,N] or [B,N,N]
        """
        assert where in {"pre_encoder", "post_encoder", "post_decoder"}

        # Wrap once to operate timewise if needed
        if not getattr(graph_block, "_accepts_time_dim", False):
            graph_block = TimewiseGraph(graph_block)
            graph_block._accepts_time_dim = True

        # Move to the same device/dtype as the model
        try:
            ref_param = next(self.parameters())
            device = ref_param.device
            dtype  = ref_param.dtype
        except StopIteration:
            # Fallback if model has no params yet
            device, dtype = torch.device("cpu"), None

        if dtype is not None:
            graph_block.to(device=device, dtype=dtype)
        else:
            graph_block.to(device=device)

        self._graph_blocks[where] = graph_block
        return self

    def get_aux_loss(self) -> torch.Tensor:
        device = (
            next(self.parameters()).device
            if any(p.requires_grad for p in self.parameters())
            else torch.device("cpu")
        )
        loss = torch.zeros((), device=device)

        for head_name in [
            "input_preprocessor",
            "output_postprocessor",
            "input_normalization",
            "output_normalization",
        ]:
            head = getattr(self, head_name, None)
            if isinstance(head, BaseHead):
                loss = loss + head.get_aux_loss().to(device)

        if hasattr(self, "encoder") and hasattr(self.encoder, "aux_loss"):
            aux = self.encoder.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        if hasattr(self, "decoder") and hasattr(self.decoder, "aux_loss"):
            aux = self.decoder.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        if hasattr(self, "aux_loss"):
            aux = self.aux_loss
            loss = loss + (aux if torch.is_tensor(aux) else loss.new_tensor(aux))

        return loss

    def get_kl(self):
        # Kept for API compatibility (not used here)
        return self._kl

    def get_model_size(self):
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

    def benchmark_inference(self, input_tensor, num_runs=100, warmup_runs=10):
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
            "avg_inference_time_ms": avg * 1000,
            "throughput_samples_per_sec": 1.0 / avg if avg > 0 else float("inf"),
            "device": str(device),
        }

    def attribute_forward(
        self, src, time_features=None, targets=None, epoch=None, output_idx=None
    ):
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
        if isinstance(module, BaseHead):
            return module
        return BaseHead(module, name)

    def _setup_output_layers(self):
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
        self.output_head = nn.Sequential(
            nn.Linear(out_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.output_size),
        )
        if self.input_size and self.input_size != self.output_size:
            self.project_output = nn.Linear(self.input_size, self.output_size)
        else:
            self.project_output = nn.Identity()

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _preprocess_input(self, src):
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _finalize_output(self, x):
        # Apply any final normalization/postprocessing exactly once.
        return self.output_postprocessor(self.output_normalization(x))

    def _next_decoder_input(self, output, targets, t, use_tf):
        if use_tf and targets is not None:
            return targets[:, t : t + 1, :]
        return output if output.dim() == 3 else output.unsqueeze(1)

    def _should_use_teacher_forcing(
        self, targets=None, epoch=None, fallback_device="cpu"
    ):
        if (not self.training) or (targets is None):
            return False
        if self._is_fx_tracing():
            return False
        ratio = (
            self.scheduled_sampling_fn(epoch)
            if (self.scheduled_sampling_fn and epoch is not None)
            else self.teacher_forcing_ratio
        )
        device = getattr(targets, "device", torch.device(fallback_device))
        return torch.rand((1,), device=device).item() < ratio

    def _prepare_decoder_hidden(self, encoder_hidden):
        """No VAE: pass encoder_hidden through (merge bidirectional if needed)."""
        if self.encoder is None or not getattr(self.encoder, "bidirectional", False):
            return encoder_hidden
        # Bidirectional RNN handling
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            return (self._merge_bidirectional(h_n), self._merge_bidirectional(c_n))
        return self._merge_bidirectional(encoder_hidden)

    def _merge_bidirectional(self, hidden: torch.Tensor) -> torch.Tensor:
        assert hidden.size(0) % 2 == 0, (
            "Expected even number of layers for bidirectional RNN"
        )
        num_layers = hidden.size(0) // 2
        reshaped = hidden.reshape(num_layers, 2, *hidden.shape[1:])
        return reshaped.sum(dim=1)

    def _get_attention_query(self, decoder_output, decoder_hidden):
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
    def _is_fx_tracing():
        try:
            import torch.fx
            return torch.fx._symbolic_trace.is_fx_tracing()
        except Exception:
            return False

    def _disable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

    def _decoder_model_dim(self) -> Optional[int]:
        if self.decoder is None:
            return None
        for name in ("d_model", "model_dim", "embed_dim", "dim", "hidden_dim"):
            if hasattr(self.decoder, name) and isinstance(
                getattr(self.decoder, name), int
            ):
                return getattr(self.decoder, name)
        return None

    def _align_memory_to_decoder(self, memory: torch.Tensor) -> torch.Tensor:
        d_model = self._decoder_model_dim()
        if d_model is None or memory is None or memory.size(-1) == d_model:
            return memory
        if self._mem_model_bridge is None:
            self._mem_model_bridge = nn.Linear(memory.size(-1), d_model).to(
                memory.device
            )
        return self._mem_model_bridge(memory)

    def _supports_time(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "supports_time_features")
            or "time" in " ".join(dir(module)).lower()
        )

    # -------------------------------------------------------------------------
    # Forward (Public) — finalize exactly once here
    # -------------------------------------------------------------------------
    def forward(self, src, targets=None, time_features=None, epoch=None):
        x = self._preprocess_input(src)

        # 4D graph dispatch [B, T, N, F]
        if x.dim() == 4:
            raw = self._forward_graph_4d(
                x, targets=targets, time_features=time_features, epoch=epoch
            )
            return self._finalize_output(raw)

        # 2D/3D pathways
        strategy_fn = {
            "direct": self._forward_direct,
            "autoregressive": self._forward_autoregressive,
            "seq2seq": self._forward_seq2seq,
            "transformer_seq2seq": self._forward_seq2seq,
        }[self.strategy]
        raw = strategy_fn(x, targets, time_features, epoch)
        return self._finalize_output(raw)

    # -------------------------------------------------------------------------
    # 4D Graph Forward — returns RAW predictions
    # -------------------------------------------------------------------------
    def _forward_graph_4d(
        self,
        src: torch.Tensor,              # [B, T, N, F]
        targets: Optional[torch.Tensor] = None,   # [B, L, N, D_out]
        time_features: Optional[torch.Tensor] = None,  # [B, T, N, Ft]
        epoch: Optional[int] = None,
        *,
        adj: Optional[torch.Tensor] = None,
        mode: str = "temporal_per_node",
    ):
        assert src.dim() == 4, "Expected [B, T, N, F]"
        B, T, N, F = src.shape
        if targets is not None and targets.dim() == 3:
            targets = targets.unsqueeze(-1)  # [B, L, N, 1]

        # Graph block BEFORE encoder
        pre_gb  = self._graph_blocks.get("pre_encoder",  None)
        src     = _apply_graph_block(src, adj, pre_gb)

        # Graph block POST encoder (but before decoder)
        postenc_gb = self._graph_blocks.get("post_encoder", None)
        src        = _apply_graph_block(src, adj, postenc_gb)

        if mode == "temporal_per_node":
            # Flatten nodes to batch
            B, T, N, F = src.shape  # (F may have changed)
            src_3d = src.permute(0, 2, 1, 3).reshape(B * N, T, F)

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

            # Route through selected temporal strategy (RAW return)
            if self.strategy in ("seq2seq", "transformer_seq2seq"):
                out_3d = self._forward_seq2seq(
                    src_3d, targets=targets_3d, time_features=time_3d, epoch=epoch
                )
            elif self.strategy == "autoregressive":
                out_3d = self._forward_autoregressive(
                    src_3d, targets=targets_3d, time_features=time_3d, epoch=epoch
                )
            elif self.strategy == "direct":
                out_3d = self._forward_direct(
                    src_3d, targets=targets_3d, time_features=time_3d, epoch=epoch
                )
            else:
                raise ValueError(f"Unsupported strategy for 4D: {self.strategy}")

            # Back to [B, L, N, D_out]
            L = out_3d.size(1)
            D_out = out_3d.size(-1)
            out_4d = out_3d.reshape(B, N, L, D_out).permute(0, 2, 1, 3).contiguous()
        else:
            raise NotImplementedError(f"mode={mode} not implemented. Use 'temporal_per_node'.")

        # Graph block AFTER decoder
        post_gb = self._graph_blocks.get("post_decoder", None)
        out_4d  = _apply_graph_block(out_4d, adj, post_gb)

        return out_4d  # RAW — finalized once in forward()

    # -------------------------------------------------------------------------
    # Forward strategies — all return RAW predictions (no finalize inside)
    # -------------------------------------------------------------------------
    def _forward_direct(self, src, targets=None, time_features=None, epoch=None):
        return self.head(src)  # RAW

    def _forward_autoregressive(
        self, src, targets=None, time_features=None, epoch=None
    ):
        if self.decoder is None:
            raise RuntimeError("Autoregressive strategy requires a decoder.")
        outputs = []
        decoder_input = self.dec_init_proj(src[:, -1:, :])
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            out = self.decoder(decoder_input)
            out = self.output_head(out)  # map to output dims; keep RAW
            outputs.append(out)
            if t < self.target_len - 1:
                next_in = self._next_decoder_input(out, targets, t, use_tf)
                decoder_input = self.dec_feedback_proj(next_in)

        return torch.cat(outputs, dim=1)  # RAW

    def _forward_seq2seq(self, src, targets=None, time_features=None, epoch=None):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Seq2seq strategy requires both encoder and decoder.")
        strategy = {
            "informer-like": self._forward_informer_style,
            "transformer": self._forward_transformer_style,
        }.get(self.model_type, self._forward_rnn_style)
        return strategy(src, targets, time_features, epoch)  # RAW

    def _forward_rnn_style(self, src, targets=None, time_features=None, epoch=None):
        enc_out, enc_hidden = self.encoder(src)
        dec_hidden = self._prepare_decoder_hidden(enc_hidden)
        decoder_input = self.dec_init_proj(src[:, -1:, :])
        use_tf = self._should_use_teacher_forcing(targets, epoch)
        outputs = []

        for t in range(self.target_len):
            dec_out, dec_hidden = self.decoder(decoder_input, dec_hidden)
            if self.use_attention:
                context, _ = self.attention_module(dec_hidden, enc_out)
                dec_out = torch.cat([dec_out, context], dim=-1)
            dec_out = self.output_head(dec_out)  # RAW mapping
            outputs.append(dec_out.squeeze(1) if dec_out.dim() == 3 else dec_out)
            if t < self.target_len - 1:
                next_in = self._next_decoder_input(dec_out, targets, t, use_tf)
                decoder_input = self.dec_feedback_proj(next_in)

        return torch.stack(outputs, dim=1)  # RAW

    def _forward_transformer_style(
        self, src, targets=None, time_features=None, epoch=None
    ):
        device = src.device
        memory = (
            self.encoder(src, time_features=time_features)
            if time_features is not None
            else self.encoder(src)
        )
        memory = self._align_memory_to_decoder(
            memory[0] if isinstance(memory, tuple) else memory
        )

        decoder_input = (
            src[:, -self.label_len :, :]
            if (self.label_len and self.label_len > 0)
            else src[:, -1:, :]
        )
        decoder_input = self.dec_init_proj(decoder_input)

        outputs = []
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.pred_len):
            L_tgt = decoder_input.size(1)
            tgt_mask = torch.triu(
                torch.ones(L_tgt, L_tgt, device=device, dtype=torch.bool), diagonal=1
            )
            try:
                out = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)
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
        self, src, targets=None, time_features=None, epoch=None
    ):
        batch_size = src.size(0)
        enc = self.encoder(src, time_features=time_features)
        enc_out = enc[0] if isinstance(enc, tuple) else enc
        enc_out = self._align_memory_to_decoder(enc_out)

        start_token = self.dec_init_proj(src[:, -1:, :])
        dec_input = start_token.expand(batch_size, self.pred_len, -1)
        dec_out = self.decoder(dec_input, enc_out)
        return self.output_head(dec_out)  # RAW
