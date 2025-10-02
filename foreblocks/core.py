import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import *

# Optional: import your heads from wherever you defined them.
# If you don't want imports here, pass an instance via `encoder_only_head=...`
try:
    from .heads import (
        DLinearHead,  # adjust path as needed
        HorizonMLPHead,
        TemporalLinearHead,
    )
except Exception:
    HorizonMLPHead = TemporalLinearHead = DLinearHead = None


class BaseForecastingModel(nn.Module):
    """
    Modular and extensible forecasting model for time series tasks.
    """

    VALID_STRATEGIES = [
        "seq2seq",
        "autoregressive",
        "direct",
        "transformer_seq2seq",
    ]
    VALID_MODEL_TYPES = [
        "lstm",
        "transformer",
        "informer-like",
    ]

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        target_len: int = 5,
        forecasting_strategy: str = "seq2seq",
        model_type: str = "lstm",
        # Processing modules
        input_preprocessor: nn.Module = None,
        output_postprocessor: nn.Module = None,
        input_normalization: nn.Module = None,
        output_normalization: nn.Module = None,
        output_block: nn.Module = None,
        input_skip_connection: bool = False,
        # Architecture options
        attention_module: nn.Module = None,
        output_size: int = None,
        hidden_size: int = 64,
        multi_encoder_decoder: bool = False,
        input_processor_output_size: int = 16,
        # Training
        teacher_forcing_ratio: float = 0.5,
        scheduled_sampling_fn: Optional[Callable] = None,
        # Time embeddings
        time_feature_embedding_enc: nn.Module = None,
        time_feature_embedding_dec: nn.Module = None,
        # Decoder prompt length for transformer-style decoding
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
        self.multi_encoder_decoder = multi_encoder_decoder

        # === Processing Blocks ===
        self.input_preprocessor = input_preprocessor or nn.Identity()
        self.output_postprocessor = output_postprocessor or nn.Identity()
        self.input_normalization = input_normalization or nn.Identity()
        self.output_normalization = output_normalization or nn.Identity()
        self.output_block = output_block or nn.Identity()

        # === Time Embeddings ===
        self.time_feature_embedding_enc = time_feature_embedding_enc
        self.time_feature_embedding_dec = time_feature_embedding_dec

        # === Attention ===
        self.use_attention = attention_module is not None
        self.attention_module = attention_module

        # === Architecture Setup ===
        self._setup_architecture(encoder, decoder, input_processor_output_size)

        # Infer I/O sizes from encoder/decoder if available
        self.input_size = getattr(encoder, "input_size", None) if encoder else None
        self.output_size = output_size or (getattr(decoder, "output_size", None) if decoder else None)

        # label_len is a *prompt length* for transformer decoding, not an output dim.
        self.label_len = label_len

        # Aux loss aggregator (for MoE etc.)
        self._kl = None
        self._mem_model_bridge: Optional[nn.Module] = None

        # === Decoder input/feedback projection bridges ===
        self._decoder_input_size = getattr(self.decoder, "input_size", self.input_size or self.output_size)
        if self.input_size and self._decoder_input_size and self.input_size != self._decoder_input_size:
            self.dec_init_proj = nn.Linear(self.input_size, self._decoder_input_size)
        else:
            self.dec_init_proj = nn.Identity()
        if self.output_size and self._decoder_input_size and self.output_size != self._decoder_input_size:
            self.dec_feedback_proj = nn.Linear(self.output_size, self._decoder_input_size)
        else:
            self.dec_feedback_proj = nn.Identity()

        # === Encoder-only configuration ===
        self._setup_output_layers()

    # === Public API ===

    def get_aux_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")
        loss = torch.zeros((), device=device)

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

    def forward(self, src, targets=None, time_features=None, epoch=None):
        x = self._preprocess_input(src)
        # Dispatch
        key = self.strategy
        strategy_fn = {
            "direct": self._forward_direct,
            "autoregressive": self._forward_autoregressive,
            "seq2seq": self._forward_seq2seq,
            "transformer_seq2seq": self._forward_seq2seq,
        }[key]
        return strategy_fn(x, targets, time_features, epoch)

    def get_kl(self):
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

    def attribute_forward(self, src, time_features=None, targets=None, epoch=None, output_idx=None):
        self.train()
        self._disable_dropout()
        src = src.requires_grad_()
        out = self.forward(src, targets=targets, time_features=time_features, epoch=epoch)
        return out[..., output_idx] if output_idx is not None else out

    # === Private / setup ===

    def _setup_architecture(self, encoder, decoder, input_processor_output_size):
        if self.multi_encoder_decoder:
            self.encoder = nn.ModuleList([encoder for _ in range(input_processor_output_size)])
            self.decoder = nn.ModuleList([decoder for _ in range(input_processor_output_size)])
            self.decoder_aggregator = nn.Linear(input_processor_output_size, 1, bias=False)
        else:
            self.encoder = encoder
            self.decoder = decoder

    def _setup_output_layers(self):
        if not self.encoder or not self.decoder:
            self.output_head = nn.Identity()
            self.project_output = nn.Identity()
            return

        encoder_hidden = self._get_attr(self.encoder, "hidden_size", self.hidden_size)
        decoder_hidden = self._get_attr(self.decoder, "hidden_size", self.hidden_size)
        if self.output_size is None:
            raise ValueError("output_size must be provided or available on decoder.output_size")

        out_dim = (decoder_hidden + encoder_hidden) if self.use_attention else decoder_hidden
        self.output_head = nn.Sequential(
            nn.Linear(out_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, self.output_size),
        )
        if self.input_size and self.input_size != self.output_size:
            self.project_output = nn.Linear(self.input_size, self.output_size)
        else:
            self.project_output = nn.Identity()

    def _get_attr(self, module, attr, fallback):
        return getattr(module[0] if isinstance(module, nn.ModuleList) else module, attr, fallback)

    # === Core helpers ===

    def _preprocess_input(self, src):
        processed = self.input_preprocessor(src)
        if self.input_skip_connection:
            processed = processed + src
        return self.input_normalization(processed)

    def _finalize_output(self, x):
        return self.output_postprocessor(self.output_normalization(x))

    def _next_decoder_input(self, output, targets, t, use_tf):
        if use_tf and targets is not None:
            return targets[:, t : t + 1, :]
        return output if output.dim() == 3 else output.unsqueeze(1)

    def _should_use_teacher_forcing(self, targets=None, epoch=None, fallback_device="cpu"):
        if (not self.training) or (targets is None):
            return False
        if self._is_fx_tracing():
            return False
        ratio = (
            self.scheduled_sampling_fn(epoch)
            if self.scheduled_sampling_fn and epoch is not None
            else self.teacher_forcing_ratio
        )
        device = getattr(targets, "device", torch.device(fallback_device))
        return torch.rand((1,), device=device).item() < ratio

    def _extract_latent_state(self, encoder_hidden) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # VAE case: (z, mu, logvar)
        if isinstance(encoder_hidden, tuple) and len(encoder_hidden) == 3:
            z, mu, logvar = encoder_hidden
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (z,), kl_div
        return self._prepare_decoder_hidden(encoder_hidden), None

    def _prepare_decoder_hidden(self, encoder_hidden):
        if not getattr(self.encoder, "bidirectional", False):
            return encoder_hidden
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n, c_n = encoder_hidden
            return (self._merge_bidirectional(h_n), self._merge_bidirectional(c_n))
        return self._merge_bidirectional(encoder_hidden)

    def _merge_bidirectional(self, hidden: torch.Tensor) -> torch.Tensor:
        assert hidden.size(0) % 2 == 0, "Expected even number of layers for bidirectional RNN"
        num_layers = hidden.size(0) // 2
        reshaped = hidden.reshape(num_layers, 2, *hidden.shape[1:])
        return reshaped.sum(dim=1)

    def _get_attention_query(self, decoder_output, decoder_hidden):
        if hasattr(self.decoder, "is_transformer") and getattr(self.decoder, "is_transformer"):
            return decoder_hidden.permute(1, 0, 2)  # (batch, seq_len, hidden)
        return (decoder_hidden[0][-1] if isinstance(decoder_hidden, tuple) else decoder_hidden[-1])

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
        for name in ("d_model", "model_dim", "embed_dim", "dim", "hidden_dim"):
            if hasattr(self.decoder, name) and isinstance(getattr(self.decoder, name), int):
                return getattr(self.decoder, name)
        return None

    def _align_memory_to_decoder(self, memory: torch.Tensor) -> torch.Tensor:
        d_model = self._decoder_model_dim()
        if d_model is None or memory is None or memory.size(-1) == d_model:
            return memory
        if self._mem_model_bridge is None:
            self._mem_model_bridge = nn.Linear(memory.size(-1), d_model).to(memory.device)
        return self._mem_model_bridge(memory)

    def _supports_time(self, module: nn.Module) -> bool:
        # Heuristic: check signature by attribute or a flag you use in your encoders
        return hasattr(module, "supports_time_features") or "time" in " ".join(dir(module)).lower()

    # === Forward strategies ===

    def _forward_direct(self, src, targets=None, time_features=None, epoch=None):
        out = self.decoder(src)
        return self._finalize_output(out)

    def _forward_autoregressive(self, src, targets=None, time_features=None, epoch=None):
        outputs = []
        decoder_input = self.dec_init_proj(src[:, -1:, :])
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            out = self.decoder(decoder_input)  # [B, 1, *]
            out = self._finalize_output(out)
            outputs.append(out)

            if t < self.target_len - 1:
                next_in = self._next_decoder_input(out, targets, t, use_tf)  # [B,1,output_size]
                decoder_input = self.dec_feedback_proj(next_in)

        return self.output_postprocessor(torch.cat(outputs, dim=1))

    def _forward_seq2seq(self, src, targets=None, time_features=None, epoch=None):
        if self.multi_encoder_decoder:
            return self._forward_multi_encoder_decoder(src, targets, epoch)

        strategy = {
            "informer-like": self._forward_informer_style,
            "transformer": self._forward_transformer_style,
        }.get(self.model_type, self._forward_rnn_style)

        return strategy(src, targets, time_features, epoch)

    def _forward_rnn_style(self, src, targets=None, time_features=None, epoch=None):
        outputs = []
        enc_out, enc_hidden = self.encoder(src)
        dec_hidden, kl = self._extract_latent_state(enc_hidden)
        self._kl = kl

        decoder_input = self.dec_init_proj(src[:, -1:, :])
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.target_len):
            dec_out, dec_hidden = self.decoder(decoder_input, dec_hidden)

            if self.use_attention:
                context, _ = self.attention_module(dec_hidden, enc_out)
                dec_out = torch.cat([dec_out, context], dim=-1)

            dec_out = self._finalize_output(self.output_head(dec_out))
            outputs.append(dec_out.squeeze(1) if dec_out.dim() == 3 else dec_out)

            if t < self.target_len - 1:
                next_in = self._next_decoder_input(dec_out, targets, t, use_tf)
                decoder_input = self.dec_feedback_proj(next_in)

        return self.output_postprocessor(torch.stack(outputs, dim=1))

    def _forward_transformer_style(self, src, targets=None, time_features=None, epoch=None):
        device = src.device
        memory = (
            self.encoder(src, time_features=time_features)
            if time_features is not None
            else self.encoder(src)
        )
        memory = self._align_memory_to_decoder(
            memory[0] if isinstance(memory, tuple) else memory
        )

        if self.label_len and self.label_len > 0:
            decoder_input = src[:, -self.label_len :, :]
        else:
            decoder_input = src[:, -1:, :]
        decoder_input = self.dec_init_proj(decoder_input)

        outputs = []
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for t in range(self.pred_len):
            L_tgt = decoder_input.size(1)
            tgt_mask = torch.triu(
                torch.ones(L_tgt, L_tgt, device=device, dtype=torch.bool),
                diagonal=1,
            )  # causal

            try:
                out = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)
            except TypeError:
                out = self.decoder(decoder_input, memory)

            out = self._finalize_output(out)
            outputs.append(out[:, -1:, :] if out.size(1) > 1 else out)

            if t < self.pred_len - 1:
                next_in = self._next_decoder_input(outputs[-1], targets, t, use_tf)
                next_in = self.dec_feedback_proj(next_in)
                decoder_input = torch.cat([decoder_input, next_in], dim=1)

        return torch.cat(outputs, dim=1)

    def _forward_informer_style(self, src, targets=None, time_features=None, epoch=None):
        batch_size = src.size(0)
        enc = self.encoder(src, time_features=time_features)
        enc_out = enc[0] if isinstance(enc, tuple) else enc
        enc_out = self._align_memory_to_decoder(enc_out)

        start_token = self.dec_init_proj(src[:, -1:, :])
        dec_input = start_token.expand(batch_size, self.pred_len, -1)
        dec_out = self.decoder(dec_input, enc_out)
        return self._finalize_output(dec_out)

    def _forward_multi_encoder_decoder(self, src, targets=None, epoch=None):
        batch_size, _, num_inputs = src.shape
        outputs = []
        use_tf = self._should_use_teacher_forcing(targets, epoch)

        for i in range(num_inputs):
            x_i = src[:, :, i : i + 1]
            enc_out, enc_hidden = self.encoder[i](x_i)
            dec_hidden, kl = self._extract_latent_state(enc_hidden)
            self._kl = kl

            dec_input = torch.zeros(batch_size, 1, self.output_size, device=src.device)
            out_i = []

            for t in range(self.target_len):
                dec_out, dec_hidden = self.decoder[i](self.dec_feedback_proj(dec_input), dec_hidden)

                if self.use_attention:
                    query = self._get_attention_query(dec_out, dec_hidden)
                    context, _ = self.attention_module(query, enc_out)
                    dec_out = torch.cat([dec_out, context], dim=-1)

                dec_out = self._finalize_output(self.output_head(dec_out))
                out_i.append(dec_out.squeeze(1) if dec_out.dim() == 3 else dec_out)

                if t < self.target_len - 1:
                    dec_input = self._next_decoder_input(dec_out, targets, t, use_tf)

            outputs.append(torch.stack(out_i, dim=1))

        stacked = torch.stack(outputs, dim=-1)         # [B,T,out,C]
        aggregated = self.decoder_aggregator(stacked).squeeze(-1)  # [B,T,out]
        return self.output_postprocessor(aggregated)


class ForecastingModel(BaseForecastingModel):
    """
    Forecasting model with knowledge distillation support.
    Extends BaseForecastingModel with teacher-student distillation capabilities.
    """

    VALID_DISTILLATION_MODES = [
        "none",
        "output",
        "feature",
        "attention",
        "comprehensive",
    ]

    def __init__(
        self,
        distillation_mode: str = "none",
        teacher_model: Optional[nn.Module] = None,
        distillation_temperature: float = 4.0,
        distillation_alpha: float = 0.7,
        feature_distillation_layers: Optional[List[str]] = None,
        attention_distillation_layers: Optional[List[str]] = None,
        **kwargs,
    ):
        assert distillation_mode in self.VALID_DISTILLATION_MODES, (
            f"Invalid distillation mode: {distillation_mode}"
        )

        super().__init__(**kwargs)

        self.distillation_mode = distillation_mode
        self.teacher_model = teacher_model
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        self.feature_distillation_layers = feature_distillation_layers or []
        self.attention_distillation_layers = attention_distillation_layers or []

        self.feature_hooks = {}
        self.attention_hooks = {}
        self.teacher_features, self.teacher_attentions = {}, {}
        self.student_features, self.student_attentions = {}, {}

        if self.distillation_mode != "none":
            self._setup_distillation()

    # ==================== DISTILLATION SETUP ====================

    def _setup_distillation(self):
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if self.distillation_mode in {"feature", "comprehensive"}:
            self._register_hooks(
                self.feature_distillation_layers,
                self.teacher_features,
                self.student_features,
                self.feature_hooks,
            )

        if self.distillation_mode in {"attention", "comprehensive"}:
            self._register_hooks(
                self.attention_distillation_layers,
                self.teacher_attentions,
                self.student_attentions,
                self.attention_hooks,
                attention=True,
            )

    def _register_hooks(
        self, layer_names, teacher_store, student_store, hook_store, attention=False
    ):
        def create_hook(name, store):
            def hook(module, _, output):
                store[name] = (
                    getattr(module, "attention_weights", output)
                    if attention
                    else output
                )

            return hook

        for name in layer_names:
            if hasattr(self.teacher_model, name):
                hook = getattr(self.teacher_model, name).register_forward_hook(
                    create_hook(name, teacher_store)
                )
                hook_store[f"teacher_{name}"] = hook
            if hasattr(self, name):
                hook = getattr(self, name).register_forward_hook(
                    create_hook(name, student_store)
                )
                hook_store[f"student_{name}"] = hook

    # ==================== DISTILLATION LOSSES ====================

    def _compute_output_distillation_loss(self, student_out, teacher_out):
        if student_out.dtype == torch.float32 and teacher_out.dtype == torch.float32:
            return F.mse_loss(student_out, teacher_out)

        T = self.distillation_temperature
        p_student = F.log_softmax(student_out / T, dim=-1)
        p_teacher = F.softmax(teacher_out / T, dim=-1)
        return F.kl_div(p_student, p_teacher, reduction="batchmean") * (T**2)

    def _compute_feature_distillation_loss(self):
        return self._compute_hooked_loss(
            self.student_features,
            self.teacher_features,
            self.feature_distillation_layers,
            self._align_feature_dimensions,
        )

    def _compute_attention_distillation_loss(self):
        return self._compute_hooked_loss(
            self.student_attentions,
            self.teacher_attentions,
            self.attention_distillation_layers,
            self._align_attention_dimensions,
        )

    def _compute_hooked_loss(self, student_dict, teacher_dict, layer_names, align_fn):
        total_loss, count = 0.0, 0
        for name in layer_names:
            if name not in student_dict or name not in teacher_dict:
                continue

            s_feat, t_feat = student_dict[name], teacher_dict[name]
            s_feat = s_feat[0] if isinstance(s_feat, (tuple, list)) else s_feat
            t_feat = t_feat[0] if isinstance(t_feat, (tuple, list)) else t_feat

            if not isinstance(s_feat, torch.Tensor) or not isinstance(
                t_feat, torch.Tensor
            ):
                continue

            if s_feat.shape != t_feat.shape:
                s_feat = align_fn(s_feat, t_feat.shape)

            total_loss += F.mse_loss(s_feat, t_feat)
            count += 1

        return total_loss / max(count, 1)

    # ==================== DIMENSION ALIGNMENT ====================

    def _align_feature_dimensions(self, student_feat, target_shape):
        if student_feat.shape[-1] != target_shape[-1]:
            proj = nn.Linear(
                student_feat.shape[-1], target_shape[-1], device=student_feat.device
            )
            student_feat = proj(student_feat)

        # Match sequence length
        if len(student_feat.shape) > 1 and student_feat.shape[1] != target_shape[1]:
            student_feat = self._resize_seq(student_feat, target_shape[1])

        return student_feat

    def _resize_seq(self, tensor, target_len):
        if tensor.shape[1] < target_len:
            return F.interpolate(
                tensor.transpose(1, 2), size=target_len, mode="linear"
            ).transpose(1, 2)
        return tensor[:, :target_len, :]

    def _align_attention_dimensions(self, student_att, target_shape):
        if student_att.shape == target_shape:
            return student_att

        B, H_s, L1, L2 = student_att.shape
        B_t, H_t, L1_t, L2_t = target_shape

        # Head alignment
        if H_s != H_t:
            if H_s < H_t:
                repeat_factor = H_t // H_s
                student_att = student_att.repeat(1, repeat_factor, 1, 1)
            else:
                group_size = H_s // H_t
                student_att = student_att.view(B, H_t, group_size, L1, L2).mean(dim=2)

        # Seq length alignment
        if (L1, L2) != (L1_t, L2_t):
            student_att = F.interpolate(
                student_att.view(-1, 1, L1, L2),
                size=(L1_t, L2_t),
                mode="bilinear",
                align_corners=False,
            ).view(B, H_t, L1_t, L2_t)

        return student_att

    # ==================== LOSS COMBINATION ====================

    def _combine_distillation_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        task = losses.get("task_loss", 0.0)
        distill = 0.0
        if "output_distillation" in losses:
            distill += losses["output_distillation"]
        if "feature_distillation" in losses:
            distill += 0.5 * losses["feature_distillation"]
        if "attention_distillation" in losses:
            distill += 0.3 * losses["attention_distillation"]

        return (1 - self.distillation_alpha) * task + self.distillation_alpha * distill

    # ==================== CONTROL ====================

    def set_teacher_model(self, model: nn.Module):
        self.teacher_model = model
        if model is not None:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            if self.distillation_mode != "none":
                self._setup_distillation()

    def enable_distillation(self, mode="output", teacher_model=None):
        assert mode in self.VALID_DISTILLATION_MODES, (
            f"Invalid distillation mode: {mode}"
        )
        self.distillation_mode = mode
        if teacher_model is not None:
            self.set_teacher_model(teacher_model)
        self._setup_distillation()

    def disable_distillation(self):
        self.distillation_mode = "none"
        self.teacher_model = None
        for h in self.feature_hooks.values():
            h.remove()
        for h in self.attention_hooks.values():
            h.remove()
        self.feature_hooks.clear()
        self.attention_hooks.clear()

    def get_distillation_info(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "distillation_enabled": self.distillation_mode != "none",
            "distillation_mode": self.distillation_mode,
            "has_teacher": self.teacher_model is not None,
            "temperature": self.distillation_temperature,
            "alpha": self.distillation_alpha,
            "feature_layers": len(self.feature_distillation_layers),
            "attention_layers": len(self.attention_distillation_layers),
            "active_feature_hooks": len(self.feature_hooks),
            "active_attention_hooks": len(self.attention_hooks),
        }

    # ==================== OVERRIDES ====================

    def forward(
        self,
        src: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        time_features: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        return_teacher_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.student_features.clear()
        self.student_attentions.clear()

        teacher_output = None
        if self.distillation_mode != "none" and self.teacher_model is not None:
            self.teacher_features.clear()
            self.teacher_attentions.clear()
            with torch.no_grad():
                teacher_output = self.teacher_model(src, targets, time_features, epoch)

        student_output = super().forward(src, targets, time_features, epoch)
        if return_teacher_outputs and teacher_output is not None:
            return student_output, teacher_output
        return student_output

    def benchmark_inference(
        self, input_tensor: torch.Tensor, num_runs=100, warmup_runs=10
    ):
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        result.update(
            {
                "distillation_mode": self.distillation_mode,
                "has_teacher": self.teacher_model is not None,
            }
        )
        return result


# ==================== QUANTIZED FORECASTING MODEL ====================


class QuantizedForecastingModel(ForecastingModel):
    """
    Forecasting model with quantization support on top of distillation-enabled model.
    Supports PTQ, QAT, static and dynamic quantization.
    """

    VALID_QUANTIZATION_MODES = ["none", "ptq", "qat", "dynamic", "static"]

    def __init__(
        self,
        quantization_mode: str = "none",
        bit_width: int = 8,
        symmetric_quantization: bool = True,
        per_channel_quantization: bool = False,
        **kwargs,
    ):
        assert quantization_mode in self.VALID_QUANTIZATION_MODES, (
            f"Invalid quantization mode: {quantization_mode}"
        )

        super().__init__(**kwargs)

        # Core quantization config
        self.quantization_mode = quantization_mode
        self.bit_width = bit_width
        self.symmetric_quantization = symmetric_quantization
        self.per_channel_quantization = per_channel_quantization
        self.is_quantized = False

        self.quantization_config = QuantizationConfig(
            bit_width=bit_width,
            symmetric=symmetric_quantization,
            per_channel=per_channel_quantization,
        )

        # Quantization stubs (only needed for static/PTQ/QAT)
        if quantization_mode in {"ptq", "qat", "static"}:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        if quantization_mode != "none":
            self._setup_quantization()

    # ==================== SETUP ====================

    def _setup_quantization(self):
        """Setup quantization observers or fake quant depending on mode"""
        self._add_quantization_observers()

    def _add_quantization_observers(self):
        """Attach fake quant observers to Linear layers"""
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear) and self.quantization_mode == "qat":
                module.weight_fake_quant = FakeQuantize(self.quantization_config)
                if module.bias is not None:
                    module.bias_fake_quant = FakeQuantize(self.quantization_config)

    def set_quantization_mode(self, mode: str):
        """Switch quantization mode"""
        assert mode in self.VALID_QUANTIZATION_MODES, f"Invalid mode: {mode}"
        self.quantization_mode = mode

        if mode == "none":
            self.quant, self.dequant = nn.Identity(), nn.Identity()
        elif mode in {"ptq", "qat", "static"}:
            self.quant = ManualQuantStub(self.quantization_config)
            self.dequant = ManualDeQuantStub()
        else:  # dynamic
            self.quant, self.dequant = nn.Identity(), nn.Identity()

    # ==================== PREPARATION MODES ====================

    def prepare_for_quantization(self, calibration_data=None):
        if self.quantization_mode == "none":
            return self
        elif self.quantization_mode == "dynamic":
            return self._apply_dynamic_quantization()
        elif self.quantization_mode == "ptq":
            return self._apply_post_training_quantization(calibration_data)
        elif self.quantization_mode == "qat":
            return self._prepare_qat_training()
        elif self.quantization_mode == "static":
            return self._apply_static_quantization()

    def _apply_dynamic_quantization(self):
        """Dynamically quantize Linear layers"""
        return self._replace_linear_layers(DynamicQuantizedLinear)

    def _apply_static_quantization(self):
        """Apply static quantization after observer setup"""
        return self._replace_linear_layers(StaticQuantizedLinear)

    def _apply_post_training_quantization(self, calibration_data):
        """PTQ = Calibrate then convert"""
        self._add_quantization_observers()
        if calibration_data:
            self._calibrate_model(calibration_data)
        return self._convert_to_quantized_model()

    def _prepare_qat_training(self):
        """Attach fake quant modules and enable them for QAT"""
        self._add_quantization_observers()
        for m in self.modules():
            if hasattr(m, "weight_fake_quant"):
                m.weight_fake_quant.fake_quant_enabled = True
        return self

    def _calibrate_model(self, dataloader):
        """Collect statistics from calibration data"""
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                src, targets = (
                    batch[0],
                    batch[1] if isinstance(batch, (tuple, list)) else (batch, None),
                )
                _ = self.forward(src, targets)

    def finalize_quantization(self):
        """Convert QAT-trained model to fully quantized version"""
        if self.quantization_mode == "qat":
            for m in self.modules():
                if hasattr(m, "weight_fake_quant"):
                    m.weight_fake_quant.calculate_qparams()
                    m.weight_fake_quant.disable_observer()
                    m.weight_fake_quant.disable_fake_quant()
            return self._convert_to_quantized_model()
        return self

    def _convert_to_quantized_model(self):
        """Convert fake-quant-aware model to real quantized model"""
        return self._replace_linear_layers(StaticQuantizedLinear)

    # ==================== MODULE REPLACEMENT ====================

    def _replace_linear_layers(self, QuantLayerClass):
        """Replace all Linear layers with quantized equivalents"""
        model = copy.deepcopy(self)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quant_module = QuantLayerClass.from_float(
                    module, self.quantization_config
                )
                self._assign_module_by_name(model, name, quant_module)
        model.is_quantized = True
        return model

    def _assign_module_by_name(self, model, full_name, new_module):
        """Replace a submodule in the model hierarchy given its full dotted path"""
        path = full_name.split(".")
        parent = model
        for name in path[:-1]:
            parent = getattr(parent, name)
        setattr(parent, path[-1], new_module)

    # ==================== FORWARD / METRICS ====================

    def forward(
        self,
        src,
        targets=None,
        time_features=None,
        epoch=None,
        return_teacher_outputs=False,
    ):
        if self.quantization_mode in {"ptq", "qat", "static"}:
            src = self.quant(src)
        output = super().forward(
            src, targets, time_features, epoch, return_teacher_outputs
        )
        if self.quantization_mode in {"ptq", "qat", "static"}:
            if isinstance(output, tuple):
                output = (self.dequant(output[0]), output[1])
            else:
                output = self.dequant(output)
        return output

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Estimate model size in MB depending on quantization"""
        param_count = sum(p.numel() for p in self.parameters())
        buffer_count = sum(b.numel() for b in self.buffers())
        element_count = param_count + buffer_count
        element_size = 1 if self.is_quantized else 4
        size_mb = element_count * element_size / (1024**2)
        return {
            "parameters": param_count,
            "buffers": buffer_count,
            "total_elements": element_count,
            "size_mb": size_mb,
            "is_quantized": self.is_quantized,
        }

    def benchmark_inference(
        self, input_tensor: torch.Tensor, num_runs=100, warmup_runs=10
    ):
        result = super().benchmark_inference(input_tensor, num_runs, warmup_runs)
        result.update(
            {
                "quantization_mode": self.quantization_mode,
                "is_quantized": self.is_quantized,
            }
        )
        return result

    def get_quantization_info(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "quantization_enabled": self.quantization_mode != "none",
            "quantization_mode": self.quantization_mode,
            "bit_width": self.bit_width,
            "symmetric": self.symmetric_quantization,
            "per_channel": self.per_channel_quantization,
            "is_quantized": self.is_quantized,
            "num_quantizable_layers": sum(
                isinstance(m, nn.Linear) for m in self.modules()
            ),
        }
