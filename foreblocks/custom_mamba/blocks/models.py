from __future__ import annotations

import torch
import torch.nn as nn

from .feedforward import FeedForward
from .hybrid import HybridMamba2Block
from .ssm import HybridMambaBlock


class TinyHybridMambaLM(nn.Module):
    """Small language model built from stacked ``HybridMambaBlock`` layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        tie_embeddings: bool = True,
        use_pre_norm: bool = True,
        mlp_every_n: int = 0,
        ffn_expansion: float = 8 / 3,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            HybridMambaBlock(
                d_model=d_model,
                d_inner=2 * d_model,
                d_state=d_state,
                d_conv=d_conv,
                dt_rank=dt_rank,
                use_cuda_scan=True,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(n_layers)
        ])
        self._has_ffn: list[bool] = [
            mlp_every_n > 0 and (i + 1) % mlp_every_n == 0 for i in range(n_layers)
        ]
        self.ffns = nn.ModuleList(
            FeedForward(d_model, expansion=ffn_expansion, dropout=ffn_dropout)
            if has
            else nn.Identity()
            for has in self._has_ffn
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk, ffn, has_ffn in zip(self.blocks, self.ffns, self._has_ffn):
            x = x + blk(x)
            if has_ffn:
                x = x + ffn(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def make_states(self, batch: int, device=None, dtype=None) -> list[dict]:
        """Return per-block recurrent states for token-by-token inference."""
        return [
            blk.make_state(batch, device=device, dtype=dtype) for blk in self.blocks
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation using recurrent step mode."""
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        states = self.make_states(B, device=device, dtype=dtype)

        def _step(tok: torch.Tensor) -> torch.Tensor:
            x = self.embed(tok)
            for i, (blk, ffn, has_ffn) in enumerate(
                zip(self.blocks, self.ffns, self._has_ffn)
            ):
                x = x + blk.step(x, states[i])
                if has_ffn:
                    x = x + ffn(x)
            return self.final_norm(x)

        for t in range(input_ids.shape[1]):
            last = _step(input_ids[:, t])

        generated: list[torch.Tensor] = []
        logits = self.lm_head(last)
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tok = logits.argmax(-1)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(
                        scaled < topk_vals[:, -1:], float("-inf")
                    )
                next_tok = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(
                    1
                )
            generated.append(next_tok)
            logits = self.lm_head(_step(next_tok))

        if not generated:
            return input_ids
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)


class TinyHybridMamba2LM(nn.Module):
    """Small LM alternating pure SSM and SSM + attention layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        n_kv_heads: int | None = None,
        window_size: int = 128,
        attn_every_n: int = 2,
        tie_embeddings: bool = True,
        use_gated_delta: bool = False,
        use_cuda_scan: bool = True,
        use_pre_norm: bool = True,
        rope_base: int = 10_000,
        max_seq_len: int = 8192,
        n_sink_tokens: int = 0,
        mlp_every_n: int = 0,
        ffn_expansion: float = 8 / 3,
        ffn_dropout: float = 0.0,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attn_logit_softcap: float | None = None,
        layer_scale_init: float | None = None,
    ):
        super().__init__()
        if attn_every_n < 1:
            raise ValueError("attn_every_n must be >= 1")

        self.embed = nn.Embedding(vocab_size, d_model)
        blocks = []
        for idx in range(n_layers):
            if idx % attn_every_n == 0:
                block = HybridMamba2Block(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    num_heads=num_heads,
                    n_kv_heads=n_kv_heads,
                    window_size=window_size,
                    use_gated_delta=use_gated_delta,
                    use_cuda_scan=use_cuda_scan,
                    rope_base=rope_base,
                    max_seq_len=max_seq_len,
                    n_sink_tokens=n_sink_tokens,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                    attn_logit_softcap=attn_logit_softcap,
                    layer_scale_init=layer_scale_init,
                )
            else:
                block = HybridMambaBlock(
                    d_model=d_model,
                    d_inner=2 * d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                    use_cuda_scan=use_cuda_scan,
                    use_pre_norm=use_pre_norm,
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self._has_ffn: list[bool] = [
            mlp_every_n > 0 and (i + 1) % mlp_every_n == 0 for i in range(n_layers)
        ]
        self.ffns = nn.ModuleList(
            FeedForward(d_model, expansion=ffn_expansion, dropout=ffn_dropout)
            if has
            else nn.Identity()
            for has in self._has_ffn
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk, ffn, has_ffn in zip(self.blocks, self.ffns, self._has_ffn):
            x = x + blk(x)
            if has_ffn:
                x = x + ffn(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def make_states(self, batch: int, device=None, dtype=None) -> list[dict]:
        """Return per-block recurrent states for token-by-token inference."""
        return [
            blk.make_state(batch, device=device, dtype=dtype) for blk in self.blocks
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation using recurrent step mode."""
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        states = self.make_states(B, device=device, dtype=dtype)

        def _step(tok: torch.Tensor) -> torch.Tensor:
            x = self.embed(tok)
            for i, (blk, ffn, has_ffn) in enumerate(
                zip(self.blocks, self.ffns, self._has_ffn)
            ):
                x = x + blk.step(x, states[i])
                if has_ffn:
                    x = x + ffn(x)
            return self.final_norm(x)

        for t in range(input_ids.shape[1]):
            last = _step(input_ids[:, t])

        generated: list[torch.Tensor] = []
        logits = self.lm_head(last)
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tok = logits.argmax(-1)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(
                        scaled < topk_vals[:, -1:], float("-inf")
                    )
                next_tok = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(
                    1
                )
            generated.append(next_tok)
            logits = self.lm_head(_step(next_tok))

        if not generated:
            return input_ids
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)
