from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

@node(
    type_id="chronos2_embed_head",
    name="Chronos2EmbedHead",
    category="Feature",
    outputs=["chronos2_embeddings"],
    color="bg-gradient-to-r from-sky-500 to-emerald-500",
)
class Chronos2EmbedHead(BaseHead):
    """
    Extract Chronos-2 embeddings and return a 3D sequence for downstream encoders.

    IMPORTANT (optimizer safety):
      This head lazily creates a projector (D -> F) on first forward when needed.
      If you build the optimizer BEFORE the first forward, the projector params
      will NOT be in the optimizer. Fix by calling:

          head.warmup(sample_x)

      once before creating the optimizer (or run one model forward pass).

    Forward:
      x: [B,T,F] -> returns [B,T,F_out] or [B,T+1,F] depending on attach.
    """

    def __init__(
        self,
        pipeline,
        channel: int = 0,
        reduction: str = "mean",
        hook_layer: str = "encoder",
        pred_len: int = 1,
        attach: str = "feature",  # 'feature' | 'time_token' | 'replace'
        proj_to_input_dim: bool = True,
        offload_cpu: bool = True,
        make_dates=None,
    ):
        module = _Chronos2EmbedderModule(
            pipeline=pipeline,
            channel=channel,
            reduction=reduction,
            hook_layer=hook_layer,
            pred_len=pred_len,
            attach=attach,
            proj_to_input_dim=proj_to_input_dim,
            offload_cpu=offload_cpu,
            make_dates=make_dates,
        )
        super().__init__(module=module, name="chronos2_embed")

    @torch.no_grad()
    def warmup(self, x: torch.Tensor) -> None:
        """Run once to instantiate any lazy projector parameters (call before optimizer)."""
        _ = self.module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class _Chronos2EmbedderModule(nn.Module):
    def __init__(
        self,
        pipeline,
        channel: int,
        reduction: str,
        hook_layer: str,
        pred_len: int,
        attach: str,
        proj_to_input_dim: bool,
        offload_cpu: bool,
        make_dates,
    ):
        super().__init__()
        if reduction not in {"mean", "last"}:
            raise ValueError("reduction must be 'mean' or 'last'")
        if hook_layer not in {"encoder", "input_patch"}:
            raise ValueError("hook_layer must be 'encoder' or 'input_patch'")
        if attach not in {"feature", "time_token", "replace"}:
            raise ValueError("attach must be 'feature', 'time_token', or 'replace'")

        self.pipeline = pipeline
        self.channel = int(channel)
        self.reduction = reduction
        self.hook_layer = hook_layer
        self.pred_len = max(1, int(pred_len))
        self.attach = attach
        self.proj_to_input_dim = bool(proj_to_input_dim)
        self.offload_cpu = bool(offload_cpu)
        self.make_dates = make_dates

        # Resolve model + hook target
        self._model = getattr(pipeline, "model", None) or getattr(pipeline, "inner_model", None)
        if self._model is None:
            raise RuntimeError("Chronos2EmbedHead: could not locate Chronos2Model.")
        if self.hook_layer == "encoder":
            if not hasattr(self._model, "encoder") or not hasattr(self._model.encoder, "final_layer_norm"):
                raise RuntimeError("Chronos2EmbedHead: encoder.final_layer_norm not found.")
            self._hook_module = self._model.encoder.final_layer_norm
        else:
            if not hasattr(self._model, "input_patch_embedding"):
                raise RuntimeError("Chronos2EmbedHead: input_patch_embedding not found.")
            self._hook_module = self._model.input_patch_embedding

        # Lazy init for projector (Chronos D -> input F)
        self._proj: Optional[nn.Linear] = None  # created on first forward once we know D and F

    @torch.no_grad()
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np
        import pandas as pd

        B, T, F = x.shape

        # Build minimal DF
        if self.make_dates is None:
            def _mk_dates(n):  # daily
                return pd.date_range("2000-01-01", periods=n, freq="D")
            make_dates = _mk_dates
        else:
            make_dates = self.make_dates

        x_cpu = x.detach().to("cpu")
        vals = x_cpu[:, :, self.channel].numpy().astype(np.float32)  # [B,T]

        ids, stamps, tgts = [], [], []
        for b in range(B):
            ids.extend([f"series_{b}"] * T)
            stamps.extend(make_dates(T))
            tgts.extend(vals[b].tolist())

        context_df = (
            pd.DataFrame({"id": ids, "timestamp": stamps, "target": tgts})
            .sort_values(["id", "timestamp"])
        )

        collected: List[torch.Tensor] = []

        def _hook(_, __, out):
            collected.append(out.detach().to("cpu") if self.offload_cpu else out.detach())

        h = self._hook_module.register_forward_hook(_hook)
        try:
            _ = self.pipeline.predict_df(
                context_df,
                future_df=None,
                prediction_length=self.pred_len,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        finally:
            h.remove()

        if not collected:
            raise RuntimeError("Chronos2EmbedHead: no activations captured.")

        enc = torch.cat(collected, dim=0)  # typically [B, P, D]
        if enc.size(0) != B:
            enc = enc[:B]

        if enc.dim() == 2:
            # If hook returns [B, D], treat as already pooled
            emb = enc
        else:
            # pool across patches/positions
            emb = enc.mean(dim=1) if self.reduction == "mean" else enc[:, -1]  # [B, D]

        return emb  # cpu or gpu depending on offload

    def _ensure_proj(self, D: int, Fout: int, device: torch.device, dtype: torch.dtype) -> nn.Linear:
        if (self._proj is None) or (self._proj.in_features != D) or (self._proj.out_features != Fout):
            self._proj = nn.Linear(D, Fout, bias=True).to(device=device, dtype=dtype)
        else:
            self._proj.to(device=device, dtype=dtype)
        return self._proj

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F] -> returns sequence with embeddings integrated:
          - attach='feature'    -> [B,T,F+D]
          - attach='time_token' -> [B,T+1,F]  (D→F projected if proj_to_input_dim)
          - attach='replace'    -> [B,T,D] (or [B,T,F] if projected)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B,T,F], got {tuple(x.shape)}")
        B, T, Fin = x.shape

        emb = self._get_embeddings(x)  # [B, D]
        D = emb.size(-1)
        device, dtype = x.device, x.dtype
        emb = emb.to(device=device, dtype=dtype)

        proj_emb = None
        if (self.attach in {"time_token", "replace"}) and self.proj_to_input_dim:
            proj = self._ensure_proj(D, Fin, device=device, dtype=dtype)
            proj_emb = proj(emb)  # [B, Fin]

        if self.attach == "feature":
            rep = emb.unsqueeze(1).repeat(1, T, 1)  # [B,T,D]
            return torch.cat([x, rep], dim=-1)      # [B,T,Fin+D]

        if self.attach == "time_token":
            if self.proj_to_input_dim:
                tok = proj_emb.unsqueeze(1)        # [B,1,Fin]
            else:
                if D == Fin:
                    tok = emb.unsqueeze(1)
                elif D > Fin:
                    tok = emb[:, :Fin].unsqueeze(1)
                else:
                    pad = torch.zeros(B, Fin - D, device=device, dtype=dtype)
                    tok = torch.cat([emb, pad], dim=-1).unsqueeze(1)
            return torch.cat([x, tok], dim=1)       # [B,T+1,Fin]

        # attach == "replace"
        base = proj_emb if (self.proj_to_input_dim and proj_emb is not None) else emb
        return base.unsqueeze(1).repeat(1, T, 1)    # [B,T,Fin or D]


# ──────────────────────────────────────────────────────────────────────────────
# DropoutTS (training-time temporal dropout / span masking)
# ──────────────────────────────────────────────────────────────────────────────
