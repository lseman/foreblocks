"""Private parallel execution for HeadComposer.

This stateless mixin handles branch extraction, alignment, fusion, and stage
projection. Parameters and helper modules remain registered on HeadComposer
through its projection registry, preserving checkpoint paths and lazy creation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.heads.core.contracts import HeadOutput
from foreblocks.nn.heads.core.types import (
    ActiveHead,
    BaseRunState,
    HeadSpec,
    ParallelNoneState,
    RunStateList,
)


class _ParallelExecution:
    """Parallel-stage methods using the owning composer's state and helpers."""

    def _parallel_target_dim(self, x_ref: torch.Tensor) -> int:
        if self.parallel_project_dim is not None:
            return int(self.parallel_project_dim)
        return int(x_ref.size(-1))

    def _extract_parallel_tensor(self, spec: HeadSpec, out: Any) -> torch.Tensor:
        if isinstance(out, HeadOutput):
            if self.parallel_structured_outputs == "error" and (
                out.residual is not None or out.inverse_state is not None
            ):
                raise RuntimeError(
                    f"[{spec.name}] parallel stage received structured HeadOutput but "
                    "parallel_structured_outputs='error'"
                )
            if (
                self.parallel_structured_outputs == "main_add_second"
                and out.residual is not None
            ):
                return self._merge_parallel_main_and_second(
                    spec, out.value, out.residual
                )
            return out.value
        if isinstance(out, torch.Tensor):
            return out

        if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] parallel stage requires Tensor or 2-tuple with Tensor first element, "
                f"got {type(out).__name__}"
            )

        main = out[0]
        second = out[1]

        if self.parallel_structured_outputs == "error":
            raise RuntimeError(
                f"[{spec.name}] parallel stage received structured output but "
                "parallel_structured_outputs='error'. "
                "Set parallel_structured_outputs='main' or 'main_add_second'."
            )

        if self.parallel_structured_outputs == "main":
            return main

        if self.parallel_structured_outputs == "main_add_second":
            if not isinstance(second, torch.Tensor):
                return main
            return self._merge_parallel_main_and_second(spec, main, second)

        raise ValueError(
            f"Unknown parallel_structured_outputs mode: {self.parallel_structured_outputs}"
        )

    def _merge_parallel_main_and_second(
        self, spec: HeadSpec, main: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        extra = second
        if extra.size(0) != main.size(0):
            raise RuntimeError(
                f"[{spec.name}] parallel_structured_outputs='main_add_second' requires same batch size. "
                f"main={tuple(main.shape)} second={tuple(extra.shape)}"
            )
        if extra.size(1) != main.size(1):
            extra = self._align_time(extra, main.size(1))
        if extra.size(-1) != main.size(-1):
            proj = self._get_or_build_projection(
                category="stage_proj",
                name=f"parallel_structured_second__{spec.name}",
                in_dim=int(extra.size(-1)),
                out_dim=int(main.size(-1)),
                device=main.device,
                dtype=main.dtype,
                allow_identity=True,
            )
            extra = proj(extra)
        return main + extra

    def _align_parallel_branches_for_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x_ref: torch.Tensor,
    ) -> list[tuple[ActiveHead, torch.Tensor]]:
        if not branches:
            return []

        if self.parallel_align_mode == "strict":
            base = branches[0][1]
            for i, (_, b) in enumerate(branches[1:], start=1):
                if b.shape != base.shape:
                    raise RuntimeError(
                        f"parallel_combine='{self.parallel_combine}' requires equal branch shapes in "
                        "parallel_align_mode='strict'. "
                        f"branch0={tuple(base.shape)}, branch{i}={tuple(b.shape)}. "
                        "Set parallel_align_mode='project' to auto-align."
                    )
            return branches

        target_dim = self._parallel_target_dim(x_ref)
        target_T = int(x_ref.size(1))
        aligned: list[tuple[ActiveHead, torch.Tensor]] = []
        for idx, (head, b) in enumerate(branches):
            if b.size(0) != x_ref.size(0):
                raise RuntimeError(
                    f"[{head.name}] parallel branch batch mismatch: "
                    f"branch={tuple(b.shape)} vs input={tuple(x_ref.shape)}"
                )
            if b.size(1) != target_T:
                b = self._align_time(b, target_T)
            proj = self._get_or_build_projection(
                category="stage_proj",
                name=f"parallel_align__{head.name}__{idx}",
                in_dim=int(b.size(-1)),
                out_dim=target_dim,
                device=x_ref.device,
                dtype=x_ref.dtype,
                allow_identity=True,
            )
            b = proj(b)
            aligned.append((head, b))
        return aligned

    def _get_or_build_parallel_hypernet(
        self,
        in_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_hypernet",
            name=f"in{int(in_dim)}",
            in_dim=int(in_dim),
            out_dim=max(1, len(self.parallel_meta)),
            proj_type="hyper_mlp",
            hidden_dim=max(8, int(self.parallel_hyper_hidden_dim)),
            out_features=max(1, len(self.parallel_meta)),
            dropout=float(self.parallel_fusion_dropout),
            device=device,
            dtype=dtype,
        )

    def _get_or_build_parallel_gate_net(
        self,
        in_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_gate",
            name=f"in{int(in_dim)}",
            in_dim=int(in_dim),
            out_dim=1,
            proj_type="gate_mlp",
            hidden_dim=max(8, int(in_dim) // 2),
            out_features=1,
            device=device,
            dtype=dtype,
        )

    def _get_or_build_moe_gate_net(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_moe_gate",
            name=f"in{int(in_dim)}_out{int(out_dim)}",
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            proj_type="gate_mlp",
            hidden_dim=max(8, int(in_dim) // 2),
            out_features=int(out_dim),
            device=device,
            dtype=dtype,
        )

    def _resolve_parallel_attention_heads(self, embed_dim: int) -> int:
        if self.parallel_attention_heads > 0:
            if embed_dim % self.parallel_attention_heads != 0:
                raise ValueError(
                    f"parallel_attention_heads={self.parallel_attention_heads} does not divide "
                    f"embed_dim={embed_dim}"
                )
            return int(self.parallel_attention_heads)

        for h in (8, 4, 2, 1):
            if h <= embed_dim and embed_dim % h == 0:
                return h
        return 1

    def _get_or_build_parallel_attention(
        self,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.MultiheadAttention:
        num_heads = self._resolve_parallel_attention_heads(embed_dim)
        mod = self.projection_registry.get_or_create(
            category="parallel_attention",
            name=f"embed{int(embed_dim)}",
            in_dim=int(embed_dim),
            out_dim=int(embed_dim),
            proj_type="multihead_attention",
            num_heads=int(num_heads),
            dropout=float(self.parallel_fusion_dropout),
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        if not isinstance(mod, nn.MultiheadAttention):
            raise TypeError("ProjectionRegistry returned non-attention module")
        return mod

    def _forward_parallel_stage_impl(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, RunStateList]:
        branches, run_state = self._collect_parallel_branches(x)

        if not branches:
            return x, run_state

        out = self._combine_parallel_branches(branches, x)

        if self.parallel_project or self.parallel_project_dim is not None:
            target_dim = self._parallel_target_dim(x)
            proj = self._get_or_build_projection(
                category="stage_proj",
                name="parallel_out",
                in_dim=int(out.size(-1)),
                out_dim=target_dim,
                device=x.device,
                dtype=x.dtype,
                allow_identity=True,
            )
            out = proj(out)

        return out, run_state

    def _collect_parallel_branches(
        self, x: torch.Tensor
    ) -> tuple[list[tuple[ActiveHead, torch.Tensor]], RunStateList]:
        run_state: RunStateList = []
        branches: list[tuple[ActiveHead, torch.Tensor]] = []

        for active_head in self.parallel_active_heads:
            spec = active_head.spec
            if not active_head.enabled:
                run_state.append(
                    BaseRunState(
                        kind="disabled",
                        name=active_head.name,
                        stage="parallel",
                    )
                )
                continue

            out = spec.head(x)
            y = self._extract_parallel_tensor(spec, out)
            w_head, w_skip = self._alpha_weights_for_head(active_head)

            rec = ParallelNoneState(
                name=active_head.name,
                stage="parallel",
                base_dim=self._last_base_dim,
            )
            y, eff_head = self._mix_with_alpha(
                spec=spec,
                head_out=y,
                skip=x,
                w_head=w_head,
                w_skip=w_skip,
                hardened=active_head.hardened,
                device=x.device,
                dtype=x.dtype,
                context="parallel alpha mixing",
            )
            if active_head.fusion_scale_param is not None:
                y = y * torch.sigmoid(active_head.fusion_scale_param)
            rec.mix_w_head = None if eff_head is None else float(eff_head.item())

            branches.append((active_head, y))
            run_state.append(rec)

        return branches, run_state

    def _combine_parallel_branches(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        handlers = {
            "concat": self._combine_parallel_concat,
            "sum": self._combine_parallel_sum,
            "mean": self._combine_parallel_mean,
            "weighted_sum": self._combine_parallel_weighted_sum,
            "hypernetwork_mix": self._combine_parallel_hypernetwork_mix,
            "attention_fusion": self._combine_parallel_attention_fusion,
            "gated_fusion": self._combine_parallel_gated_fusion,
            "lora_mix": self._combine_parallel_lora_mix,
            "moe_routing": self._combine_parallel_moe_routing,
            "multi_scale_attn": self._combine_parallel_multi_scale_attn,
        }
        try:
            handler = handlers[self.parallel_combine]
        except KeyError as exc:
            raise ValueError(
                f"Unknown parallel_combine: {self.parallel_combine}"
            ) from exc
        return handler(branches, x)

    @staticmethod
    def _combine_parallel_concat(
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        del x
        return torch.cat([branch for _, branch in branches], dim=-1)

    def _combine_parallel_sum(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        return torch.stack([branch for _, branch in aligned], dim=0).sum(dim=0)

    def _combine_parallel_mean(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        branch_tensors = [branch for _, branch in aligned]
        return torch.stack(branch_tensors, dim=0).mean(dim=0)

    def _combine_parallel_weighted_sum(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        if any(head.mix_logit is None for head, _ in aligned):
            raise RuntimeError("Missing parallel mix logits for weighted_sum.")

        logits = torch.stack(
            [head.mix_logit[0] for head, _ in aligned],
            dim=0,
        ).to(device=x.device)
        weights = F.softmax(logits.float(), dim=0).to(dtype=aligned[0][1].dtype)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[i]
        return out

    def _combine_parallel_hypernetwork_mix(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        branch_stack = torch.stack([branch for _, branch in aligned], dim=0)
        context = branch_stack.mean(dim=0).mean(dim=1)
        hyper = self._get_or_build_parallel_hypernet(
            in_dim=int(context.size(-1)),
            device=x.device,
            dtype=x.dtype,
        )
        logits_all = hyper(context)
        active_idx = torch.tensor(
            [head.index for head, _ in aligned],
            device=x.device,
            dtype=torch.long,
        )
        logits = logits_all.index_select(1, active_idx)
        weights = F.softmax(logits.float(), dim=1).to(dtype=aligned[0][1].dtype)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[:, i].view(-1, 1, 1)
        return out

    def _combine_parallel_attention(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        batch_size, time_steps, feature_dim = aligned[0][1].shape
        num_branches = len(aligned)
        attn = self._get_or_build_parallel_attention(
            embed_dim=feature_dim,
            device=x.device,
            dtype=x.dtype,
        )
        stacked = torch.stack([branch for _, branch in aligned], dim=2)
        tokens = stacked.reshape(batch_size * time_steps, num_branches, feature_dim)
        attn_out, _ = attn(tokens, tokens, tokens, need_weights=False)
        return attn_out.mean(dim=1).reshape(batch_size, time_steps, feature_dim)

    def _combine_parallel_attention_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._combine_parallel_attention(branches, x)

    def _combine_parallel_multi_scale_attn(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._combine_parallel_attention(branches, x)

    def _combine_parallel_lora_mix(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        out = torch.zeros_like(aligned[0][1])
        for head, branch in aligned:
            rank = head.spec.lora_rank or max(1, int(branch.size(-1) // 4))
            adapter = self._get_or_build_projection(
                category="lora_adapter",
                name=f"lora_mix_{head.name}",
                in_dim=int(branch.size(-1)),
                out_dim=int(branch.size(-1)),
                device=branch.device,
                dtype=branch.dtype,
                proj_type="lora_adapter",
                rank=rank,
            )
            out = out + adapter(branch)
        return out

    def _combine_parallel_moe_routing(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        context = x.mean(dim=1)
        moe_net = self._get_or_build_moe_gate_net(
            in_dim=int(context.size(-1)),
            out_dim=len(aligned),
            device=x.device,
            dtype=x.dtype,
        )
        logits = moe_net(context)
        weights = F.softmax(logits / self.moe_temperature, dim=-1)
        topk = min(max(1, min(head.spec.moe_k for head, _ in aligned)), len(aligned))
        _, topk_idx = torch.topk(weights, k=topk, dim=-1)
        mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self._EPS)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[:, i].view(-1, 1, 1).to(dtype=branch.dtype)
        return out

    def _combine_parallel_gated_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        feature_dim = aligned[0][1].size(-1)
        gate_net = self._get_or_build_parallel_gate_net(
            in_dim=int(feature_dim),
            device=x.device,
            dtype=x.dtype,
        )
        gate_scores = torch.stack(
            [torch.sigmoid(gate_net(branch.mean(dim=1))) for _, branch in aligned],
            dim=1,
        )
        gate_norm = gate_scores / gate_scores.sum(dim=1, keepdim=True).clamp_min(
            self._EPS
        )
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * gate_norm[:, i, :].unsqueeze(1).to(dtype=branch.dtype)
        return out
