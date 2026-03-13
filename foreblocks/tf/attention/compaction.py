from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .paged import PagedKVCache


@dataclass
class AttentionMatchingConfig:
    keep_ratio: float = 0.25
    trigger_len: int = 512
    min_keep: int = 64
    query_budget: int = 64
    force_single_step: bool = False


class AttentionMatchingCompactor:
    """
    Experimental KV compactor inspired by attention matching.

    This first-pass implementation keeps a subset of KV slots, aggregates nearby
    values into those retained slots, and stores a learned log-bias (`beta`) per
    retained slot so decode can approximate the removed attention mass.
    """

    def __init__(self, config: AttentionMatchingConfig):
        self.config = config

    def should_compact(
        self,
        cache: PagedKVCache,
        batch_idx: int,
        t_new: int,
    ) -> bool:
        physical_len = cache.get_seq_len(batch_idx)
        if physical_len <= max(self.config.trigger_len, self.config.min_keep):
            return False
        if t_new <= 0:
            return False
        if self.config.force_single_step and t_new != 1:
            return False
        target_keep = self._target_keep(physical_len)
        return target_keep < physical_len

    def compact_batch(
        self,
        cache: PagedKVCache,
        q_bhtd: torch.Tensor,
        q_start_pos: torch.Tensor,
        kv_repeat: int,
        scale: float,
        t_new: int,
    ) -> None:
        if cache.use_latent_cache:
            raise RuntimeError(
                "Attention-matching compaction currently supports dense KV cache only."
            )
        if kv_repeat <= 0:
            raise ValueError(f"kv_repeat must be > 0, got {kv_repeat}")

        for b in range(cache.B):
            if not self.should_compact(cache, b, t_new=t_new):
                continue
            compacted = self._compact_sequence(
                cache=cache,
                batch_idx=b,
                q_bhtd=q_bhtd[b],
                q_start=int(q_start_pos[b].item()),
                kv_repeat=kv_repeat,
                scale=scale,
            )
            if compacted is None:
                continue
            k_comp, v_comp, pos_comp, beta_comp, logical_len = compacted
            cache.rewrite_seq_dense(
                b,
                k_comp,
                v_comp,
                pos_comp,
                beta_ht=beta_comp,
                logical_seq_len=logical_len,
            )

    def _target_keep(self, seq_len: int) -> int:
        keep = int(round(seq_len * float(self.config.keep_ratio)))
        keep = max(int(self.config.min_keep), keep)
        return min(seq_len, max(1, keep))

    def _compact_sequence(
        self,
        cache: PagedKVCache,
        batch_idx: int,
        q_bhtd: torch.Tensor,
        q_start: int,
        kv_repeat: int,
        scale: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        k_full, v_full = cache.gather_kv_for_seq(batch_idx)
        pos_full = cache.gather_positions_for_seq(batch_idx)
        seq_len = int(k_full.size(1))
        if seq_len == 0:
            return None

        target_keep = self._target_keep(seq_len)
        if target_keep >= seq_len:
            return None

        q_ref = self._select_query_budget(q_bhtd, self.config.query_budget)
        q_pos = q_start + torch.arange(q_ref.size(1), device=q_ref.device)

        kept_indices = self._select_kept_indices(
            q_ref=q_ref,
            q_pos=q_pos,
            k_full=k_full,
            pos_full=pos_full,
            kv_repeat=kv_repeat,
            scale=scale,
            target_keep=target_keep,
        )
        if kept_indices.numel() == 0:
            return None

        k_comp = k_full.index_select(1, kept_indices)
        v_comp, beta_comp = self._aggregate_values_and_bias(
            q_ref=q_ref,
            q_pos=q_pos,
            k_full=k_full,
            v_full=v_full,
            pos_full=pos_full,
            kept_indices=kept_indices,
            kv_repeat=kv_repeat,
            scale=scale,
        )
        pos_comp = pos_full.index_select(0, kept_indices)
        logical_len = cache.get_logical_seq_len(batch_idx)
        return k_comp, v_comp, pos_comp, beta_comp, logical_len

    def _select_query_budget(
        self,
        q_bhtd: torch.Tensor,
        query_budget: int,
    ) -> torch.Tensor:
        if q_bhtd.size(1) <= query_budget:
            return q_bhtd
        return q_bhtd[:, -query_budget:, :]

    def _select_kept_indices(
        self,
        q_ref: torch.Tensor,
        q_pos: torch.Tensor,
        k_full: torch.Tensor,
        pos_full: torch.Tensor,
        kv_repeat: int,
        scale: float,
        target_keep: int,
    ) -> torch.Tensor:
        repeated_k = k_full.repeat_interleave(kv_repeat, dim=0)
        scores = torch.matmul(q_ref, repeated_k.transpose(-2, -1)) * scale
        causal_mask = pos_full.view(1, 1, -1) > q_pos.view(1, -1, 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        key_mass = attn.sum(dim=(0, 1))
        target_keep = min(target_keep, key_mass.numel())
        topk = torch.topk(key_mass, k=target_keep, largest=True, sorted=False).indices
        return torch.sort(topk).values.to(dtype=torch.long)

    def _aggregate_values_and_bias(
        self,
        q_ref: torch.Tensor,
        q_pos: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        pos_full: torch.Tensor,
        kept_indices: torch.Tensor,
        kv_repeat: int,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_kv, _, _ = k_full.shape
        keep = int(kept_indices.numel())
        v_comp = v_full.new_zeros((h_kv, keep, v_full.size(-1)))
        beta_comp = v_full.new_zeros((h_kv, keep))

        for h in range(h_kv):
            q_group = q_ref[h * kv_repeat : (h + 1) * kv_repeat]
            if q_group.numel() == 0:
                q_group = q_ref[h : h + 1]
            scores = torch.matmul(q_group, k_full[h].transpose(-2, -1)) * scale
            causal_mask = pos_full.view(1, 1, -1) > q_pos.view(1, -1, 1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
            mass = attn.sum(dim=(0, 1)).clamp_min(1e-9)

            kept_keys = k_full[h].index_select(0, kept_indices)
            sim = torch.matmul(k_full[h], kept_keys.transpose(0, 1))
            nearest = sim.argmax(dim=-1)

            for j in range(keep):
                cluster = torch.nonzero(nearest == j, as_tuple=False).squeeze(-1)
                if cluster.numel() == 0:
                    cluster = kept_indices.new_tensor([int(kept_indices[j].item())])
                weights = mass.index_select(0, cluster)
                weights = weights / weights.sum().clamp_min(1e-9)
                v_comp[h, j] = (weights.unsqueeze(-1) * v_full[h].index_select(0, cluster)).sum(dim=0)

                sel_idx = int(kept_indices[j].item())
                sel_mass = mass[sel_idx]
                cluster_mass = mass.index_select(0, cluster).sum()
                beta_comp[h, j] = torch.log(cluster_mass / sel_mass.clamp_min(1e-9))

        return v_comp, beta_comp
