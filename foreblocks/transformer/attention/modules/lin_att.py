import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.compute.triton_helpers import (
        HAS_TRITON as _HAS_TRITON,
    )
    from foreblocks.transformer.compute.triton_helpers import (
        triton_causal_linear_attn,
    )
except Exception:
    _HAS_TRITON = False
    triton_causal_linear_attn = None  # type: ignore[assignment]


class LinearAttention(nn.Module):
    """
    State-of-the-art linear attention using positive kernel approximation (ELU+1 feature map).
    O(L * d^2) complexity; drop-in for MultiAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = "standard",  # Ignored; always linear
        freq_modes: int = 16,  # Ignored
        cross_attention: bool = False,  # Ignored (separate Q/K/V projs anyway)
        feature_map: str = "elu",  # "elu" (default), "relu", or "rff" (random Fourier features)
        num_features: int | None = None,  # For "rff"; else uses d_head
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.feature_map = feature_map
        self.num_features = num_features or self.d_head
        self.cross_attention = cross_attention  # For future incremental tweaks
        self.scale = self.d_head**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if self.feature_map == "rff":
            # Random projection for unbiased softmax approx (Performer-style)
            self.omega = nn.Parameter(
                torch.randn(n_heads, self.d_head, self.num_features)
                * (1.0 / self.num_features**0.5),
                requires_grad=False,
            )

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map phi(x)."""
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "rff":
            # Simplified RFF: cos(proj) * exp(-||x||^2 / 2); project first
            proj = torch.einsum("b h l d, h d f -> b h l f", x, self.omega)
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            return torch.exp(-0.5 * norm_sq) * torch.cos(proj)
        else:
            raise ValueError(f"Unknown feature_map: {self.feature_map}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # Linear projections + reshape/transpose
        q = (
            self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lq Dh
        k = (
            self.k_proj(key).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lk Dh
        v = (
            self.v_proj(value).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lk Dh

        q = q * self.scale
        k = k * self.scale

        # Apply feature map
        q_prime = self._feature_map(q)  # B H Lq F (F= num_features or Dh)
        k_prime = self._feature_map(k)  # B H Lk F

        # Handle padding mask
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # B 1 Lk 1
            k_prime = k_prime.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # Compute linear attention (global sum or causal cumsum)
        if is_causal:
            # Causal: sequential scan or cumsum
            assert Lq == Lk, "Causal mode requires Lq == Lk"
            if (
                _HAS_TRITON
                and q_prime.is_cuda
                and not torch.is_grad_enabled()
                and not torch.jit.is_scripting()
            ):
                # Streaming Triton scan: O(B·H·T·(F+Dh)) memory, no huge intermediate
                out_heads = triton_causal_linear_attn(q_prime, k_prime, v, eps=1e-6)
            else:
                # PyTorch fallback — creates O(B·H·T·F·Dh) intermediate
                k_cum = torch.cumsum(k_prime, dim=2)  # B H L F
                kv_cum = torch.cumsum(
                    k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=2
                )  # B H L F Dh
                denom = torch.sum(q_prime * k_cum, dim=-1, keepdim=True)  # B H L 1
                numer = torch.matmul(q_prime, kv_cum)  # B H L Dh
                out_heads = numer / (denom + 1e-6)
        else:
            # Non-causal: fused einsum avoids O(B·H·L·F·Dh) intermediate tensor
            k_sum = k_prime.sum(dim=2)  # B H F
            kv_sum = torch.einsum("bhlf,bhld->bhfd", k_prime, v)  # B H F Dh
            denom = torch.matmul(q_prime, k_sum.unsqueeze(-1))  # B H L 1
            numer = torch.matmul(q_prime, kv_sum)  # B H L Dh
            out_heads = numer / (denom + 1e-6)

        # Reshape + project + dropout
        out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # B L D
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, None, None
