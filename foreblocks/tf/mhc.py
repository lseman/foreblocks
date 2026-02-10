import torch
import torch.nn as nn
# ──────────────────────────────────────────────────────────────────────────────
# mHC utilities
# ──────────────────────────────────────────────────────────────────────────────
def sinkhorn_doubly_stochastic(
    logits: torch.Tensor,
    iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable Sinkhorn-Knopp projection to (approx) doubly-stochastic matrices.

    logits: [N, N] unconstrained
    returns: [N, N] approx doubly stochastic (rows/cols sum to 1, nonnegative)
    """
    if logits.dim() != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Expected square [N,N] logits, got {logits.shape}")

    # Stabilize exponentiation
    x = logits - logits.max(dim=-1, keepdim=True).values
    p = torch.exp(x) + eps

    for _ in range(iters):
        p = p / (p.sum(dim=-1, keepdim=True) + eps)  # row normalize
        p = p / (p.sum(dim=-2, keepdim=True) + eps)  # col normalize

    return p


class MHCResidualMixer(nn.Module):
    """
    mHC residual stream mixer.
      streams: [B, N, T, D]
      streams <- H @ streams  (mix across N)
    where H is constrained to the Birkhoff polytope via Sinkhorn.
    """

    def __init__(
        self,
        n_streams: int,
        sinkhorn_iters: int = 20,
        init: str = "identity",  # "identity" or "uniform"
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n = int(n_streams)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.logits = nn.Parameter(torch.zeros(self.n, self.n))

        if init == "identity":
            with torch.no_grad():
                self.logits.fill_(-4.0)
                self.logits.diagonal().fill_(4.0)
        elif init == "uniform":
            with torch.no_grad():
                self.logits.zero_()
        else:
            raise ValueError(f"Unknown init: {init}")

    def get_H(self) -> torch.Tensor:
        temp = max(self.temperature, 1e-6)
        H = sinkhorn_doubly_stochastic(
            self.logits / temp,
            iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        return H  # [N, N]

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """
        streams: [B, N, T, D]
        """
        if streams.dim() != 4:
            raise ValueError(f"Expected [B,N,T,D], got {streams.shape}")
        B, N, T, D = streams.shape
        if N != self.n:
            raise ValueError(f"streams N={N} != mixer n={self.n}")
        H = self.get_H().to(dtype=streams.dtype, device=streams.device)
        # out[b,i,t,d] = sum_j H[i,j] * streams[b,j,t,d]
        return torch.einsum("ij,bjtd->bitd", H, streams)


def mhc_init_streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """
    x: [B,T,D] -> streams: [B,N,T,D] (replicate)
    """
    if x.dim() != 3:
        raise ValueError(f"Expected [B,T,D], got {x.shape}")
    B, T, D = x.shape
    return x.unsqueeze(1).expand(B, int(n_streams), T, D).contiguous()


def mhc_collapse_streams(streams: torch.Tensor, mode: str = "first") -> torch.Tensor:
    """
    streams: [B,N,T,D] -> [B,T,D]
    """
    if streams.dim() != 4:
        raise ValueError(f"Expected [B,N,T,D], got {streams.shape}")
    if mode == "first":
        return streams[:, 0]
    if mode == "mean":
        return streams.mean(dim=1)
    raise ValueError(f"Unknown collapse mode: {mode}")


def mhc_apply_norm_streamwise(norm: nn.Module, streams: torch.Tensor) -> torch.Tensor:
    """
    Apply a norm layer (expecting [B,T,D]) to each stream.
    streams: [B,N,T,D] -> [B,N,T,D]
    """
    B, N, T, D = streams.shape
    flat = streams.reshape(B * N, T, D)
    out = norm(flat)
    return out.reshape(B, N, T, D)


def mhc_run_sublayer_streamwise(
    fn_flat: callable,
    streams: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a function that maps [B*T,D] shaped as [B',T,D] (flattened streams)
    across each stream by reshaping.

    fn_flat: takes [B*N, T, D] -> [B*N, T, D]
    streams: [B,N,T,D] -> [B,N,T,D]
    """
    B, N, T, D = streams.shape
    flat = streams.reshape(B * N, T, D)
    out = fn_flat(flat)
    return out.reshape(B, N, T, D)

def mhc_repeat_kpm(kpm: torch.Tensor | None, n_streams: int) -> torch.Tensor | None:
    """
    key_padding_mask: [B, T] -> [B*n_streams, T]
    """
    if kpm is None:
        return None
    if kpm.dim() != 2:
        raise ValueError(f"Expected key_padding_mask [B,T], got {tuple(kpm.shape)}")
    return kpm.repeat_interleave(n_streams, dim=0)

def mhc_repeat_attn_mask(attn_mask: torch.Tensor | None, n_streams: int) -> torch.Tensor | None:
    """
    If attn_mask is batched, repeat batch dim. If it's [T,T], keep it.
    Supports common shapes:
      - [T, T] (shared)  -> unchanged
      - [B, T, T]        -> [B*n, T, T]
      - [B, 1, T, T]     -> [B*n, 1, T, T]
    """
    if attn_mask is None:
        return None
    if attn_mask.dim() == 2:
        return attn_mask
    if attn_mask.dim() == 3:
        return attn_mask.repeat_interleave(n_streams, dim=0)
    if attn_mask.dim() == 4:
        return attn_mask.repeat_interleave(n_streams, dim=0)
    raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")
