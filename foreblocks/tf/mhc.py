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

    logits: [..., N, N] unconstrained
    returns: [..., N, N] approx doubly stochastic (rows/cols sum to 1, nonnegative)
    """
    if logits.dim() < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError(f"Expected [...,N,N] logits, got {tuple(logits.shape)}")

    x = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.exp(x) + eps

    for _ in range(iters):
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        p = p / (p.sum(dim=-2, keepdim=True) + eps)

    return p


def _rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms


class MHCHyperConnection(nn.Module):
    """
    Token-wise manifold-constrained Hyper-Connection.

    streams: [B, N, T, D]
    H_pre:   [B, T, N]
    H_post:  [B, T, N]
    H_res:   [B, T, N, N]
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n = int(n_streams)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)

        flat_dim = self.n * self.d_model
        self.pre_proj = nn.Linear(flat_dim, self.n, bias=False)
        self.post_proj = nn.Linear(flat_dim, self.n, bias=False)
        self.res_proj = nn.Linear(flat_dim, self.n * self.n, bias=False)

        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.pre_bias = nn.Parameter(self._init_pre_bias())
        self.post_bias = nn.Parameter(self._init_post_bias())
        self.res_bias = nn.Parameter(self._init_res_bias())

        self._reset_parameters()

    def _init_pre_bias(self) -> torch.Tensor:
        # Select the first stream at initialization.
        bias = torch.full((self.n,), -6.0)
        bias[0] = 6.0
        return bias

    def _init_post_bias(self) -> torch.Tensor:
        # Write the block update primarily into the first stream.
        bias = torch.full((self.n,), -6.0)
        bias[0] = 0.0
        return bias

    def _init_res_bias(self) -> torch.Tensor:
        bias = torch.full((self.n, self.n), -6.0)
        bias.diagonal().fill_(6.0)
        return bias

    def _reset_parameters(self) -> None:
        for proj in (self.pre_proj, self.post_proj, self.res_proj):
            nn.init.zeros_(proj.weight)

    def _flatten_streams(self, streams: torch.Tensor) -> torch.Tensor:
        if streams.dim() != 4:
            raise ValueError(f"Expected [B,N,T,D], got {tuple(streams.shape)}")
        if streams.size(1) != self.n or streams.size(-1) != self.d_model:
            raise ValueError(
                f"streams shape {tuple(streams.shape)} incompatible with "
                f"(n_streams={self.n}, d_model={self.d_model})"
            )
        x = streams.permute(0, 2, 1, 3).contiguous()  # [B,T,N,D]
        return x.view(x.size(0), x.size(1), self.n * self.d_model)

    def compute_maps(self, streams: torch.Tensor) -> dict:
        flat = self._flatten_streams(streams)
        flat_norm = _rms_norm_last_dim(flat, eps=self.eps)

        pre_logits = self.alpha_pre * self.pre_proj(flat_norm) + self.pre_bias
        post_logits = self.alpha_post * self.post_proj(flat_norm) + self.post_bias
        res_logits = self.alpha_res * self.res_proj(flat_norm)
        res_logits = res_logits.view(flat.size(0), flat.size(1), self.n, self.n)
        res_logits = res_logits + self.res_bias

        h_pre = torch.sigmoid(pre_logits)
        h_post = 2.0 * torch.sigmoid(post_logits)
        h_res = sinkhorn_doubly_stochastic(
            res_logits,
            iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        return {"pre": h_pre, "post": h_post, "res": h_res}

    def pre_aggregate(
        self,
        streams: torch.Tensor,
        maps: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        maps = self.compute_maps(streams) if maps is None else maps
        streams_btnd = streams.permute(0, 2, 1, 3).contiguous()
        x_in = torch.einsum("btn,btnd->btd", maps["pre"], streams_btnd)
        return x_in, maps

    def combine(
        self,
        streams: torch.Tensor,
        update: torch.Tensor,
        maps: dict | None = None,
    ) -> torch.Tensor:
        maps = self.compute_maps(streams) if maps is None else maps
        streams_btnd = streams.permute(0, 2, 1, 3).contiguous()
        mixed = torch.einsum("btij,btjd->btid", maps["res"], streams_btnd)
        written = maps["post"].unsqueeze(-1) * update.unsqueeze(2)
        out = mixed + written
        return out.permute(0, 2, 1, 3).contiguous()


def mhc_init_streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """
    x: [B,T,D] -> streams: [B,N,T,D]
    Paper-style initialization places the input in the first stream and zeros the rest.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected [B,T,D], got {tuple(x.shape)}")
    B, T, D = x.shape
    streams = x.new_zeros(B, int(n_streams), T, D)
    streams[:, 0] = x
    return streams


def mhc_collapse_streams(streams: torch.Tensor, mode: str = "first") -> torch.Tensor:
    """
    streams: [B,N,T,D] -> [B,T,D]
    """
    if streams.dim() != 4:
        raise ValueError(f"Expected [B,N,T,D], got {tuple(streams.shape)}")
    if mode == "first":
        return streams[:, 0]
    if mode == "mean":
        return streams.mean(dim=1)
    raise ValueError(f"Unknown collapse mode: {mode}")


def mhc_apply_norm_streamwise(norm: nn.Module, streams: torch.Tensor) -> torch.Tensor:
    """
    Apply a norm layer (expecting [B,T,D]) to each stream.
    """
    B, N, T, D = streams.shape
    flat = streams.reshape(B * N, T, D)
    out = norm(flat)
    return out.reshape(B, N, T, D)
