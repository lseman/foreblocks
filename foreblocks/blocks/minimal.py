import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_batch_adj(A, B):
    # A: [N,N] or [B,N,N] or None -> return [B,N,N] or None
    if A is None:
        return None
    if A.dim() == 2:
        N = A.size(0)
        return A.unsqueeze(0).expand(B, N, N).contiguous()
    return A


class GraphNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B,N,F]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * xhat + self.bias


class PreNorm(nn.Module):
    def __init__(self, dim, fn, use_graphnorm=True):
        super().__init__()
        self.norm = GraphNorm(dim) if use_graphnorm else nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, A=None):
        return self.fn(self.norm(x), A)


class ChebConv(nn.Module):
    def __init__(
        self, in_dim, out_dim, K=3, bias=True, dynamic_if_none=True, topk: int = 0
    ):
        super().__init__()
        self.K = K
        self.dynamic_if_none = dynamic_if_none
        self.topk = topk
        self.lin = nn.Linear(in_dim * K, out_dim, bias=bias)

    @staticmethod
    def _cosine_adj(x: torch.Tensor, topk: int = 0) -> torch.Tensor:
        # x:[B,N,F] -> A:[B,N,N] (cosine >=0)
        x = F.normalize(x, p=2, dim=-1)
        A = torch.einsum("bnf,bmf->bnm", x, x).clamp_min(0)
        if topk and topk < A.size(-1):
            # keep topk per node (including self if present)
            vals, idx = torch.topk(A, k=topk, dim=-1)
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            A = A.masked_fill(~mask, 0.0)
        return A

    @staticmethod
    def _norm_laplacian(A: torch.Tensor) -> torch.Tensor:
        # A:[B,N,N] -> normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, N, N)
        deg = A.sum(-1) + 1e-6
        Dm12 = deg.pow(-0.5).unsqueeze(-1)
        An = Dm12 * A * Dm12.transpose(1, 2)
        return I - An

    def forward(self, x, A):
        # x:[B,N,F], A:[B,N,N] or None
        B, N, F = x.shape
        if A is None:
            if not self.dynamic_if_none:
                raise ValueError(
                    "ChebConv received adj=None; pass an adjacency or set dynamic_if_none=True."
                )
            A = self._cosine_adj(x, topk=self.topk)

        L = self._norm_laplacian(A)
        I = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, N, N)

        # Chebyshev on (I - 2L) (affine scaling of spectrum) — simple stable variant
        G = I - 2 * L

        T0 = x
        T = [T0]
        if self.K > 1:
            T1 = torch.bmm(G, T0)
            T.append(T1)
        for _ in range(2, self.K):
            T_next = 2 * torch.bmm(G, T[-1]) - T[-2]
            T.append(T_next)

        h = torch.cat(T, dim=-1)  # [B,N,F*K]
        y = self.lin(h)
        return y


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, A):
        B, N, F = x.shape
        A = _ensure_batch_adj(A, B)
        deg = A.sum(-1, keepdim=True).clamp_min(1)
        neigh = torch.bmm(A, x) / deg  # mean agg
        y = torch.cat([x, neigh], dim=-1)
        return F.relu(self.lin(y))


class GATv2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_dim))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, A):
        B, N, F = x.shape
        A = _ensure_batch_adj(A, B)
        h = self.W(x)  # [B,N,H]
        # compute attention logits e_ij = a^T [h_i || h_j]
        hi = h.unsqueeze(2).expand(B, N, N, h.size(-1))
        hj = h.unsqueeze(1).expand(B, N, N, h.size(-1))
        e = torch.einsum("d,bind->bin", self.a, torch.cat([hi, hj], dim=-1))
        e = self.leaky(e)
        e = e.masked_fill(A <= 0, float("-inf"))
        alpha = torch.softmax(e, dim=-1)  # [B,N,N]
        y = torch.bmm(alpha, h)
        return y


class DiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(in_dim * (K + 1), out_dim)

    def forward(self, x, A):
        B, N, F = x.shape
        A = _ensure_batch_adj(A, B)
        deg = A.sum(-1, keepdim=True).clamp_min(1)
        P = A / deg  # row-stochastic
        xs = [x]
        Pk = P
        cur = x
        for _ in range(self.K):
            cur = torch.bmm(Pk, cur)
            xs.append(cur)
            Pk = torch.bmm(Pk, P)  # next power
        y = self.lin(torch.cat(xs, dim=-1))
        return y


class APPNP(nn.Module):
    def __init__(self, in_dim, out_dim, K=10, alpha=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.K = K
        self.alpha = alpha

    def forward(self, x, A):
        B, N, F = x.shape
        A = _ensure_batch_adj(A, B)
        deg = A.sum(-1) + 1e-6
        Dm12 = deg.pow(-0.5).unsqueeze(-1)
        An = Dm12 * A * Dm12.transpose(1, 2)
        Z0 = F.relu(self.lin(x))
        Z = Z0
        for _ in range(self.K):
            Z = (1 - self.alpha) * torch.bmm(An, Z) + self.alpha * Z0
        return Z


class MixHop(nn.Module):
    def __init__(self, in_dim, out_dim, powers=(0, 1, 2)):
        super().__init__()
        self.powers = powers
        self.lin = nn.Linear(in_dim * len(powers), out_dim)

    def forward(self, x, A):
        B, N, F = x.shape
        A = _ensure_batch_adj(A, B)
        outs = []
        Ap = torch.eye(N, device=x.device).unsqueeze(0).expand(B, N, N)
        for p in range(max(self.powers) + 1):
            if p in self.powers:
                outs.append(torch.bmm(Ap, x))
            Ap = torch.bmm(Ap, A)
        y = self.lin(torch.cat(outs, dim=-1))
        return F.relu(y)


class NodeMixer(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, x, A=None):
        # x:[B,N,F]; mix across nodes by projecting N as tokens -> depthwise 1D conv via einsum
        # Centering improves stability
        x_c = x - x.mean(dim=1, keepdim=True)
        z = self.mlp(x_c)  # feature mixing
        token_mean = z.mean(
            dim=2, keepdim=True
        )  # reduce features, broadcast over nodes
        return x + token_mean  # residual


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.size(0), 1, 1).bernoulli_(keep) / keep
        return x * mask


class GraphBlock(nn.Module):
    def __init__(self, gop, dim, mlp_ratio=2.0, droppath=0.0, prenorm=True):
        super().__init__()
        self.gop = PreNorm(dim, gop) if prenorm else gop
        self.dp1 = DropPath(droppath)
        self.mlp = nn.Sequential(
            GraphNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dp2 = DropPath(droppath)

    def forward(self, x, A=None):
        x = x + self.dp1(self.gop(x, A))
        x = x + self.dp2(self.mlp(x))
        return x


class AdaptiveChebBlock(nn.Module):
    """
    Stabilized Chebyshev graph block with:
    - dynamic cosine adjacency when A is None (learnable temperature)
    - top-k sparsification
    - renormalization + teleport (shrink toward I)
    - residual gating (learnable scalar)
    - light bottleneck MLP
    """
    def __init__(self, in_dim, hidden=None, out_dim=None, K=2, topk=16,
                 tau_init=0.5, teleport=0.1, gate_init=0.2):
        super().__init__()
        self.K = K
        self.topk = topk
        self.teleport = teleport

        hidden = hidden or in_dim
        out_dim = out_dim or in_dim

        # feature bottleneck → graph mix → expand
        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU()
        )
        self.lin = nn.Linear(hidden*K, out_dim)

        # learnable temperature for cosine sim
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
        # residual gate
        self.gate = nn.Parameter(torch.tensor(gate_init))

    @staticmethod
    def _cosine_topk(x, topk):
        # x:[B,N,F] -> A:[B,N,N] (non-negative)
        x = F.normalize(x, p=2, dim=-1)
        A = torch.einsum("bnf,bmf->bnm", x, x).clamp_min(0)

        if topk and topk < A.size(-1):
            vals, idx = torch.topk(A, k=topk, dim=-1)
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            A = A.masked_fill(~mask, 0.0)

        # add self-loops
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, N, N)
        return A + I

    def _normalized(self, A):
        # symmetric normalization + teleport toward I
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, N, N)
        A = (1 - self.teleport) * A + self.teleport * I

        deg = A.sum(-1).clamp_min(1e-6)
        Dm12 = deg.pow(-0.5).unsqueeze(-1)
        return Dm12 * A * Dm12.transpose(1, 2)  # Ā

    def forward(self, x, A=None):
        # x:[B,N,F], A:[B,N,N] or None
        residual = x
        x = self.pre(x)  # [B,N,H]

        # adjacency
        if A is None:
            tau = self.log_tau.exp().clamp_min(1e-3)
            # temperature shrinkage via softmax along neighbors (optional but stabilizing)
            A = self._cosine_topk(x, self.topk)
            A = A / tau  # scale logits before normalization

        Ahat = self._normalized(A)

        # Chebyshev over Ā (use powers of Ā instead of Laplacian; numerically stable & simple)
        T0 = x
        outs = [T0]
        if self.K > 1:
            T1 = torch.bmm(Ahat, T0); outs.append(T1)
        for _ in range(2, self.K):
            T_next = 2*torch.bmm(Ahat, outs[-1]) - outs[-2]
            outs.append(T_next)

        h = torch.cat(outs, dim=-1)  # [B,N,H*K]
        y = self.lin(h)              # [B,N,out_dim]

        # gated residual
        return residual + self.gate.tanh() * y
