import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWarp(nn.Module):
    """
    Adaptive warp module:
      Δτ_t = Softplus(w^T h_t)
      τ_t = cumsum(Δτ_t)
    Matches Eq. (5). :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, d_model)
        dtau = F.softplus(self.w(h)).squeeze(-1)  # (B, N), strictly positive
        tau = torch.cumsum(dtau, dim=1)  # (B, N)
        return tau


class SyPERotator(nn.Module):
    """
    Implements the symplectic flow per 2D pair and applies it to q and k as in:
      q~_t = S(τ_t) q_t
      k~_t = J S(τ_t) k_t
    See SyPE mechanism around Eq. (6). :contentReference[oaicite:4]{index=4}

    Closed form:
      S(t) = cos(ωt) I + sin(ωt)/ω * A,  with A=JK  :contentReference[oaicite:5]{index=5}
    Stable parameterization:
      a=exp(α), b=exp(β), c=ρ sqrt(ab), ρ=tanh(γ)  :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0, "SyPE needs head_dim even (pairs of 2)."
        self.head_dim = head_dim
        self.n_pairs = head_dim // 2

        # Learn K per pair: K = [[a,c],[c,b]] with stable params (α,β,γ)
        self.alpha = nn.Parameter(torch.zeros(self.n_pairs))
        self.beta = nn.Parameter(torch.zeros(self.n_pairs))
        self.gamma = nn.Parameter(torch.zeros(self.n_pairs))

    def _params(self):
        a = torch.exp(self.alpha)  # (P,)
        b = torch.exp(self.beta)  # (P,)
        rho = torch.tanh(self.gamma)  # (P,) in (-1,1)
        c = rho * torch.sqrt(a * b)  # (P,)
        # ω^2 = ab - c^2 = ab(1-rho^2) > 0
        omega = torch.sqrt(a * b - c * c + 1e-12)  # (P,)
        return a, b, c, omega

    @staticmethod
    def _apply_2x2(M11, M12, M21, M22, x):
        """
        Apply 2x2 matrix M to x in R^2, batched.
        x: (..., 2)
        Returns: (..., 2)
        """
        x1, x2 = x[..., 0], x[..., 1]
        y1 = M11 * x1 + M12 * x2
        y2 = M21 * x1 + M22 * x2
        return torch.stack([y1, y2], dim=-1)

    def _flow(self, tau: torch.Tensor):
        """
        Build S(τ) for all tokens and pairs using the closed form.
        tau: (B, N) or (B, N, 1)
        Returns matrix entries (S11,S12,S21,S22) each of shape (B, N, P)
        """
        a, b, c, omega = self._params()  # (P,)
        # A = JK = [[ c,  b],
        #           [-a, -c]]  :contentReference[oaicite:7]{index=7}
        # Broadcast shapes:
        #   tau: (B,N,1), omega: (1,1,P)
        if tau.dim() == 2:
            tau = tau.unsqueeze(-1)  # (B,N,1)

        omega_ = omega.view(1, 1, self.n_pairs)  # (1,1,P)
        wt = tau * omega_  # (B,N,P)

        cos = torch.cos(wt)
        sin = torch.sin(wt)
        inv_omega = 1.0 / (omega_ + 1e-12)

        # S = cos I + (sin/ω) A
        # => S11 = cos + (sin/ω)*c
        #    S12 =       (sin/ω)*b
        #    S21 =       (sin/ω)*(-a)
        #    S22 = cos + (sin/ω)*(-c)
        a_ = a.view(1, 1, self.n_pairs)
        b_ = b.view(1, 1, self.n_pairs)
        c_ = c.view(1, 1, self.n_pairs)

        coef = sin * inv_omega
        S11 = cos + coef * c_
        S12 = coef * b_
        S21 = coef * (-a_)
        S22 = cos + coef * (-c_)
        return S11, S12, S21, S22

    def rotate_qk(self, q: torch.Tensor, k: torch.Tensor, tau: torch.Tensor):
        """
        q,k: (B, H, N, Dh)
        tau: (B, N)

        Returns:
          q_tilde, k_tilde with same shapes as q,k
        """
        B, H, N, Dh = q.shape
        assert Dh == self.head_dim

        # reshape into 2D pairs: (B,H,N,P,2)
        q2 = q.view(B, H, N, self.n_pairs, 2)
        k2 = k.view(B, H, N, self.n_pairs, 2)

        S11, S12, S21, S22 = self._flow(tau)  # (B,N,P) each

        # expand to (B,1,N,P) so it broadcasts over heads
        S11 = S11.unsqueeze(1)
        S12 = S12.unsqueeze(1)
        S21 = S21.unsqueeze(1)
        S22 = S22.unsqueeze(1)

        # q~ = S q
        q_tilde2 = self._apply_2x2(S11, S12, S21, S22, q2)

        # k~ = J S k, where J=[[0,1],[-1,0]]  :contentReference[oaicite:8]{index=8}
        # First u = S k
        u2 = self._apply_2x2(S11, S12, S21, S22, k2)
        # Then J u = [u2, -u1]
        k_tilde2 = torch.stack([u2[..., 1], -u2[..., 0]], dim=-1)

        # back to (B,H,N,Dh)
        q_tilde = q_tilde2.reshape(B, H, N, Dh)
        k_tilde = k_tilde2.reshape(B, H, N, Dh)
        return q_tilde, k_tilde
