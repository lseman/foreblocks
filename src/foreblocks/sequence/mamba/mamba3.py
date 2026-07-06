"""foreblocks.sequence.mamba.mamba3.

Mamba3-style SSM block without causal conv1d, with blockwise rotary on B/C.

Implements Mamba3 architecture: no causal convolution — B and C come directly
from in_proj. Input-dependent dt, A (softplus parametrized), trap, and rotary
angles all projected from input. B and C receive RMSNormGated then blockwise
rotary embedding before the chunked SSD scan. Includes autoregressive step.

Core API:
- Mamba3Block: Mamba3-style SSM block with blockwise rotary on B/C
- blockwise_rotary: blockwise rotary embedding on (i, i+num_angles) pairs

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ops.mamba import chunked_ssd_forward, dt_prep, fused_out, rms_norm
from foreblocks.sequence.mamba.norms import RMSNormWeightOnly
from foreblocks.sequence.mamba.utils import fused_out_2d

# ── helpers ──────────────────────────────────────────────────────────────


def blockwise_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise rotary on (i, i+num_angles) pairs.

    Mamba3 rotates the first ``2 * num_angles`` elements of the state dimension
    using learned per-token, per-head angles.  Pair index ``(i, i+num_angles)``
    where ``num_angles = num_rope_angles``.

    Args:
        q: ``[B, L, H, N]`` — C tensor (acts as Q in the SSM scan)
        k: ``[B, L, H, N]`` — B tensor (acts as K in the SSM scan)
        angles: ``[B, L, H, num_angles]`` — learned rotation angles

    Returns:
        Rotated ``(q_rot, k_rot)``, same shape.
    """
    num_angles = angles.shape[-1]
    if num_angles <= 0:
        return q, k

    sin_a = torch.sin(angles)
    cos_a = torch.cos(angles)

    q_rot = q.clone()
    k_rot = k.clone()

    first_q = q[..., :num_angles]
    second_q = q[..., num_angles : 2 * num_angles]
    q_rot[..., :num_angles] = first_q * cos_a - second_q * sin_a
    q_rot[..., num_angles : 2 * num_angles] = first_q * sin_a + second_q * cos_a

    first_k = k[..., :num_angles]
    second_k = k[..., num_angles : 2 * num_angles]
    k_rot[..., :num_angles] = first_k * cos_a - second_k * sin_a
    k_rot[..., num_angles : 2 * num_angles] = first_k * sin_a + second_k * cos_a

    return q_rot, k_rot


# ── Mamba3Block ──────────────────────────────────────────────────────────


class Mamba3Block(nn.Module):
    """Mamba3-style block (no conv, blockwise rotary on B/C).

    Key differences from Mamba2:

    * **No causal conv1d** — B and C come directly from ``in_proj``.
    * **dt / A / trap / angles** are direct linear projections from the
      input (all packed into one ``in_proj``).
    * **B / C** get a learnable per-head bias, then ``RMSNormGated``, then
      blockwise rotary embedding before the scan.
    * **A** = ``-softplus(dd_A)`` (softplus parametrisation instead of
      ``-exp(A_log)``).
    * **D** is per-head scalar (not per-head_dim).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,  # unused in Mamba3 but kept for API compat
        dt_rank: int | None = None,
        head_dim: int = 64,
        num_heads: int | None = None,
        n_groups: int = 1,
        chunk_size: int | None = 256,
        rope_fraction: float = 0.5,
        # dt range follows the official FLA Mamba3 reference (0.001, 0.1). The
        # SSM decay is adt = A*dt; a too-small dt (e.g. 1e-4) drives exp(adt)->1,
        # giving a near-non-decaying (cumsum-like) scan that is badly
        # ill-conditioned over long sequences.
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_min: float | None = None,
        dt_init_max: float | None = None,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        # Initial decay strength: A starts in [-A_init_max, -A_init_min] per head
        # (matches Mamba2's A_log uniform(1, 16) range) so the scan decays from
        # the start instead of behaving like an ill-conditioned cumsum.
        A_init_min: float = 1.0,
        A_init_max: float = 16.0,
        norm_eps: float = 1e-6,
        use_triton_ssd: bool = True,
        use_pre_norm: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.rope_fraction = rope_fraction
        self.A_floor = A_floor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.norm_eps = norm_eps
        self.use_triton_ssd = use_triton_ssd

        # ── Validate num_heads ─────────────────────────────────────────
        if num_heads is not None:
            if self.d_inner % num_heads != 0:
                raise ValueError("d_inner must be divisible by num_heads")
            self.num_heads = num_heads
        else:
            if self.d_inner % head_dim != 0:
                raise ValueError("d_inner must be divisible by head_dim")
            self.num_heads = self.d_inner // head_dim

        # ── Rotary angle dimensions ────────────────────────────────────
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2

        if self.num_rope_angles <= 0:
            raise ValueError(
                f"rope_fraction={rope_fraction} and d_state={d_state} "
                f"produce too few angles (need > 0)"
            )

        # ── in_proj dimension ──────────────────────────────────────────
        # Layout: [z, x, B, C, dd_dt, dd_A, trap, angles]
        # All come from one linear layer (FLA pattern)
        self.d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * n_groups
            + 3 * self.num_heads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, self.d_in_proj, bias=use_bias)

        # ── dt_bias (inverse softplus) ─────────────────────────────────
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        if dt_init_min is not None and dt_init_max is not None:
            dt = torch.exp(
                torch.rand(self.num_heads)
                * (math.log(dt_init_max) - math.log(dt_init_min))
                + math.log(dt_init_min)
            )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── A_floor (clamp for -softplus(dd_A)) ────────────────────────
        # Plain Python float: device-agnostic, unlike a CPU tensor that would
        # not follow the module to CUDA.
        self.A_floor_val = float(A_floor)

        # ── A bias (per-head) ──────────────────────────────────────────
        # dd_A from in_proj has no bias, so at init dd_A_raw ~ 0 →
        # A = -softplus(0) ≈ -0.69 (very weak decay). Combined with small dt
        # this makes exp(A*dt) ≈ 1, a near-non-decaying cumsum that is badly
        # ill-conditioned. We add a per-head bias so the *initial* A matches
        # Mamba2's strong-decay range A ∈ [-A_init_max, -A_init_min]:
        #   want softplus(A_bias) = a  ⇒  A_bias = inv_softplus(a)
        a0 = torch.empty(self.num_heads).uniform_(A_init_min, A_init_max)
        self.A_bias = nn.Parameter(a0 + torch.log(-torch.expm1(-a0)))  # inv_softplus
        self.A_bias._no_weight_decay = True

        # ── B / C bias ─────────────────────────────────────────────────
        bias_shape = (self.num_heads, self.d_state)
        self.B_bias = nn.Parameter(torch.ones(bias_shape, dtype=torch.float32))
        self.C_bias = nn.Parameter(torch.ones(bias_shape, dtype=torch.float32))

        # ── B / C RMSNorm weights (RMSNormGated: rms_norm(x,w)*silu(x)) ─
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state))

        # ── D skip (per-head scalar) ───────────────────────────────────
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
        self.D._no_weight_decay = True

        # ── RMSNorm + out ──────────────────────────────────────────────
        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        self.residual_proj = (
            nn.Identity()
            if d_model == self.d_inner
            else nn.Linear(d_model, self.d_inner, bias=False)
        )
        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()

        self.reset_parameters()

    # ── parameter initialisation ──────────────────────────────────────

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        residual_inner = self.residual_proj(self.pre_norm(x))

        # ── in_proj split (FLA pattern — all direct) ───────────────────
        p = self.in_proj(x)
        z, x_raw, BCraw, dd_dt_raw, dd_A_raw, trap_raw, angles_raw = torch.split(
            p,
            [
                self.d_inner,
                self.d_inner,
                2 * self.d_state * self.n_groups,  # B + C concatenated
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        Braw, Craw = torch.split(BCraw, self.d_state * self.n_groups, dim=-1)

        # ── Reshape z, x to head layout ────────────────────────────────
        # [B, L, H*P] → [B, L, H, P]
        z = z.view(z.shape[0], z.shape[1], self.num_heads, self.head_dim)
        u = x_raw.view(x_raw.shape[0], x_raw.shape[1], self.num_heads, self.head_dim)

        # ── Expand B / C groups → heads ────────────────────────────────
        # [B, L, ng*N] → [B, L, H, N]
        B = Braw.view(Braw.shape[0], Braw.shape[1], self.n_groups, self.d_state)
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = Craw.view(Craw.shape[0], Craw.shape[1], self.n_groups, self.d_state)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

        # ── dt / A / trap ──────────────────────────────────────────────
        dt = dt_prep(
            dd_dt_raw,
            self.dt_bias,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
        )  # [B, L, H]
        # A = -softplus(...) is negative; clamp the NEGATED value so A <= -A_floor.
        # (Parenthesised: `-softplus(x).clamp(max=-f)` would clamp the positive
        # softplus to <= -f, collapsing every A to +f — a non-decaying scan.)
        dd_A = (-F.softplus(dd_A_raw.float() + self.A_bias)).clamp(
            max=-self.A_floor_val
        )  # [B, L, H]
        trap = torch.sigmoid(trap_raw)  # [B, L, H]
        # ADT = A * dt (element-wise, both [B, L, H])
        adt = dd_A * dt  # [B, L, H]

        # ── Blockwise rotary on B/C ────────────────────────────────────
        # angles_raw: [B, L, num_angles] → [B, L, H, num_angles] (broadcast)
        angles = angles_raw.unsqueeze(2).expand(
            -1, -1, self.num_heads, -1
        )  # [B, L, H, num_angles]
        B_rot, C_rot = blockwise_rotary(B, C, angles)

        # ── RMSNormGated on B_rot / C_rot ──────────────────────────────
        # RMSNormGated(x, gate=x) = rms_norm(x, weight) * silu(x)
        B_rot = rms_norm(B_rot, self.B_norm_weight, eps=self.norm_eps) * F.silu(B_rot)
        C_rot = rms_norm(C_rot, self.C_norm_weight, eps=self.norm_eps) * F.silu(C_rot)

        # ── Chunked SSD scan ───────────────────────────────────────────
        # Q=C_rot, K=B_rot, V=u, with dt, A(=dd_A), trap, D
        # Pass adt = A * dt for inter-chunk decay (Mamba3: A is time-dependent)
        chunk_size = self.chunk_size or u.shape[1]
        y = chunked_ssd_forward(
            u=u,
            dt=dt,
            A=self.D,  # dummy; unused when adt is provided
            B=B_rot,
            C=C_rot,
            D=self.D,
            chunk_size=chunk_size,
            use_triton=self.use_triton_ssd,
            adt=adt,
            trap=trap,  # trapezoidal discretisation gate
        )  # [B, L, H*P]

        # ── norm + gate + out ──────────────────────────────────────────
        # y: [B, L, H*P], z: [B, L, H*P] (reshape back from [H, P])
        y_reshaped = y.reshape(y.shape[0], y.shape[1], self.d_inner)
        z_reshaped = z.reshape(z.shape[0], z.shape[1], self.d_inner)
        y = fused_out(y_reshaped, z_reshaped, self.norm.weight, eps=self.norm_eps)
        return self.out_proj(y)

    # ── helpers ───────────────────────────────────────────────────────

    def _expand_groups(self, flat: torch.Tensor) -> torch.Tensor:
        """Expand [B, L, ng*N] → [B, L, H, N] by repeating groups."""
        return flat.view(
            flat.shape[0], flat.shape[1], self.n_groups, self.d_state
        ).repeat_interleave(self.num_heads // self.n_groups, dim=2)

    # ── state management (for autoregressive decoding) ────────────────

    def make_state(self, batch: int, device=None, dtype=None) -> dict:
        return {
            "ssm": torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
            # previous-token rank-1 injection k_{t-1} = B_{t-1} ⊗ u_{t-1}, kept
            # for the trapezoidal blend (zero before the first token).
            "k_prev": torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.d_state,
                device=device,
                dtype=torch.float32,
            ),
        }

    def step(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Single-token autoregressive step (no chunking).

        ``x`` is ``[B, D]``; internally promoted to ``[B, 1, D]`` so the same
        ``view(Bsz, 1, ...)`` reshapes used elsewhere are valid.
        """
        Bsz = x.shape[0]
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        residual_inner = self.residual_proj(self.pre_norm(x))

        # ── in_proj split ──────────────────────────────────────────────
        p = self.in_proj(x)
        z, x_raw, BCraw, dd_dt_raw, dd_A_raw, trap_raw, angles_raw = torch.split(
            p,
            [
                self.d_inner,
                self.d_inner,
                2 * self.d_state * self.n_groups,
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        Braw, Craw = torch.split(BCraw, self.d_state * self.n_groups, dim=-1)

        # ── Reshape ────────────────────────────────────────────────────
        z = z.view(Bsz, 1, self.num_heads, self.head_dim)
        u = x_raw.view(Bsz, 1, self.num_heads, self.head_dim)
        B = self._expand_groups(Braw)  # [B, 1, H, N]
        C = self._expand_groups(Craw)  # [B, 1, H, N]

        # ── dt / A / trap ──────────────────────────────────────────────
        dt = dt_prep(dd_dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        # [B, 1, H]
        dd_A = (-F.softplus(dd_A_raw.float() + self.A_bias)).clamp(
            max=-self.A_floor_val
        )
        trap = torch.sigmoid(trap_raw)  # [B, 1, H]

        # ── Blockwise rotary on B/C ────────────────────────────────────
        angles = (
            angles_raw.squeeze(1)
            .view(Bsz, self.num_rope_angles)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, 1, self.num_heads, self.num_rope_angles)
        )
        B, C = blockwise_rotary(B, C, angles)

        # ── RMSNormGated on B/C ────────────────────────────────────────
        B = rms_norm(B, self.B_norm_weight, eps=self.norm_eps) * F.silu(B)
        C = rms_norm(C, self.C_norm_weight, eps=self.norm_eps) * F.silu(C)

        # ── SSM step ───────────────────────────────────────────────────
        h = state["ssm"].to(dtype=torch.float32)  # [B, H, P, N]

        # Transpose to [B, H, ...] layout for clean broadcasting
        dt_h = dt.transpose(1, 2)  # [B, H, 1]
        dd_A_h = dd_A.transpose(1, 2)  # [B, H, 1]
        B_h = B.transpose(1, 2).squeeze(2)  # [B, H, N]
        C_h = C.transpose(1, 2).squeeze(2)  # [B, H, N]
        u_h = u.transpose(1, 2).squeeze(2)  # [B, H, P]

        abar = torch.exp(
            dt_h[:, :, :, None, None] * dd_A_h[:, :, :, None, None]
        )  # [B, H, 1, 1, 1]

        # k_t = B_t ⊗ u_t (rank-1 injection, pre-dt) — [B, H, P, N]
        k_cur = B_h[:, :, None, :] * u_h[:, :, :, None]
        # Trapezoidal blend: dt * [trap*k_t + (1-trap)*k_{t-1}] (trap=1 → Euler)
        trap_h = trap.transpose(1, 2)  # [B, H, 1]
        k_prev = state["k_prev"].to(dtype=torch.float32)
        blended = trap_h[:, :, :, None] * k_cur + (1.0 - trap_h)[:, :, :, None] * k_prev
        dB = dt_h[:, :, :, None] * blended  # [B, H, P, N]

        h = abar.squeeze(-1) * h + dB  # [B, H, P, N]
        state["k_prev"] = k_cur.detach()
        # y = C*h + D*u — both must have 4 dims for correct broadcasting
        y = (C_h[:, :, None, :] * h).sum(dim=-1).unsqueeze(1) + self.D[
            None, None, :, None
        ] * u  # [1, 1, H, 1] * [B, 1, H, P] → [B, 1, H, P]
        state["ssm"] = h.detach()
        # y: [B, H, P] → [B, 1, H*P]

        # ── norm + gate + out ──────────────────────────────────────────
        y = y.reshape(Bsz, 1, self.d_inner)
        y_normed = fused_out_2d(
            y,
            z.reshape(Bsz, 1, self.d_inner),
            self.norm.weight,
            eps=self.norm_eps,
        )
        return self.out_proj(y_normed)
