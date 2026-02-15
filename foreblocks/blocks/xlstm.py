import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foreblocks.blocks.enc_dec import DecoderBase
from foreblocks.blocks.enc_dec import EncoderBase
from foreblocks.ui.node_spec import node


# ==============================================================
# Utilities
# ==============================================================


def _zeros_like_param(shape, device):
    return torch.zeros(*shape, device=device, requires_grad=False)


def _stack_states_per_layer(states: List[Tensor]) -> Tensor:
    # stacks list length = num_layers into [num_layers, B, ...]
    return torch.stack(states, dim=0)


# ==============================================================
# sLSTM Building Block (paper-faithful)
# States per unit: c_t, n_t, m_t (log-stabilizer); output gate o_t
# ==============================================================


class sLSTMLayer(nn.Module):
    """
    Paper-faithful sLSTM cell.

    Equations (sketch):
      i_t = exp(tilde_i_t),   f_t = sigmoid(tilde_f_t)  (or exp; we use sigmoid for f)
      m_t = max( log f_t + m_{t-1},  tilde_i_t )
      i'_t = exp(tilde_i_t - m_t),   f'_t = exp(log f_t + m_{t-1} - m_t)

      z_t = tanh(W_z x_t + U_z h_{t-1} + b_z)
      c_t = f'_t * c_{t-1} + i'_t * z_t
      n_t = f'_t * n_{t-1} + i'_t
      h_t = o_t * (c_t / clamp(n_t, min=1e-6)),  o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)

    Notes:
      - No tanh on (c_t/n_t) path; φ=tanh is applied only to z_t (cell input).
      - m_t is a log-domain stabilizer; keep requires_grad=False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        forget_as_exp: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.forget_as_exp = forget_as_exp

        # Input/recurrent projections for gates and z
        self.i_x = nn.Linear(input_size, hidden_size, bias=bias)
        self.i_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.f_x = nn.Linear(input_size, hidden_size, bias=bias)
        self.f_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.z_x = nn.Linear(input_size, hidden_size, bias=bias)
        self.z_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.o_x = nn.Linear(input_size, hidden_size, bias=bias)
        self.o_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Xavier for weights; gate biases: f bias > 0 to prefer remembering; others ~0
        for mod in [
            self.i_x,
            self.i_h,
            self.f_x,
            self.f_h,
            self.z_x,
            self.z_h,
            self.o_x,
            self.o_h,
        ]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

        # Forget gate bias to positive values
        if self.f_x.bias is not None:
            self.f_x.bias.data.fill_(2.0)

    def forward(
        self,
        x: Tensor,  # [B, D]
        state: Tuple[
            Tensor, Tensor, Tensor, Tensor
        ],  # (h_prev, c_prev, n_prev, m_prev)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        h_prev, c_prev, n_prev, m_prev = state  # each [B, H] except m: [B, H]

        # Gates pre-activations
        i_tilde = self.i_x(x) + self.i_h(h_prev)  # [B, H]
        f_affine = self.f_x(x) + self.f_h(h_prev)  # [B, H]
        z_affine = self.z_x(x) + self.z_h(h_prev)  # [B, H]
        o_affine = self.o_x(x) + self.o_h(h_prev)  # [B, H]

        # Parametrizations
        i_lin = i_tilde  # log(i) = i_tilde  => i = exp(i_tilde)
        if self.forget_as_exp:
            f_lin = f_affine  # log(f) = f_affine => f = exp(f_affine)
            log_f = f_lin
        else:
            f_val = torch.sigmoid(f_affine)
            # Avoid log(0)
            log_f = torch.log(torch.clamp(f_val, min=1e-8))

        # Stabilization
        # m_t = max(log f_t + m_{t-1}, i_tilde)
        m_t = torch.maximum(log_f + m_prev, i_lin)

        i_prime = torch.exp(i_lin - m_t)  # [B, H]
        f_prime = torch.exp(log_f + m_prev - m_t)  # [B, H]

        # Cell/input/update
        z = torch.tanh(z_affine)
        c_t = f_prime * c_prev + i_prime * z
        n_t = f_prime * n_prev + i_prime

        # Output
        denom = torch.clamp(n_t, min=1e-6)
        h_tilde = c_t / denom
        o = torch.sigmoid(o_affine)
        h_t = o * h_tilde

        return h_t, (h_t, c_t, n_t, m_t)


# ==============================================================
# mLSTM Building Block (paper-faithful)
# Per head states: C_t in R[Dh,Dh], n_t in R[Dh,1], m_t (scalar per head)
# ==============================================================


class mLSTMLayer(nn.Module):
    """
    Paper-faithful mLSTM cell (multi-head).
    - q = W_q x
    - k = (W_k x) / sqrt(Dh)
    - v = W_v x
    - gates (i,f) from x (num_heads each), with log-stabilized i', f'
    - C_t = f' * C_{t-1} + i' * (k v^T)
    - n_t = f' * n_{t-1} + i' * k
    - readout per head: h~ = (C_t q) / max(|n_t^T q|, 1)
    - output gate o = sigmoid(W_o x), h = o ⊙ concat_heads(h~)
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int = 4, bias: bool = True
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.bias = bias

        # q, k, v projections
        self.q_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(input_size, hidden_size, bias=bias)

        # gates from x -> [B, NH]
        self.i_proj = nn.Linear(input_size, num_heads, bias=bias)
        self.f_proj = nn.Linear(input_size, num_heads, bias=bias)

        # output gate
        self.o_proj = nn.Linear(input_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for proj in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.i_proj,
            self.f_proj,
            self.o_proj,
        ]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Prefer remembering at init
        if self.f_proj.bias is not None:
            self.f_proj.bias.data.fill_(2.0)

    def forward(
        self,
        x: Tensor,  # [B, D]
        mlstm_state: Tuple[
            Tensor, Tensor, Tensor
        ],  # (c_prev [B,NH,Dh,Dh], n_prev [B,NH,Dh,1], m_prev [B,NH,1,1])
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        B = x.size(0)
        c_prev, n_prev, m_prev = mlstm_state  # shapes per comment

        # Projections
        q = self.q_proj(x)  # [B, H]
        k = self.k_proj(x) / math.sqrt(self.head_dim)  # [B, H]
        v = self.v_proj(x)  # [B, H]

        # Reshape per head
        q_h = q.view(B, self.num_heads, self.head_dim)  # [B, NH, Dh]
        k_h = k.view(B, self.num_heads, self.head_dim)  # [B, NH, Dh]
        v_h = v.view(B, self.num_heads, self.head_dim)  # [B, NH, Dh]

        # Gates (log-param for i; sigmoid for f -> log f)
        i_lin = self.i_proj(x)  # [B, NH]
        f_affine = self.f_proj(x)  # [B, NH]
        f_val = torch.sigmoid(f_affine)
        log_f = torch.log(torch.clamp(f_val, min=1e-8))  # [B, NH]

        # Stabilization per head
        s_prev = m_prev.squeeze(-1).squeeze(-1)  # [B, NH]
        s_new = torch.maximum(log_f + s_prev, i_lin)  # [B, NH]
        i_prime = torch.exp(i_lin - s_new).unsqueeze(-1)  # [B, NH, 1]
        f_prime = torch.exp(log_f + s_prev - s_new).unsqueeze(-1)  # [B, NH, 1]

        # Update C and n
        kv_outer = k_h.unsqueeze(-1) * v_h.unsqueeze(-2)  # [B, NH, Dh, Dh]
        c_new = f_prime.unsqueeze(-1) * c_prev + i_prime.unsqueeze(-1) * kv_outer
        n_new = f_prime * n_prev + i_prime * k_h.unsqueeze(-1)  # [B, NH, Dh, 1]

        # Readout per head:
        # denom = max(|n^T q|, 1)
        nq = torch.matmul(n_new.squeeze(-1), q_h.unsqueeze(-1)).squeeze(-1)  # [B, NH]
        denom = torch.maximum(nq.abs(), torch.ones_like(nq))  # [B, NH]
        h_tilde_h = torch.matmul(c_new, q_h.unsqueeze(-1)).squeeze(
            -1
        ) / denom.unsqueeze(
            -1
        )  # [B,NH,Dh]

        # Output gate and merge heads
        o = torch.sigmoid(self.o_proj(x))  # [B, H]
        h_tilde = (
            h_tilde_h.transpose(1, 2).contiguous().view(B, self.hidden_size)
        )  # [B, H]
        h = o * h_tilde

        # New stabilizer (keep shape [B,NH,1,1] for consistency)
        m_new = s_new.unsqueeze(-1).unsqueeze(-1)

        return h, (c_new, n_new, m_new)


# ==============================================================
# sLSTM Encoder / Decoder
# ==============================================================


@node(
    type_id="slstm_encoder",
    name="sLSTM Encoder",
    category="Encoder",
    color="bg-gradient-to-br from-purple-700 to-purple-800",
    outputs=["encoder"],
    inputs=[],
)
class sLSTMEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,  # not supported
        forget_as_exp: bool = False,  # set True to use exp for f gate too
    ):
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Bidirectional sLSTM encoder not implemented.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forget_as_exp = forget_as_exp

        self.layers = nn.ModuleList(
            [
                sLSTMLayer(
                    input_size if l == 0 else hidden_size,
                    hidden_size,
                    bias=True,
                    forget_as_exp=forget_as_exp,
                )
                for l in range(num_layers)
            ]
        )
        self.dropout_layer = (
            nn.Dropout(dropout) if (dropout and num_layers > 1) else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,  # [B, T, D]
        hidden: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        time_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        B, T, _ = x.shape
        device = x.device

        # Initialize states per layer: h, c, n, m
        layer_h = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_c = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_n = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_m = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]

        if hidden is not None:
            h0, c0, n0, m0 = hidden
            for l in range(self.num_layers):
                layer_h[l].copy_(h0[l])
                layer_c[l].copy_(c0[l])
                layer_n[l].copy_(n0[l])
                layer_m[l].copy_(m0[l])

        outputs = torch.zeros(B, T, self.hidden_size, device=device)
        last_layer_h_list = [None] * self.num_layers

        for t in range(T):
            xt = x[:, t, :]
            prev_h = None
            tmp_h_per_layer = []
            for l in range(self.num_layers):
                inp = xt if l == 0 else prev_h
                h_new, (h_new_full, c_new, n_new, m_new) = self.layers[l](
                    inp, (layer_h[l], layer_c[l], layer_n[l], layer_m[l])
                )
                # Update
                layer_h[l] = h_new_full
                layer_c[l] = c_new
                layer_n[l] = n_new
                layer_m[l] = m_new
                prev_h = h_new
                tmp_h_per_layer.append(h_new)

            outputs[:, t, :] = self.dropout_layer(prev_h)
            if t == T - 1:
                last_layer_h_list = tmp_h_per_layer

        h_final = _stack_states_per_layer(last_layer_h_list)  # [L,B,H]
        c_final = _stack_states_per_layer(layer_c)
        n_final = _stack_states_per_layer(layer_n)
        m_final = _stack_states_per_layer(layer_m)
        return outputs, (h_final, c_final, n_final, m_final)


@node(
    type_id="slstm_decoder",
    name="sLSTM Decoder",
    category="Decoder",
    color="bg-gradient-to-br from-indigo-700 to-indigo-800",
    outputs=["decoder"],
    inputs=[],
)
class sLSTMDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        output_size: int = 1,  # kept for compatibility (no direct projection here)
        num_layers: int = 1,
        dropout: float = 0.0,
        forget_as_exp: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                sLSTMLayer(
                    input_size if l == 0 else hidden_size,
                    hidden_size,
                    bias=True,
                    forget_as_exp=forget_as_exp,
                )
                for l in range(num_layers)
            ]
        )
        self.dropout_layer = (
            nn.Dropout(dropout) if (dropout and num_layers > 1) else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,  # [B,D] or [B,T,D]
        hidden: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,D]
        elif x.dim() != 3:
            raise ValueError(
                f"sLSTMDecoder input must be 2D or 3D, got {tuple(x.shape)}"
            )

        B, T, _ = x.shape
        device = x.device

        layer_h = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_c = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_n = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]
        layer_m = [
            _zeros_like_param((B, self.hidden_size), device)
            for _ in range(self.num_layers)
        ]

        if hidden is not None:
            h0, c0, n0, m0 = hidden
            for l in range(self.num_layers):
                layer_h[l].copy_(h0[l])
                layer_c[l].copy_(c0[l])
                layer_n[l].copy_(n0[l])
                layer_m[l].copy_(m0[l])

        last_layer_h_list = [None] * self.num_layers
        for t in range(T):
            xt = x[:, t, :]
            prev_h = None
            tmp_h_per_layer = []
            for l in range(self.num_layers):
                inp = xt if l == 0 else prev_h
                h_new, (h_new_full, c_new, n_new, m_new) = self.layers[l](
                    inp, (layer_h[l], layer_c[l], layer_n[l], layer_m[l])
                )
                layer_h[l] = h_new_full
                layer_c[l] = c_new
                layer_n[l] = n_new
                layer_m[l] = m_new
                prev_h = h_new
                tmp_h_per_layer.append(h_new)
            _ = self.dropout_layer(prev_h)
            if t == T - 1:
                last_layer_h_list = tmp_h_per_layer

        last_out = last_layer_h_list[-1]  # [B,H]
        h_final = _stack_states_per_layer(last_layer_h_list)  # [L,B,H]
        c_final = _stack_states_per_layer(layer_c)
        n_final = _stack_states_per_layer(layer_n)
        m_final = _stack_states_per_layer(layer_m)
        return last_out, (h_final, c_final, n_final, m_final)


# ==============================================================
# mLSTM Encoder / Decoder
# ==============================================================


@node(
    type_id="mlstm_encoder",
    name="mLSTM Encoder",
    category="Encoder",
    color="bg-gradient-to-br from-orange-700 to-orange-800",
    outputs=["encoder"],
    inputs=[],
)
class mLSTMEncoder(EncoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,  # not supported
    ):
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Bidirectional mLSTM encoder not implemented.")
        assert hidden_size % num_heads == 0, "hidden_size % num_heads != 0"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.layers = nn.ModuleList(
            [
                mLSTMLayer(
                    input_size if l == 0 else hidden_size,
                    hidden_size,
                    num_heads=num_heads,
                )
                for l in range(num_layers)
            ]
        )
        self.dropout_layer = (
            nn.Dropout(dropout) if (dropout and num_layers > 1) else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,  # [B,T,D]
        hidden: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        time_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        B, T, _ = x.shape
        device = x.device

        # Per-layer states
        layer_c = [
            _zeros_like_param((B, self.num_heads, self.head_dim, self.head_dim), device)
            for _ in range(self.num_layers)
        ]
        layer_n = [
            _zeros_like_param((B, self.num_heads, self.head_dim, 1), device)
            for _ in range(self.num_layers)
        ]
        layer_m = [
            _zeros_like_param((B, self.num_heads, 1, 1), device)
            for _ in range(self.num_layers)
        ]
        layer_h_last = [None] * self.num_layers

        if hidden is not None:
            h0, c0, n0, m0 = hidden
            for l in range(self.num_layers):
                layer_c[l].copy_(c0[l])
                layer_n[l].copy_(n0[l])
                layer_m[l].copy_(m0[l])

        outputs = torch.zeros(B, T, self.hidden_size, device=device)
        for t in range(T):
            xt = x[:, t, :]
            prev_h = None
            tmp_h = []
            for l in range(self.num_layers):
                inp = xt if l == 0 else prev_h
                h_new, (c_new, n_new, m_new) = self.layers[l](
                    inp, (layer_c[l], layer_n[l], layer_m[l])
                )
                layer_c[l], layer_n[l], layer_m[l] = c_new, n_new, m_new
                prev_h = h_new
                tmp_h.append(h_new)
            outputs[:, t, :] = self.dropout_layer(prev_h)
            if t == T - 1:
                layer_h_last = tmp_h

        h_final = _stack_states_per_layer(layer_h_last)  # [L,B,H]
        c_final = _stack_states_per_layer(layer_c)  # [L,B,NH,Dh,Dh]
        n_final = _stack_states_per_layer(layer_n)  # [L,B,NH,Dh,1]
        m_final = _stack_states_per_layer(layer_m)  # [L,B,NH,1,1]
        return outputs, (h_final, c_final, n_final, m_final)


@node(
    type_id="mlstm_decoder",
    name="mLSTM Decoder",
    category="Decoder",
    color="bg-gradient-to-br from-red-700 to-red-800",
    outputs=["decoder"],
    inputs=[],
)
class mLSTMDecoder(DecoderBase):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_heads: int = 4,
        output_size: int = 1,  # kept for compatibility; no projection here
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size % num_heads != 0"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                mLSTMLayer(
                    input_size if l == 0 else hidden_size,
                    hidden_size,
                    num_heads=num_heads,
                )
                for l in range(num_layers)
            ]
        )
        self.dropout_layer = (
            nn.Dropout(dropout) if (dropout and num_layers > 1) else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,  # [B,D] or [B,T,D]
        hidden: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(
                f"mLSTMDecoder input must be 2D or 3D, got {tuple(x.shape)}"
            )

        B, T, _ = x.shape
        device = x.device

        layer_c = [
            _zeros_like_param((B, self.num_heads, self.head_dim, self.head_dim), device)
            for _ in range(self.num_layers)
        ]
        layer_n = [
            _zeros_like_param((B, self.num_heads, self.head_dim, 1), device)
            for _ in range(self.num_layers)
        ]
        layer_m = [
            _zeros_like_param((B, self.num_heads, 1, 1), device)
            for _ in range(self.num_layers)
        ]
        layer_h_last = [None] * self.num_layers

        if hidden is not None:
            h0, c0, n0, m0 = hidden
            for l in range(self.num_layers):
                layer_c[l].copy_(c0[l])
                layer_n[l].copy_(n0[l])
                layer_m[l].copy_(m0[l])

        for t in range(T):
            xt = x[:, t, :]
            prev_h = None
            tmp_h = []
            for l in range(self.num_layers):
                inp = xt if l == 0 else prev_h
                h_new, (c_new, n_new, m_new) = self.layers[l](
                    inp, (layer_c[l], layer_n[l], layer_m[l])
                )
                layer_c[l], layer_n[l], layer_m[l] = c_new, n_new, m_new
                prev_h = h_new
                tmp_h.append(h_new)
            _ = self.dropout_layer(prev_h)
            if t == T - 1:
                layer_h_last = tmp_h

        last_out = layer_h_last[-1]  # [B,H]
        h_final = _stack_states_per_layer(layer_h_last)  # [L,B,H]
        c_final = _stack_states_per_layer(layer_c)  # [L,B,NH,Dh,Dh]
        n_final = _stack_states_per_layer(layer_n)  # [L,B,NH,Dh,1]
        m_final = _stack_states_per_layer(layer_m)  # [L,B,NH,1,1]
        return last_out, (h_final, c_final, n_final, m_final)
