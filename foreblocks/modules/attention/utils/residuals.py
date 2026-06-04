"""Attention Residuals — depth-wise softmax attention over layer outputs.

Implements two variants of the Attention Residual (AttnRes) mechanism from
Chen et al. (Kimi Team, 2026), which replaces fixed-unit residual accumulation
with learned, content-dependent softmax attention over preceding layer outputs.

Original paper:
    Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S., Wang, Y., … Zhou, X.
    (2026).
    "Attention Residuals."
    arXiv:2603.15031v1 [[arXiv]](https://arxiv.org/abs/2603.15031)

Key ideas:
    1. *Full AttnRes* (§2): a single learnable depth-query attends over every
       preceding layer output, replacing the fixed ``h_prev + o`` accumulation.
    2. *Block AttnRes* (§3): layers are partitioned into blocks; attention is
       computed over block-level representations, reducing memory/communication
       overhead while preserving most of the gains.
    3. Both variants use an RMSNorm on the stacked history before attention,
       and a single learnable query vector ``q ∈ R^D`` for all layers.

Classes
-------
AttentionResidual
    Full AttnRes — attends over all preceding layer outputs.
BlockAttentionResidual
    Block AttnRes — attends over block-level representations.
"""

import torch
import torch.nn.functional as F
from torch import nn

from foreblocks.layers.norms import RMSNorm


def normalize_attention_residual_mode(attn_residual_type: str) -> str:
    r"""Normalize public residual mode strings.

    Parameters
    ----------
    attn_residual_type : str
        One of ``"full"`` or ``"block"``.

    Returns
    -------
    str
        Normalised mode string.

    Raises
    ------
    ValueError
        If the mode is not recognised.
    """
    mode = str(attn_residual_type).strip().lower()
    if mode == "block":
        return "block"
    if mode == "full":
        return "full"
    raise ValueError(
        "attn_residual_type must be one of {'full', 'block'}; "
        f"got {attn_residual_type!r}"
    )


class AttentionResidual(nn.Module):
    r"""Full Attention Residuals (AttnRes) — softmax attention over depth.

    Maintains a history of previous layer outputs and replaces the fixed
    residual accumulation (``h = h_prev + o``) with a content-dependent,
    learnable softmax-weighted aggregation:

    .. math::

        h = \sum_{i=0}^{l} \mathrm{softmax}_i(q^\top \mathrm{Norm}(v_i)) \cdot v_i

    where ``q ∈ R^D`` is a learnable depth-query, ``v_i`` are the outputs of
    preceding layers, and ``Norm`` is an RMSNorm applied to the stacked
    history.

    Reference:
        Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S., Wang, Y., … Zhou, X.
        (2026).
        "Attention Residuals."
        arXiv:2603.15031v1 [[arXiv]](https://arxiv.org/abs/2603.15031)
        See §2 "Attention Residuals" and §3 "Block AttnRes".

    Parameters
    ----------
    dim : int
        Hidden dimension (also used for the learnable depth query ``q``).
    scale : float | None
        Factor applied to the depth logits ``q^T K`` before the softmax. The
        raw dot product grows with ``dim``, which makes the depth-softmax
        saturate toward a one-hot distribution; scaling keeps the logits in a
        stable range. Defaults to ``dim ** -0.5``.

    Attributes
    ----------
    query : torch.nn.Parameter
        Learnable depth-query vector ``q ∈ R^D``.
    norm : RMSNorm
        Per-token RMSNorm applied to the stacked history before attention.
    """

    def __init__(self, dim, scale=None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # w_l
        self.norm = RMSNorm(dim)
        self.scale = dim ** -0.5 if scale is None else scale

    def forward(self, history, return_weights=False):
        r"""Aggregate previous layer outputs via softmax attention over depth.

        Parameters
        ----------
        history : list[torch.Tensor]
            Layer outputs ``[v_0, v_1, ..., v_{l-1}]``, each shaped
            ``[B, T, D]``.
        return_weights : bool
            If ``True``, also return the depth-softmax weights ``α`` shaped
            ``[L, B, T]``. Default: ``False``.

        Returns
        -------
        torch.Tensor
            Weighted sum of history, shaped ``[B, T, D]``.
        torch.Tensor
            Depth-softmax weights ``[L, B, T]``, only if ``return_weights``.

        Notes
        -----
        The computation follows Equation 2 of Chen et al. (2026):

        1. Stack history into ``V ∈ R^{L×B×T×D}``.
        2. Apply RMSNorm to keys: ``K = Norm(V)``.
        3. Compute scaled logits ``scale · q^T K ∈ R^{L×B×T}``.
        4. Softmax over the depth dimension ``L``.
        5. Weighted sum ``h = Σ_i α_i · v_i``.
        """

        # [L, B, T, D]
        V = torch.stack(history, dim=0)

        # normalize keys
        K = self.norm(V)

        # compute scaled logits: scale * q^T k
        # [L, B, T]
        logits = torch.einsum("d, l b t d -> l b t", self.query, K) * self.scale

        # softmax over depth (L dimension)
        attn = F.softmax(logits, dim=0)

        # weighted sum
        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        if return_weights:
            return h, attn
        return h


class BlockAttentionResidual(nn.Module):
    r"""Block Attention Residuals (Block AttnRes) — partitioned depth-wise attention.

    Reduces memory and communication overhead by partitioning layers into
    blocks and attending over block-level representations instead of
    individual layers.

    .. math::

        h = \sum_{j} \mathrm{softmax}_j(q^\top \mathrm{Norm}(b_j)) \cdot b_j

    where ``b_j`` are block-level accumulations, each of which may itself
    contain multiple layer outputs (the ``partial`` argument allows the
    current block's running accumulation to be included).

    Reference:
        Chen, G., Zhang, Y., Su, J., Xu, W., Pan, S., Wang, Y., … Zhou, X.
        (2026).
        "Attention Residuals."
        arXiv:2603.15031v1 [[arXiv]](https://arxiv.org/abs/2603.15031)
        See §3 "Block AttnRes".

    Parameters
    ----------
    dim : int
        Hidden dimension (also used for the learnable depth query ``q``).
    scale : float | None
        Factor applied to the depth logits ``q^T K`` before the softmax. The
        raw dot product grows with ``dim``, which makes the depth-softmax
        saturate toward a one-hot distribution; scaling keeps the logits in a
        stable range. Defaults to ``dim ** -0.5``.

    Attributes
    ----------
    query : torch.nn.Parameter
        Learnable depth-query vector ``q ∈ R^D``.
    norm : RMSNorm
        Per-token RMSNorm applied to the stacked block history.
    """

    def __init__(self, dim, scale=None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim)
        self.scale = dim ** -0.5 if scale is None else scale

    def forward(self, blocks, partial=None, return_weights=False):
        r"""Aggregate block-level representations via softmax attention over depth.

        Parameters
        ----------
        blocks : list[torch.Tensor]
            Block-level accumulations ``[b_0, b_1, ..., b_{n-1}]``, each
            shaped ``[B, T, D]``.
        partial : torch.Tensor | None
            Current block's running accumulation, or ``None`` if this is
            the first sub-layer in a block (only completed blocks visible).
        return_weights : bool
            If ``True``, also return the depth-softmax weights ``α`` shaped
            ``[L, B, T]``. Default: ``False``.

        Returns
        -------
        torch.Tensor
            Weighted sum, shaped ``[B, T, D]``.
        torch.Tensor
            Depth-softmax weights ``[L, B, T]``, only if ``return_weights``.

        Raises
        ------
        ValueError
            If ``len(blocks) == 0`` and ``partial is None``.

        Notes
        -----
        The computation follows Equation 3 of Chen et al. (2026):

        1. Stack ``blocks`` (and optionally ``partial``) into ``V ∈
           R^{L×B×T×D}``.
        2. Apply RMSNorm to keys: ``K = Norm(V)``.
        3. Compute scaled logits ``scale · q^T K ∈ R^{L×B×T}``.
        4. Softmax over the depth dimension ``L``.
        5. Weighted sum ``h = Σ_i α_i · v_i``.
        """

        values = list(blocks)
        if partial is not None:
            values.append(partial)
        if not values:
            raise ValueError(
                "BlockAttentionResidual requires at least one value tensor."
            )

        V = torch.stack(values, dim=0)
        K = self.norm(V)

        logits = torch.einsum("d, l b t d -> l b t", self.query, K) * self.scale
        attn = F.softmax(logits, dim=0)

        h = torch.einsum("l b t, l b t d -> b t d", attn, V)

        if return_weights:
            return h, attn
        return h
