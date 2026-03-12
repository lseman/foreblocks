"""
transformer_diagram.py
======================
Render publication-quality transformer block diagrams with matplotlib.

Features
--------
- Encoder-only, decoder-only, or encoder+decoder side-by-side
- Pre-norm  vs  Post-norm  layout
- Pluggable attention : mha | gated_attn | linear_attn | deltanet | ...
- Pluggable FFN        : feedforward | moe | swiglu | geglu | ...
- Pluggable norm       : rmsnorm | layernorm
- Correct L-shaped residual skip connections
- N× repeat badge with bracket
- Cross-attention arrows for encoder→decoder

Quick start
-----------
    from transformer_diagram import (
        make_encoder_layers, make_decoder_layers,
        draw_single_block, draw_encoder_decoder
    )

    # GPT-style decoder, pre-norm, SwiGLU FFN
    layers = make_decoder_layers(
        attn_type="mha", ffn_type="swiglu",
        norm_type="rmsnorm", norm_pos="pre"
    )
    fig, ax = draw_single_block(layers, title="GPT Block", num_repeats=32)
    fig.savefig("gpt.png", dpi=150, bbox_inches="tight")
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "embedding":    "#B8D4E8",
    "norm":         "#D5E8D4",
    "self_attn":    "#DAD5EA",
    "cross_attn":   "#F9D5A7",
    "linear_attn":  "#E8D5E8",
    "feedforward":  "#FFF2CC",
    "moe":          "#FFE6CC",
    "swiglu":       "#FFF2CC",
    "geglu":        "#FFEEDD",
    "output":       "#B8D4E8",
    "add":          "#FFFFFF",
    "generic":      "#E8E8E8",
    "encoder_bg":   "#EEF2F8",
    "decoder_bg":   "#F5EEF8",
    "block_border": "#8899AA",
    "label":        "#1A1A2E",
    "arrow":        "#445566",
    "dim_label":    "#667788",
    "repeat_badge": "#3B4A6B",
    "cross_arrow":  "#CC6600",
    "residual":     "#556677",
}

_TYPE_COLOR = {
    "embedding":   COLORS["embedding"],
    "tokenizer":   COLORS["embedding"],
    "query_input": "#CDE7E2",
    "memory":      "#FDE6C9",
    "rmsnorm":     COLORS["norm"],
    "layernorm":   COLORS["norm"],
    "norm":        COLORS["norm"],
    "self_attn":   COLORS["self_attn"],
    "sdp":         COLORS["self_attn"],
    "linear":      COLORS["linear_attn"],
    "probsparse":  "#E7D9F5",
    "cosine":      "#DCECF7",
    "local":       "#F0DCE8",
    "mha":         COLORS["self_attn"],
    "gated_attn":  COLORS["self_attn"],
    "linear_attn": COLORS["linear_attn"],
    "deltanet":    COLORS["linear_attn"],
    "cross_attn":  COLORS["cross_attn"],
    "feedforward": COLORS["feedforward"],
    "ffn":         COLORS["feedforward"],
    "swiglu":      COLORS["swiglu"],
    "geglu":       COLORS["geglu"],
    "moe":         COLORS["moe"],
    "output":      COLORS["output"],
    "linear":      COLORS["output"],
    "add":         COLORS["add"],
}

_TYPE_NAME = {
    "embedding":   "Token Embedding",
    "tokenizer":   "Tokenizer",
    "query_input": "Decoder Queries",
    "memory":      "Memory Pool",
    "rmsnorm":     "RMSNorm",
    "layernorm":   "LayerNorm",
    "norm":        "Norm",
    "self_attn":   "Self-Attention",
    "sdp":         "SDP Attention",
    "linear":      "Linear Attention",
    "probsparse":  "ProbSparse Attention",
    "cosine":      "Cosine Attention",
    "local":       "Local Attention",
    "mha":         "Multi-Head Attn",
    "gated_attn":  "Gated Attention",
    "linear_attn": "Linear Attention",
    "deltanet":    "Gated DeltaNet",
    "cross_attn":  "Cross-Attention",
    "feedforward": "Feed-Forward (FFN)",
    "ffn":         "FFN",
    "swiglu":      "SwiGLU FFN",
    "geglu":       "GeGLU FFN",
    "moe":         "MoE FFN",
    "output":      "Linear Output",
    "linear":      "Linear",
    "add":         "Add",
}


def _color(t):
    return _TYPE_COLOR.get(t.lower(), COLORS["generic"])


def _name(t, override=None):
    return override or _TYPE_NAME.get(t.lower(), t)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, label, color, fontsize=8.5,
         radius=0.07, sublabel=None, ec=None):
    ec = ec or COLORS["block_border"]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor=ec, facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    ty = y + (0.06 if sublabel else 0)
    ax.text(x, ty, label, ha="center", va="center",
            fontsize=fontsize, color=COLORS["label"],
            fontfamily="monospace", zorder=4)
    if sublabel:
        ax.text(x, y - 0.12, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=COLORS["dim_label"],
                style="italic", zorder=4)


def _plus(ax, x, y, r=0.13):
    c = plt.Circle((x, y), r, color="white",
                   ec=COLORS["block_border"], lw=1.2, zorder=3)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center",
            fontsize=10, color=COLORS["label"], fontweight="bold", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.5):
    color = color or COLORS["arrow"]
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=10,
                        connectionstyle="arc3,rad=0"),
        zorder=2)


def _residual_skip(ax, x_right, rx, y_start, y_end, plus_r):
    """
    L-shaped skip connection:
      horizontal tap from box right edge → right column rx
      vertical from y_start up to y_end
      horizontal arrow into ⊕ from the right
    """
    c  = COLORS["residual"]
    lw = 1.4
    ax.plot([x_right + 0.03, rx], [y_start, y_start],
            color=c, lw=lw, zorder=2, solid_capstyle="round")
    ax.plot([rx, rx], [y_start, y_end],
            color=c, lw=lw, zorder=2, solid_capstyle="round")
    ax.annotate("",
        xy=(x_right - plus_r + 0.03, y_end),
        xytext=(rx, y_end),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                        mutation_scale=9,
                        connectionstyle="arc3,rad=0"),
        zorder=3)


def _bracket(ax, x, y_bot, y_top, color=None):
    c = color or COLORS["repeat_badge"]
    lw, tk = 1.8, 0.08
    ax.plot([x, x],      [y_bot, y_top], color=c, lw=lw, zorder=2)
    ax.plot([x, x + tk], [y_top, y_top], color=c, lw=lw, zorder=2)
    ax.plot([x, x + tk], [y_bot, y_bot], color=c, lw=lw, zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# Layer-stack builders  (pre-norm / post-norm)
# ─────────────────────────────────────────────────────────────────────────────

def _sub_block(attn_type, ffn_type, norm_type, norm_pos,
               attn_label=None, ffn_label=None,
               attn_sublabel=None, ffn_sublabel=None):
    """
    One (attention + FFN) sub-block.

    norm_pos "pre"  →  Norm → Attn → Add  +  Norm → FFN → Add
             "post" →  Attn → Add → Norm  +  FFN  → Add → Norm
    """
    nl = norm_type.upper()
    al = attn_label or _name(attn_type)
    fl = ffn_label  or _name(ffn_type)
    if norm_pos == "pre":
        return [
            {"type": norm_type,  "label": nl},
            {"type": attn_type,  "label": al, "sublabel": attn_sublabel},
            {"type": "add"},
            {"type": norm_type,  "label": nl},
            {"type": ffn_type,   "label": fl, "sublabel": ffn_sublabel},
            {"type": "add"},
        ]
    else:  # post-norm
        return [
            {"type": attn_type,  "label": al, "sublabel": attn_sublabel},
            {"type": "add"},
            {"type": norm_type,  "label": nl},
            {"type": ffn_type,   "label": fl, "sublabel": ffn_sublabel},
            {"type": "add"},
            {"type": norm_type,  "label": nl},
        ]


def make_encoder_layers(
        attn_type      = "mha",
        ffn_type       = "feedforward",
        norm_type      = "rmsnorm",
        norm_pos       = "pre",
        attn_label     = None,
        ffn_label      = None,
        include_embed  = True,
        include_output = True,
        embed_type     = "embedding",
        embed_label    = "Token Embedding",
        embed_sublabel = None,
        output_label   = "Linear Output",
        output_sublabel = None):
    """
    Build a full encoder layer stack.

    Parameters
    ----------
    attn_type  : "mha" | "gated_attn" | "linear_attn" | "deltanet" | …
    ffn_type   : "feedforward" | "moe" | "swiglu" | "geglu" | …
    norm_type  : "rmsnorm" | "layernorm"
    norm_pos   : "pre"  (Norm→Attn→Add) or "post" (Attn→Add→Norm)
    """
    layers = []
    if include_embed:
        layers.append(
            {"type": embed_type, "label": embed_label, "sublabel": embed_sublabel}
        )
    layers += _sub_block(attn_type, ffn_type, norm_type, norm_pos,
                         attn_label=attn_label, ffn_label=ffn_label)
    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})
    if include_output:
        layers.append(
            {"type": "output", "label": output_label, "sublabel": output_sublabel}
        )
    return layers


def make_decoder_layers(
        self_attn_type  = "mha",
        cross_attn      = True,
        ffn_type        = "feedforward",
        norm_type       = "rmsnorm",
        norm_pos        = "pre",
        self_attn_label = None,
        ffn_label       = None,
        include_embed   = True,
        include_output  = True,
        embed_type      = "embedding",
        embed_label     = "Token Embedding",
        embed_sublabel  = None,
        output_label    = "Linear Output",
        output_sublabel = None,
        cross_attn_type = "cross_attn",
        cross_attn_label = "Cross-Attention",
        cross_attn_sublabel = None):
    """
    Build a full decoder layer stack.

    Parameters
    ----------
    self_attn_type : attention type for masked self-attention
    cross_attn     : if True, insert a cross-attention sub-block
    ffn_type       : "feedforward" | "moe" | "swiglu" | "geglu" | …
    norm_type      : "rmsnorm" | "layernorm"
    norm_pos       : "pre" or "post"
    """
    nt  = norm_type
    nl  = nt.upper()
    sal = self_attn_label or (_name(self_attn_type) + " (masked)")
    fl  = ffn_label or _name(ffn_type)

    layers = []
    if include_embed:
        layers.append(
            {"type": embed_type, "label": embed_label, "sublabel": embed_sublabel}
        )

    # masked self-attention sub-block
    layers += _sub_block(self_attn_type, ffn_type, nt, norm_pos,
                         attn_label=sal, ffn_label=fl)

    if cross_attn:
        # strip the FFN half (last 3 entries), insert cross-attn + FFN
        half = 3  # norm+ffn+add (pre) or ffn+add+norm (post)
        layers = layers[:-half]
        if norm_pos == "pre":
            layers += [
                {"type": nt,           "label": nl},
                {
                    "type": cross_attn_type,
                    "label": cross_attn_label,
                    "sublabel": cross_attn_sublabel,
                },
                {"type": "add"},
                {"type": nt,           "label": nl},
                {"type": ffn_type,     "label": fl},
                {"type": "add"},
            ]
        else:
            layers += [
                {
                    "type": cross_attn_type,
                    "label": cross_attn_label,
                    "sublabel": cross_attn_sublabel,
                },
                {"type": "add"},
                {"type": nt,           "label": nl},
                {"type": ffn_type,     "label": fl},
                {"type": "add"},
                {"type": nt,           "label": nl},
            ]

    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})
    if include_output:
        layers.append(
            {"type": "output", "label": output_label, "sublabel": output_sublabel}
        )
    return layers


def make_hybrid_layers(
        linear_attn_type = "deltanet",
        full_attn_type   = "gated_attn",
        ffn_type         = "moe",
        norm_type        = "rmsnorm",
        norm_pos         = "pre",
        ratio            = 3,
        include_embed    = True,
        include_output   = True):
    """
    Hybrid linear/full-attention stack (e.g. Qwen3.5 3:1 ratio).
    `ratio` sub-blocks use linear_attn_type, then 1 uses full_attn_type.
    """
    layers = []
    if include_embed:
        layers.append({"type": "embedding", "label": "Token Embedding"})
    for _ in range(ratio):
        layers += _sub_block(linear_attn_type, ffn_type, norm_type, norm_pos,
                             attn_sublabel="linear attn")
    layers += _sub_block(full_attn_type, ffn_type, norm_type, norm_pos,
                         attn_sublabel="full attn")
    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})
    if include_output:
        layers.append({"type": "output", "label": "Linear Output"})
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# TransformerBlock  –  core drawing object
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock:
    """
    Draws one transformer block (encoder, decoder, or hybrid).

    Parameters
    ----------
    title        : str
    layers       : list[dict]  bottom → top, each with keys:
                     type (required), label, sublabel
    num_repeats  : int | None
    input_label  : str
    bg_color     : str
    show_input   : bool
    show_output  : bool
    """

    BOX_W  = 2.7
    BOX_H  = 0.42
    GAP    = 0.22
    V_PAD  = 0.50
    IO_H   = 0.55
    PLUS_R = 0.13
    RES_X  = 0.32   # how far right of box the skip column sits

    def __init__(self, title, layers,
                 num_repeats=None, input_label="Input",
                 bg_color=None, show_input=True, show_output=True):
        self.title       = title
        self.layers      = layers
        self.num_repeats = num_repeats
        self.input_label = input_label
        self.bg_color    = bg_color or COLORS["encoder_bg"]
        self.show_input  = show_input
        self.show_output = show_output
        # populated by draw()
        self._cx = self._y0 = self._y1 = None
        self._lys = []

    # ── geometry ──────────────────────────────────────────────────────
    def block_width(self):
        return self.BOX_W + 0.28 + self.RES_X + 0.30

    def block_height(self):
        n = len(self.layers)
        return n * self.BOX_H + (n - 1) * self.GAP + 2 * self.V_PAD

    def total_height(self):
        h = self.block_height()
        if self.show_input:  h += self.IO_H
        if self.show_output: h += self.IO_H
        return h

    # ── draw ──────────────────────────────────────────────────────────
    def draw(self, ax, cx, y_bottom):
        self._cx = cx
        step = self.BOX_H + self.GAP

        # ── y-positions ───────────────────────────────────────────────
        y = y_bottom
        if self.show_input:
            y_in_anchor = y
            y += self.IO_H

        bg_y0 = y
        y += self.V_PAD

        lys = []
        for _ in self.layers:
            lys.append(y + self.BOX_H / 2)
            y += step
        y -= self.GAP
        y += self.V_PAD
        bg_y1 = y
        self._y0, self._y1, self._lys = bg_y0, bg_y1, lys

        bw = self.block_width()
        rx = cx + self.BOX_W / 2 + self.RES_X

        # ── background ────────────────────────────────────────────────
        bg = FancyBboxPatch(
            (cx - bw / 2, bg_y0), bw, bg_y1 - bg_y0,
            boxstyle="round,pad=0,rounding_size=0.15",
            linewidth=1.5, edgecolor=COLORS["block_border"],
            facecolor=self.bg_color, alpha=0.9, zorder=1)
        ax.add_patch(bg)

        # ── title ─────────────────────────────────────────────────────
        ax.text(cx, bg_y1 + 0.20, self.title,
                ha="center", va="bottom", fontsize=12,
                color=COLORS["label"], fontweight="bold",
                fontfamily="sans-serif", zorder=5)

        # ── input ─────────────────────────────────────────────────────
        if self.show_input:
            _arrow(ax, cx, y_in_anchor + 0.06, cx, bg_y0 - 0.04)
            ax.text(cx, y_in_anchor + 0.05, self.input_label,
                    ha="center", va="top", fontsize=8.5,
                    color=COLORS["dim_label"], style="italic", zorder=4)

        # ── output ────────────────────────────────────────────────────
        if self.show_output:
            _arrow(ax, cx, bg_y1 + 0.04, cx, bg_y1 + self.IO_H - 0.07)

        # ── layer boxes + main-flow arrows ───────────────────────────
        residuals    = self._residual_groups()
        drawn_res    = set()

        for i, (spec, ly) in enumerate(zip(self.layers, lys)):
            lt     = spec.get("type", "generic").lower()
            label  = _name(lt, spec.get("label"))
            sub    = spec.get("sublabel")
            is_add = (lt == "add")

            if is_add:
                _plus(ax, cx, ly, r=self.PLUS_R)
            else:
                _box(ax, cx, ly, self.BOX_W, self.BOX_H,
                     label, color=_color(lt), sublabel=sub)

            # main-flow arrow from layer above (below in visual space)
            if i > 0:
                prev_lt = self.layers[i - 1].get("type", "").lower()
                y_from  = lys[i - 1] + (self.PLUS_R  if prev_lt == "add" else self.BOX_H / 2)
                y_to    = ly         - (self.PLUS_R  if is_add            else self.BOX_H / 2)
                _arrow(ax, cx, y_from, cx, y_to)

            # residual skip connections (drawn when we reach the ⊕ layer)
            for grp in residuals:
                si, ei = grp
                if i == ei and grp not in drawn_res:
                    drawn_res.add(grp)
                    self._draw_residual(ax, cx, lys, si, ei, rx)

        # ── N× badge + bracket ────────────────────────────────────────
        if self.num_repeats is not None:
            inner = [(i, s) for i, s in enumerate(self.layers)
                     if s.get("type", "").lower() not in
                     ("embedding", "output", "linear")]
            if inner:
                fi  = inner[0][0]
                li_ = inner[-1][0]
                yb  = lys[fi]  - self.BOX_H / 2 - 0.06
                yt  = lys[li_] + self.BOX_H / 2 + 0.06
                xb  = cx - bw / 2 - 0.12
                _bracket(ax, xb, yb, yt)
                ax.text(xb - 0.50, (yt + yb) / 2,
                        f"{self.num_repeats}×",
                        ha="left", va="center", fontsize=11,
                        color=COLORS["repeat_badge"], fontweight="bold", zorder=5)

        return bg_y1 + (self.IO_H if self.show_output else 0)

    # ── residual helpers ──────────────────────────────────────────────
    def _residual_groups(self):
        """
        For each ⊕ (add) layer, find the source of its skip connection:
        the nearest preceding 'add' or 'embedding' layer output.
        """
        groups = []
        for ei, spec in enumerate(self.layers):
            if spec.get("type", "").lower() != "add":
                continue
            si = ei - 1
            while si > 0:
                t = self.layers[si].get("type", "").lower()
                if t in ("add", "embedding"):
                    break
                si -= 1
            groups.append((si, ei))
        return groups

    def _draw_residual(self, ax, cx, lys, si, ei, rx):
        src_lt = self.layers[si].get("type", "").lower()
        is_add = (src_lt == "add")
        # tap from top of source layer
        y_tap  = lys[si] + (self.PLUS_R if is_add else self.BOX_H / 2)
        y_dst  = lys[ei]   # centre of ⊕

        _residual_skip(ax,
                       x_right = cx + self.BOX_W / 2,
                       rx      = rx,
                       y_start = y_tap,
                       y_end   = y_dst,
                       plus_r  = self.PLUS_R)

    # ── ports for cross-attention wiring ─────────────────────────────
    def cross_attn_port(self):
        for i, spec in enumerate(self.layers):
            if spec.get("type", "").lower() == "cross_attn":
                return (self._cx - self.BOX_W / 2, self._lys[i])
        return None

    def encoder_mem_port(self):
        bw = self.block_width()
        return (self._cx + bw / 2, (self._y0 + self._y1) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture extraction and selected-architecture builders
# ─────────────────────────────────────────────────────────────────────────────

_TOKENIZER_NAMES = {
    "direct": "direct",
    "patch_8": "patch-8",
    "patch_16": "patch-16",
    "patch_32": "patch-32",
    "multi_scale_patch": "multi-scale patch",
    "hierarchical": "hierarchical patch",
    "variate_tokens": "variate tokens",
}

_QUERY_MODE_NAMES = {
    "repeat_last": "repeat-last",
    "zeros": "zero queries",
    "learned_horizon_queries": "learned horizon",
    "shifted_target": "shifted target",
    "future_covariate_queries": "future covariates",
}

_STYLE_NAMES = {
    "autoregressive": "autoregressive",
    "informer": "informer",
}


def _best_choice(weights, default="unknown"):
    if not isinstance(weights, dict) or not weights:
        return default
    return str(max(weights.items(), key=lambda item: float(item[1]))[0])


def _compact_choice(value, mapping=None, default="unknown"):
    if value is None:
        return default
    key = str(value)
    if not key:
        return default
    return mapping.get(key, key.replace("_", " ")) if mapping else key.replace("_", " ")


def _top_level_transformer(module_obj):
    if module_obj is None:
        return None
    return (
        getattr(module_obj, "transformer", None)
        or getattr(module_obj, "rnn", None)
        or module_obj
    )


def extract_selected_transformer_spec(model):
    """Extract the currently selected transformer decisions from a search or fixed model."""
    weights = {}
    getter = getattr(model, "get_operation_weights", None)
    if callable(getter):
        try:
            weights = getter() or {}
        except Exception:
            weights = {}

    arch_mode = str(getattr(model, "arch_mode", "unknown"))
    enc_wrapper = getattr(model, "forecast_encoder", None)
    dec_wrapper = getattr(model, "forecast_decoder", None)
    enc = _top_level_transformer(enc_wrapper)
    dec = _top_level_transformer(dec_wrapper)

    encoder_spec = None
    if enc_wrapper is not None:
        encoder_spec = {
            "num_layers": int(getattr(enc, "num_layers", 1)) if enc is not None else None,
            "tokenizer": _best_choice(
                weights.get("encoder_tokenizer"),
                default=getattr(enc_wrapper, "patching_mode", getattr(enc, "patching_mode", "direct")),
            ),
            "self_attention": _best_choice(
                weights.get("encoder_self_attention"),
                default=getattr(enc_wrapper, "self_attention_type", getattr(enc, "self_attention_type", "unknown")),
            ),
            "self_position": _best_choice(
                weights.get("encoder_attention_position"),
                default=getattr(
                    enc_wrapper,
                    "self_attention_position_mode",
                    getattr(enc, "self_attention_position_mode", "unknown"),
                ),
            ),
            "ffn": _best_choice(
                weights.get("encoder_ffn"),
                default=getattr(enc_wrapper, "ffn_mode", getattr(enc, "ffn_variant", "unknown")),
            ),
        }

    decoder_spec = None
    if dec_wrapper is not None:
        decoder_spec = {
            "num_layers": int(getattr(dec, "num_layers", 1)) if dec is not None else None,
            "style": _best_choice(
                weights.get("decoder_style"),
                default=getattr(dec_wrapper, "decode_style", "autoregressive"),
            ),
            "query_mode": _best_choice(
                weights.get("decoder_query_generator"),
                default=getattr(model, "decoder_query_mode", "unknown"),
            ),
            "self_attention": _best_choice(
                weights.get("decoder_self_attention"),
                default=getattr(dec_wrapper, "self_attention_type", getattr(dec, "self_attention_type", "unknown")),
            ),
            "self_position": _best_choice(
                weights.get("decoder_attention_position"),
                default=getattr(
                    dec_wrapper,
                    "self_attention_position_mode",
                    getattr(dec, "self_attention_position_mode", "unknown"),
                ),
            ),
            "cross_attention": _best_choice(
                weights.get("decoder_cross_attention"),
                default=getattr(
                    dec_wrapper,
                    "cross_attention_type",
                    getattr(dec, "cross_attention_type", "none"),
                ),
            ),
            "cross_position": _best_choice(
                weights.get("decoder_cross_attention_position"),
                default=getattr(
                    dec_wrapper,
                    "cross_attention_position_mode",
                    getattr(dec, "cross_attention_position_mode", "unknown"),
                ),
            ),
            "ffn": _best_choice(
                weights.get("decoder_ffn"),
                default=getattr(dec_wrapper, "ffn_mode", getattr(dec, "ffn_variant", "unknown")),
            ),
            "memory_queries": _best_choice(
                weights.get("decoder_memory_queries"),
                default="unknown",
            ),
        }

    return {
        "arch_mode": arch_mode,
        "tie_encoder_decoder_arch": bool(getattr(model, "tie_encoder_decoder_arch", False)),
        "encoder": encoder_spec,
        "decoder": decoder_spec,
    }


def _summary_title(base, pairs):
    parts = [f"{key}={value}" for key, value in pairs if value and value != "unknown"]
    if not parts:
        return base
    return f"{base}\n" + " | ".join(parts)


def make_encoder_layers_from_spec(spec):
    tokenizer = _compact_choice(spec.get("tokenizer"), _TOKENIZER_NAMES)
    self_attn = _compact_choice(spec.get("self_attention"))
    position = _compact_choice(spec.get("self_position"))
    ffn = _compact_choice(spec.get("ffn"))
    layers = make_encoder_layers(
        attn_type=str(spec.get("self_attention", "self_attn")),
        ffn_type=str(spec.get("ffn", "ffn")),
        norm_type="rmsnorm",
        norm_pos="pre",
        attn_label=_name(str(spec.get("self_attention", "self_attn"))),
        ffn_label=_name(str(spec.get("ffn", "ffn"))),
        include_embed=True,
        include_output=True,
        embed_type="tokenizer",
        embed_label="Input Tokenizer",
        embed_sublabel=tokenizer,
        output_label="Encoder States",
        output_sublabel=None,
    )
    title = _summary_title(
        "Encoder",
        [
            ("tok", tokenizer),
            ("attn", self_attn),
            ("pos", position),
            ("ffn", ffn),
        ],
    )
    return layers, title, int(spec.get("num_layers") or 1)


def make_decoder_layers_from_spec(spec, *, encoder_available):
    style = _compact_choice(spec.get("style"), _STYLE_NAMES)
    query_mode = _compact_choice(spec.get("query_mode"), _QUERY_MODE_NAMES)
    self_attn = _compact_choice(spec.get("self_attention"))
    self_pos = _compact_choice(spec.get("self_position"))
    cross_attn = _compact_choice(spec.get("cross_attention"))
    cross_pos = _compact_choice(spec.get("cross_position"))
    ffn = _compact_choice(spec.get("ffn"))
    memory_queries = spec.get("memory_queries")
    memory_display = (
        f"{memory_queries} queries"
        if memory_queries not in (None, "", "unknown")
        else "unknown"
    )
    include_cross = bool(encoder_available and str(spec.get("cross_attention", "none")).lower() != "none")
    embed_label = "Parallel Decoder Input" if str(spec.get("style", "")).lower() == "informer" else "Decoder Queries"
    layers = make_decoder_layers(
        self_attn_type=str(spec.get("self_attention", "self_attn")),
        cross_attn=include_cross,
        ffn_type=str(spec.get("ffn", "ffn")),
        norm_type="rmsnorm",
        norm_pos="pre",
        self_attn_label=_name(str(spec.get("self_attention", "self_attn"))) + " (masked)",
        ffn_label=_name(str(spec.get("ffn", "ffn"))),
        include_embed=True,
        include_output=True,
        embed_type="query_input",
        embed_label=embed_label,
        embed_sublabel=query_mode,
        output_label="Forecast Head",
        output_sublabel=style,
        cross_attn_type="cross_attn",
        cross_attn_label="Cross-Attention",
        cross_attn_sublabel=(
            f"{cross_attn} | {cross_pos}" if include_cross else None
        ),
    )
    title = _summary_title(
        "Decoder",
        [
            ("style", style),
            ("query", query_mode),
            ("self", self_attn),
            ("pos", self_pos),
            ("ffn", ffn),
            ("mem", memory_display if include_cross else None),
            ("cross", cross_attn if include_cross else "none"),
        ],
    )
    cross_label = None
    if include_cross:
        cross_label = "encoder memory\n(K, V)"
        if memory_queries not in (None, "", "unknown"):
            cross_label = f"encoder memory\n(K, V), {memory_queries} queries"
    return layers, title, int(spec.get("num_layers") or 1), cross_label


def draw_selected_transformer_architecture(model, title=None, figsize=None):
    """Render the selected DARTS transformer architecture from the current model state."""
    spec = extract_selected_transformer_spec(model)
    arch_mode = spec["arch_mode"]
    encoder_spec = spec.get("encoder")
    decoder_spec = spec.get("decoder")
    overall_title = title or f"Selected Transformer Architecture ({arch_mode})"

    if arch_mode == "encoder_only" and encoder_spec is not None:
        encoder_layers, encoder_title, encoder_repeats = make_encoder_layers_from_spec(
            encoder_spec
        )
        return draw_single_block(
            encoder_layers,
            title=f"{overall_title}\n{encoder_title}",
            num_repeats=encoder_repeats,
            input_label="History / endogenous series",
            bg_color=COLORS["encoder_bg"],
            figsize=figsize,
        )

    if arch_mode == "decoder_only" and decoder_spec is not None:
        decoder_layers, decoder_title, decoder_repeats, _ = make_decoder_layers_from_spec(
            decoder_spec, encoder_available=False
        )
        return draw_single_block(
            decoder_layers,
            title=f"{overall_title}\n{decoder_title}",
            num_repeats=decoder_repeats,
            input_label="History-conditioned decoder input",
            bg_color=COLORS["decoder_bg"],
            figsize=figsize,
        )

    if arch_mode == "encoder_decoder" and encoder_spec is not None and decoder_spec is not None:
        encoder_layers, encoder_title, encoder_repeats = make_encoder_layers_from_spec(
            encoder_spec
        )
        decoder_layers, decoder_title, decoder_repeats, cross_label = make_decoder_layers_from_spec(
            decoder_spec, encoder_available=True
        )
        return draw_encoder_decoder(
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_title=encoder_title,
            decoder_title=decoder_title,
            encoder_repeats=encoder_repeats,
            decoder_repeats=decoder_repeats,
            encoder_input="History / endogenous series",
            decoder_input="Decoder input sequence",
            title=overall_title,
            figsize=figsize,
            cross_label=cross_label,
        )

    fig, ax = plt.subplots(figsize=figsize or (8, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Unsupported or non-transformer architecture: {arch_mode}",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["label"],
        transform=ax.transAxes,
    )
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# High-level figure builders
# ─────────────────────────────────────────────────────────────────────────────

def draw_single_block(layers,
                      title="Transformer Block",
                      num_repeats=None,
                      input_label="Input",
                      bg_color=None,
                      figsize=None):
    """Draw one encoder or decoder block. Returns (fig, ax)."""
    block = TransformerBlock(title, layers,
                             num_repeats=num_repeats,
                             input_label=input_label,
                             bg_color=bg_color)
    bw = block.block_width() + 1.4
    bh = block.total_height() + 1.6
    figsize = figsize or (max(5.5, bw), max(7.0, bh))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, bw)
    ax.set_ylim(-0.3, bh)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    block.draw(ax, bw / 2, 0.45)
    plt.tight_layout(pad=0.4)
    return fig, ax


def draw_encoder_decoder(encoder_layers,
                          decoder_layers,
                          encoder_title   = "Encoder",
                          decoder_title   = "Decoder",
                          encoder_repeats = 6,
                          decoder_repeats = 6,
                          encoder_input   = "Source tokens",
                          decoder_input   = "Target tokens (shifted right)",
                          title           = "Transformer Architecture",
                          figsize         = None,
                          cross_label     = "encoder memory\n(K, V)"):
    """
    Draw encoder + decoder side by side with cross-attention wiring.
    Returns (fig, ax).
    """
    enc = TransformerBlock(encoder_title, encoder_layers,
                           num_repeats=encoder_repeats,
                           input_label=encoder_input,
                           bg_color=COLORS["encoder_bg"])
    dec = TransformerBlock(decoder_title, decoder_layers,
                           num_repeats=decoder_repeats,
                           input_label=decoder_input,
                           bg_color=COLORS["decoder_bg"])

    gap = 2.0
    bw  = max(enc.block_width(), dec.block_width())
    tw  = 2 * bw + gap + 2.4
    th  = max(enc.total_height(), dec.total_height()) + 2.0

    figsize = figsize or (tw * 1.05, th * 1.05)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, tw)
    ax.set_ylim(-0.3, th)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx_enc = tw / 2 - bw / 2 - gap / 2
    cx_dec = tw / 2 + bw / 2 + gap / 2
    enc.draw(ax, cx_enc, 0.45)
    dec.draw(ax, cx_dec, 0.45)

    # ── cross-attention arrow ─────────────────────────────────────────
    ca_port = dec.cross_attn_port()
    if ca_port:
        mem_x = cx_enc + enc.block_width() / 2
        mem_y = (enc._y0 + enc._y1) / 2
        ax.annotate("",
            xy=(ca_port[0] - 0.02, ca_port[1]),
            xytext=(mem_x, mem_y),
            arrowprops=dict(
                arrowstyle="-|>", color=COLORS["cross_arrow"],
                lw=1.8, mutation_scale=11,
                connectionstyle="arc3,rad=-0.20"),
            zorder=3)
        mid_x = (mem_x + ca_port[0]) / 2
        mid_y = (mem_y  + ca_port[1]) / 2 - 0.4
        ax.text(mid_x, mid_y, cross_label,
                ha="center", va="top", fontsize=7.5,
                color=COLORS["cross_arrow"], style="italic",
                bbox=dict(fc="white", ec="none", pad=1.5))

    ax.text(tw / 2, th - 0.08, title,
            ha="center", va="top", fontsize=15, fontweight="bold",
            color=COLORS["label"], fontfamily="sans-serif")

    plt.tight_layout(pad=0.4)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    OUT = "/mnt/user-data/outputs"
    os.makedirs(OUT, exist_ok=True)

    # 1. Classic Transformer (post-norm, Vaswani 2017)
    fig, _ = draw_encoder_decoder(
        encoder_layers  = make_encoder_layers(
            attn_type="mha", ffn_type="feedforward",
            norm_type="layernorm", norm_pos="post"),
        decoder_layers  = make_decoder_layers(
            self_attn_type="mha", cross_attn=True,
            ffn_type="feedforward",
            norm_type="layernorm", norm_pos="post"),
        encoder_title="Encoder", decoder_title="Decoder",
        encoder_repeats=6, decoder_repeats=6,
        encoder_input="Source tokens",
        decoder_input="Target tokens (shifted)",
        title="Classic Transformer · post-norm  (Vaswani et al., 2017)",
        figsize=(15, 12),
    )
    fig.savefig(f"{OUT}/1_transformer_classic.png", dpi=150, bbox_inches="tight")
    print("Saved: 1_transformer_classic.png")

    # 2. GPT / LLaMA decoder (pre-norm, SwiGLU)
    fig, _ = draw_single_block(
        layers      = make_decoder_layers(
            self_attn_type="mha", cross_attn=False,
            ffn_type="swiglu", norm_type="rmsnorm", norm_pos="pre"),
        title       = "GPT / LLaMA-style · pre-norm · SwiGLU",
        num_repeats = 32,
        input_label = "Tokenized text",
        bg_color    = COLORS["decoder_bg"],
    )
    fig.savefig(f"{OUT}/2_gpt_prenorm_swiglu.png", dpi=150, bbox_inches="tight")
    print("Saved: 2_gpt_prenorm_swiglu.png")

    # 3. BERT encoder (post-norm)
    fig, _ = draw_single_block(
        layers      = make_encoder_layers(
            attn_type="mha", ffn_type="feedforward",
            norm_type="layernorm", norm_pos="post"),
        title       = "BERT Encoder · post-norm",
        num_repeats = 12,
        input_label = "[CLS] token sequence",
        bg_color    = COLORS["encoder_bg"],
    )
    fig.savefig(f"{OUT}/3_bert_postnorm.png", dpi=150, bbox_inches="tight")
    print("Saved: 3_bert_postnorm.png")

    # 4. MoE encoder (pre-norm)
    fig, _ = draw_single_block(
        layers      = make_encoder_layers(
            attn_type="mha", ffn_type="moe",
            norm_type="rmsnorm", norm_pos="pre"),
        title       = "MoE Encoder · pre-norm",
        num_repeats = 24,
        input_label = "Tokenized text",
        bg_color    = COLORS["encoder_bg"],
    )
    fig.savefig(f"{OUT}/4_moe_encoder.png", dpi=150, bbox_inches="tight")
    print("Saved: 4_moe_encoder.png")

    # 5. Hybrid DeltaNet + MoE (Qwen3.5 style)
    fig, _ = draw_single_block(
        layers      = make_hybrid_layers(
            linear_attn_type="deltanet", full_attn_type="gated_attn",
            ffn_type="moe", norm_type="rmsnorm", norm_pos="pre", ratio=3),
        title       = "Hybrid DeltaNet · MoE  (3 : 1 ratio)",
        num_repeats = 24,
        input_label = "Tokenized text",
        bg_color    = "#EDF4F0",
    )
    fig.savefig(f"{OUT}/5_deltanet_moe_hybrid.png", dpi=150, bbox_inches="tight")
    print("Saved: 5_deltanet_moe_hybrid.png")

    # 6. Pre-norm vs Post-norm comparison
    fig, _ = draw_encoder_decoder(
        encoder_layers  = make_encoder_layers(
            attn_type="mha", ffn_type="feedforward",
            norm_type="rmsnorm", norm_pos="pre",
            include_embed=False, include_output=False),
        decoder_layers  = make_encoder_layers(
            attn_type="mha", ffn_type="feedforward",
            norm_type="layernorm", norm_pos="post",
            include_embed=False, include_output=False),
        encoder_title   = "Pre-Norm",
        decoder_title   = "Post-Norm",
        encoder_repeats = None,
        decoder_repeats = None,
        encoder_input   = "x (residual stream)",
        decoder_input   = "x (residual stream)",
        title           = "Pre-Norm  vs  Post-Norm",
        figsize         = (13, 8),
    )
    fig.savefig(f"{OUT}/6_prenorm_vs_postnorm.png", dpi=150, bbox_inches="tight")
    print("Saved: 6_prenorm_vs_postnorm.png")

    plt.close("all")
    print(f"\nAll diagrams saved to {OUT}/")
