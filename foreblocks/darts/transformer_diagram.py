"""
transformer_diagram.py
======================

Raschka-style transformer architecture diagrams using matplotlib.

Main stylistic goals
--------------------
- Flat textbook-style design
- Nested rounded containers (model -> repeated stack -> inner modules)
- Thin borders, straight arrows, generous whitespace
- Curly brace for repetition (e.g. 32×)
- Minimal but readable labels
- Optional encoder-decoder cross-attention wiring
- Optional FFN zoom panel

Quick start
-----------
from transformer_diagram import (
    make_encoder_layers,
    make_decoder_layers,
    draw_single_block,
    draw_encoder_decoder,
)

layers = make_decoder_layers(
    self_attn_type="mha",
    cross_attn=False,
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_pos="pre",
)

fig, ax = draw_single_block(
    layers,
    title="GPT-style Decoder",
    num_repeats=32,
    input_label="Tokenized text",
)
fig.savefig("gpt_style.png", dpi=220, bbox_inches="tight")
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path


# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# -----------------------------------------------------------------------------
# Palette: flat, pastel, textbook-like
# -----------------------------------------------------------------------------

COLORS = {
    "page_bg": "#F3F3F3",
    "model_bg": "#D9D9D9",
    "stack_bg": "#D79ABC",
    "inner_group_bg": "#D79ABC",
    "zoom_bg": "#E7E7E7",
    "embedding": "#F7F7F7",
    "tokenizer": "#F7F7F7",
    "query_input": "#F7F7F7",
    "memory": "#F7F7F7",
    "norm": "#F7F7F7",
    "self_attn": "#6B6B6B",
    "cross_attn": "#F7F7F7",
    "linear_attn": "#D9C8F2",
    "feedforward": "#F7F7F7",
    "moe": "#F7F7F7",
    "swiglu": "#F7F7F7",
    "geglu": "#F7F7F7",
    "output": "#F7F7F7",
    "add": "#FFFFFF",
    "generic": "#F7F7F7",
    "block_border": "#111111",
    "text": "#111111",
    "dim_text": "#444444",
    "accent": "#D11B7E",
    "arrow": "#111111",
    "residual": "#111111",
    "cross_arrow": "#111111",
}

_TYPE_COLOR = {
    "embedding": COLORS["embedding"],
    "tokenizer": COLORS["tokenizer"],
    "query_input": COLORS["query_input"],
    "memory": COLORS["memory"],
    "rmsnorm": COLORS["norm"],
    "layernorm": COLORS["norm"],
    "norm": COLORS["norm"],
    "self_attn": COLORS["self_attn"],
    "sdp": COLORS["self_attn"],
    "mha": COLORS["self_attn"],
    "gated_attn": COLORS["self_attn"],
    "probsparse": COLORS["self_attn"],
    "cosine": COLORS["self_attn"],
    "local": COLORS["self_attn"],
    "linear_attn": COLORS["linear_attn"],
    "deltanet": COLORS["linear_attn"],
    "linear": COLORS["generic"],
    "cross_attn": COLORS["cross_attn"],
    "feedforward": COLORS["feedforward"],
    "ffn": COLORS["feedforward"],
    "swiglu": COLORS["swiglu"],
    "geglu": COLORS["geglu"],
    "moe": COLORS["moe"],
    "output": COLORS["output"],
    "add": COLORS["add"],
}

_TYPE_NAME = {
    "embedding": "Token embedding layer",
    "tokenizer": "Tokenizer",
    "query_input": "Decoder queries",
    "memory": "Memory pool",
    "rmsnorm": "RMSNorm",
    "layernorm": "LayerNorm",
    "norm": "Norm",
    "self_attn": "Self-attention",
    "sdp": "SDP attention",
    "mha": "Masked multi-head\nattention",
    "gated_attn": "Gated attention",
    "linear_attn": "Linear attention",
    "deltanet": "DeltaNet",
    "probsparse": "ProbSparse attention",
    "cosine": "Cosine attention",
    "local": "Local attention",
    "cross_attn": "Cross-attention",
    "feedforward": "Feed forward",
    "ffn": "Feed forward",
    "swiglu": "Feed forward",
    "geglu": "Feed forward",
    "moe": "MoE feed forward",
    "output": "Linear output layer",
    "linear": "Linear layer",
    "add": "+",
}


def _color(t: str) -> str:
    return _TYPE_COLOR.get(t.lower(), COLORS["generic"])


def _name(t: str, override: str | None = None) -> str:
    return override or _TYPE_NAME.get(t.lower(), t)


# -----------------------------------------------------------------------------
# Primitive helpers
# -----------------------------------------------------------------------------


def _container(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    fc: str = "#FFFFFF",
    ec: str | None = None,
    lw: float = 1.4,
    dashed: bool = False,
    label_size: float = 12,
    label_weight: str = "bold",
    rounding: float = 0.26,
    zorder: int = 1,
):
    ec = ec or COLORS["block_border"]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=(0, (2.0, 1.5)) if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(patch)

    if label:
        ax.text(
            x,
            y + h / 2 + 0.20,
            label,
            ha="center",
            va="bottom",
            fontsize=label_size,
            fontweight=label_weight,
            color=COLORS["text"],
            zorder=zorder + 1,
        )
    return patch


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    *,
    fontsize: float = 9.0,
    radius: float = 0.10,
    sublabel: str | None = None,
    ec: str | None = None,
    text_color: str | None = None,
    fontweight: str = "normal",
    zorder: int = 3,
):
    ec = ec or COLORS["block_border"]
    text_color = text_color or COLORS["text"]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=color,
        zorder=zorder,
    )
    ax.add_patch(patch)

    ty = y + (0.06 if sublabel else 0.0)
    ax.text(
        x,
        ty,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight=fontweight,
        zorder=zorder + 1,
    )
    if sublabel:
        ax.text(
            x,
            y - 0.14,
            sublabel,
            ha="center",
            va="center",
            fontsize=max(7.3, fontsize - 1.5),
            color=COLORS["dim_text"],
            style="italic",
            zorder=zorder + 1,
        )


def _plus(ax, x: float, y: float, r: float = 0.14):
    c = Circle(
        (x, y),
        r,
        facecolor="white",
        edgecolor=COLORS["block_border"],
        linewidth=1.1,
        zorder=5,
    )
    ax.add_patch(c)
    ax.text(
        x,
        y,
        "+",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["text"],
        fontweight="normal",
        zorder=5,
    )


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    lw: float = 1.4,
    zorder: int = 2,
):
    color = color or COLORS["arrow"]
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=12,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def _residual_skip(
    ax,
    x_right: float,
    rx: float,
    y_start: float,
    y_end: float,
    plus_r: float,
):
    c = COLORS["residual"]
    lw = 1.4
    ax.plot([x_right, rx], [y_start, y_start], color=c, lw=lw, zorder=2)
    ax.plot([rx, rx], [y_start, y_end], color=c, lw=lw, zorder=2)
    ax.annotate(
        "",
        xy=(x_right - plus_r, y_end),
        xytext=(rx, y_end),
        arrowprops=dict(
            arrowstyle="-|>",
            color=c,
            lw=lw,
            mutation_scale=12,
            connectionstyle="arc3,rad=0",
        ),
        zorder=3,
    )


def _curly_brace(
    ax,
    x: float,
    y0: float,
    y1: float,
    *,
    width: float = 0.16,
    lw: float = 1.6,
    color: str | None = None,
    facing: str = "right",
    zorder: int = 3,
):
    """
    Vertical curly brace between y0 and y1.

    facing='right' -> looks like {
    facing='left'  -> looks like }

    Continuous symmetric brace with inward-curved top and bottom tips.
    """
    color = color or COLORS["text"]
    if y1 < y0:
        y0, y1 = y1, y0

    dy = y1 - y0
    mid = 0.5 * (y0 + y1)
    s = 1.0 if facing == "right" else -1.0
    w = width * s

    verts = [
        # start at top inner tip
        (x + 0.6 * w, y0 + 0.1),
        # top inward hook -> shoulder
        (x + 0.02 * w, y0 + 0.015 * dy + 0.2),
        (x + 0.00 * w, y0 + 0.060 * dy + 0.2),
        (x + 0.00 * w, y0 + 0.110 * dy),
        # shoulder -> outer bulge
        (x + 0.00 * w, y0 + 0.170 * dy),
        (x + 0.30 * w, y0 + 0.220 * dy),
        (x + 0.82 * w, y0 + 0.320 * dy),
        # outer bulge -> center pinch
        (x + 1.00 * w, y0 + 0.390 * dy),
        (x + 0.35 * w, y0 + 0.470 * dy),
        (x + 0.00 * w, mid),
        # center pinch -> lower outer bulge
        (x + 0.35 * w, y0 + 0.530 * dy),
        (x + 1.00 * w, y0 + 0.610 * dy),
        (x + 0.82 * w, y0 + 0.680 * dy),
        # lower bulge -> bottom shoulder
        (x + 0.30 * w, y0 + 0.780 * dy),
        (x + 0.00 * w, y0 + 0.830 * dy),
        (x + 0.00 * w, y0 + 0.890 * dy),
        # bottom shoulder -> bottom inner tip
        (x + 0.00 * w, y0 + 0.940 * dy - 0.2),
        (x + 0.02 * w, y0 + 0.985 * dy - 0.2),
        (x + 0.6 * w, y1 - 0.15),
    ]

    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)

    patch = PathPatch(
        Path(verts, codes),
        facecolor="none",
        edgecolor=color,
        lw=lw,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _accent_text(
    ax,
    x: float,
    y: float,
    prefix: str,
    accent: str,
    suffix: str = "",
    *,
    fontsize: float = 14,
    ha: str = "left",
    va: str = "center",
    weight: str = "bold",
    accent_color: str | None = None,
    base_color: str | None = None,
    zorder: int = 5,
):
    accent_color = accent_color or COLORS["accent"]
    base_color = base_color or COLORS["text"]

    if ha != "left":
        full = prefix + accent + suffix
        ax.text(
            x,
            y,
            full,
            fontsize=fontsize,
            ha=ha,
            va=va,
            color=base_color,
            fontweight=weight,
            zorder=zorder,
        )
        return

    ax.text(
        x,
        y,
        prefix,
        fontsize=fontsize,
        ha="left",
        va=va,
        color=base_color,
        fontweight=weight,
        zorder=zorder,
    )
    dx1 = 0.115 * fontsize * len(prefix) / 10.0
    ax.text(
        x + dx1,
        y,
        accent,
        fontsize=fontsize,
        ha="left",
        va=va,
        color=accent_color,
        fontweight=weight,
        zorder=zorder,
    )
    dx2 = 0.115 * fontsize * len(accent) / 10.0
    if suffix:
        ax.text(
            x + dx1 + dx2,
            y,
            suffix,
            fontsize=fontsize,
            ha="left",
            va=va,
            color=base_color,
            fontweight=weight,
            zorder=zorder,
        )


# -----------------------------------------------------------------------------
# Layer stack builders
# -----------------------------------------------------------------------------


def _sub_block(
    attn_type: str,
    ffn_type: str,
    norm_type: str,
    norm_pos: str,
    *,
    attn_label: str | None = None,
    ffn_label: str | None = None,
    attn_sublabel: str | None = None,
    ffn_sublabel: str | None = None,
) -> list[dict[str, Any]]:
    nl = norm_type.upper()
    al = attn_label or _name(attn_type)
    fl = ffn_label or _name(ffn_type)

    if norm_pos == "pre":
        return [
            {"type": norm_type, "label": nl},
            {"type": attn_type, "label": al, "sublabel": attn_sublabel},
            {"type": "add"},
            {"type": norm_type, "label": nl},
            {"type": ffn_type, "label": fl, "sublabel": ffn_sublabel},
            {"type": "add"},
        ]

    return [
        {"type": attn_type, "label": al, "sublabel": attn_sublabel},
        {"type": "add"},
        {"type": norm_type, "label": nl},
        {"type": ffn_type, "label": fl, "sublabel": ffn_sublabel},
        {"type": "add"},
        {"type": norm_type, "label": nl},
    ]


def make_encoder_layers(
    attn_type: str = "mha",
    ffn_type: str = "feedforward",
    norm_type: str = "rmsnorm",
    norm_pos: str = "pre",
    attn_label: str | None = None,
    ffn_label: str | None = None,
    include_embed: bool = True,
    include_output: bool = True,
    embed_type: str = "embedding",
    embed_label: str = "Token embedding layer",
    embed_sublabel: str | None = None,
    output_label: str = "Linear output layer",
    output_sublabel: str | None = None,
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    if include_embed:
        layers.append(
            {
                "type": embed_type,
                "label": embed_label,
                "sublabel": embed_sublabel,
            }
        )

    layers += _sub_block(
        attn_type,
        ffn_type,
        norm_type,
        norm_pos,
        attn_label=attn_label,
        ffn_label=ffn_label,
    )

    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})

    if include_output:
        layers.append(
            {
                "type": "output",
                "label": output_label,
                "sublabel": output_sublabel,
            }
        )
    return layers


def make_decoder_layers(
    self_attn_type: str = "mha",
    cross_attn: bool = True,
    ffn_type: str = "feedforward",
    norm_type: str = "rmsnorm",
    norm_pos: str = "pre",
    self_attn_label: str | None = None,
    ffn_label: str | None = None,
    include_embed: bool = True,
    include_output: bool = True,
    embed_type: str = "embedding",
    embed_label: str = "Token embedding layer",
    embed_sublabel: str | None = None,
    output_label: str = "Linear output layer",
    output_sublabel: str | None = None,
    cross_attn_type: str = "cross_attn",
    cross_attn_label: str = "Cross-attention",
    cross_attn_sublabel: str | None = None,
) -> list[dict[str, Any]]:
    nt = norm_type
    nl = nt.upper()
    sal = self_attn_label or (_name(self_attn_type) + " (masked)")
    fl = ffn_label or _name(ffn_type)

    layers: list[dict[str, Any]] = []
    if include_embed:
        layers.append(
            {
                "type": embed_type,
                "label": embed_label,
                "sublabel": embed_sublabel,
            }
        )

    layers += _sub_block(
        self_attn_type,
        ffn_type,
        nt,
        norm_pos,
        attn_label=sal,
        ffn_label=fl,
    )

    if cross_attn:
        half = 3
        layers = layers[:-half]
        if norm_pos == "pre":
            layers += [
                {"type": nt, "label": nl},
                {
                    "type": cross_attn_type,
                    "label": cross_attn_label,
                    "sublabel": cross_attn_sublabel,
                },
                {"type": "add"},
                {"type": nt, "label": nl},
                {"type": ffn_type, "label": fl},
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
                {"type": nt, "label": nl},
                {"type": ffn_type, "label": fl},
                {"type": "add"},
                {"type": nt, "label": nl},
            ]

    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})

    if include_output:
        layers.append(
            {
                "type": "output",
                "label": output_label,
                "sublabel": output_sublabel,
            }
        )
    return layers


def make_hybrid_layers(
    linear_attn_type: str = "deltanet",
    full_attn_type: str = "gated_attn",
    ffn_type: str = "moe",
    norm_type: str = "rmsnorm",
    norm_pos: str = "pre",
    ratio: int = 3,
    include_embed: bool = True,
    include_output: bool = True,
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    if include_embed:
        layers.append({"type": "embedding", "label": "Token embedding layer"})
    for _ in range(ratio):
        layers += _sub_block(
            linear_attn_type,
            ffn_type,
            norm_type,
            norm_pos,
            attn_sublabel="linear attention",
        )
    layers += _sub_block(
        full_attn_type,
        ffn_type,
        norm_type,
        norm_pos,
        attn_sublabel="full attention",
    )
    if norm_pos == "pre" and include_output:
        layers.append({"type": norm_type, "label": f"Final {norm_type.upper()}"})
    if include_output:
        layers.append({"type": "output", "label": "Linear output layer"})
    return layers


# -----------------------------------------------------------------------------
# Core block class
# -----------------------------------------------------------------------------


class TransformerBlock:
    """
    Hierarchical block renderer:
      outer model container
        -> repeated stack container
          -> inner repeated layer container
            -> module boxes

    Important style rule:
      The colored repeated stack contains only the repeatable transformer body.
      Token embedding / decoder query input at the bottom and
      final norm / output head at the top stay outside the colored stack.
    """

    BOX_W = 2.35
    BOX_H = 0.46
    GAP = 0.30
    V_PAD = 0.52
    IO_H = 0.72
    PLUS_R = 0.13
    RES_X = 0.42

    OUTER_W_PAD = 0.82
    OUTER_TOP_PAD = 0.70
    OUTER_BOTTOM_PAD = 0.60

    STACK_W_PAD = 0.42
    STACK_TOP_PAD = 0.34
    STACK_BOTTOM_PAD = 0.30

    INNER_W_PAD = 0.26
    INNER_TOP_PAD = 0.26
    INNER_BOTTOM_PAD = 0.22

    OUTSIDE_STACK_TYPES = {"embedding", "tokenizer", "query_input", "memory", "output"}

    def __init__(
        self,
        title: str,
        layers: Sequence[dict[str, Any]],
        *,
        num_repeats: int | None = None,
        input_label: str = "Input",
        bg_color: str | None = None,
        show_input: bool = True,
        show_output: bool = True,
        stack_label: str | None = None,
        show_inner_group: bool = True,
    ):
        self.title = title
        self.layers = list(layers)
        self.num_repeats = num_repeats
        self.input_label = input_label
        self.bg_color = bg_color or COLORS["model_bg"]
        self.show_input = show_input
        self.show_output = show_output
        self.stack_label = stack_label
        self.show_inner_group = show_inner_group

        self._cx = None
        self._y0 = None
        self._y1 = None
        self._lys: list[float] = []
        self._stack_bounds = None
        self._inner_bounds = None

    def block_width(self) -> float:
        return self.BOX_W + self.RES_X + self.OUTER_W_PAD + 0.65

    def modules_height(self) -> float:
        n = len(self.layers)
        return n * self.BOX_H + max(0, n - 1) * self.GAP

    def outer_height(self) -> float:
        return (
            self.modules_height()
            + self.OUTER_TOP_PAD
            + self.OUTER_BOTTOM_PAD
            + (self.IO_H if self.show_input else 0)
            + (self.IO_H if self.show_output else 0)
            + 0.20
        )

    def total_height(self) -> float:
        return self.outer_height()

    def _is_final_norm(self, idx: int) -> bool:
        spec = self.layers[idx]
        t = spec.get("type", "").lower()
        label = str(spec.get("label", ""))
        return t in {"rmsnorm", "layernorm", "norm"} and label.startswith("Final ")

    def _is_outside_stack(self, idx: int) -> bool:
        spec = self.layers[idx]
        t = spec.get("type", "").lower()
        if t in self.OUTSIDE_STACK_TYPES:
            return True
        if self._is_final_norm(idx):
            return True
        return False

    def _repeat_core_indices(self) -> list[int]:
        return [i for i in range(len(self.layers)) if not self._is_outside_stack(i)]

    def draw(self, ax, cx: float, y_bottom: float) -> float:
        self._cx = cx
        step = self.BOX_H + self.GAP

        y = y_bottom
        if self.show_input:
            y_input_anchor = y
            y += self.IO_H
        else:
            y_input_anchor = None

        outer_y0 = y
        y += self.OUTER_BOTTOM_PAD

        lys: list[float] = []
        for _ in self.layers:
            lys.append(y + self.BOX_H / 2)
            y += step
        y -= self.GAP

        y += self.OUTER_TOP_PAD
        outer_y1 = y

        self._y0, self._y1, self._lys = outer_y0, outer_y1, lys

        bw = self.block_width()
        rx = cx + self.BOX_W / 2 + self.RES_X
        outer_h = outer_y1 - outer_y0

        core_indices = self._repeat_core_indices()

        stack_bounds = None
        inner_bounds = None
        if core_indices:
            fi = core_indices[0]
            li = core_indices[-1]

            # # # tighter colored region: only the repeated stack
            stack_y0 = lys[fi] - self.BOX_H / 2 - self.STACK_BOTTOM_PAD
            stack_y1 = lys[li] + self.BOX_H / 2 + self.STACK_TOP_PAD
            stack_cy = 0.5 * (stack_y0 + stack_y1)
            stack_w = self.BOX_W + self.RES_X + self.STACK_W_PAD
            stack_h = stack_y1 - stack_y0
            stack_bounds = (cx, stack_cy, stack_w, stack_h)

            # inner group should sit INSIDE the colored stack, not create a second visible band
            inner_y0 = stack_y0 + 0.2
            inner_y1 = stack_y1 - 0.2
            inner_cy = 0.5 * (inner_y0 + inner_y1)
            inner_h = max(0.1, inner_y1 - inner_y0)
            inner_w = self.BOX_W + self.RES_X - 0.10
            inner_bounds = (cx, inner_cy, inner_w, inner_h)

        self._stack_bounds = stack_bounds
        self._inner_bounds = inner_bounds

        _container(
            ax,
            cx,
            outer_y0 + outer_h / 2,
            bw,
            outer_h,
            fc=self.bg_color,
            ec=COLORS["block_border"],
            lw=1.4,
            zorder=1,
        )

        # if stack_bounds is not None:
        # sx, sy, sw, sh = stack_bounds
        # _container(
        #     ax,
        #     sx,
        #     sy,
        #     sw,
        #     sh,
        #     fc=COLORS["stack_bg"],
        #     ec=COLORS["block_border"],
        #     lw=1.3,
        #     zorder=2,
        # )

        if inner_bounds and self.show_inner_group:
            ix, iy, iw, ih = inner_bounds
            _container(
                ax,
                ix,
                iy,
                iw,
                ih,
                fc=COLORS["inner_group_bg"],
                ec=COLORS["block_border"],
                lw=1.3,
                zorder=2,
            )

        ax.text(
            cx,
            outer_y1 + 0.30,
            self.title,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=COLORS["text"],
            zorder=5,
        )

        if self.show_input and y_input_anchor is not None:
            _arrow(ax, cx, y_input_anchor + 0.03, cx, outer_y0 - 0.03)
            ax.text(
                cx,
                y_input_anchor - 0.10,
                self.input_label,
                ha="center",
                va="top",
                fontsize=8.5,
                color=COLORS["dim_text"],
                zorder=5,
            )

        if self.show_output:
            _arrow(ax, cx, outer_y1 + 0.03, cx, outer_y1 + self.IO_H - 0.10)

        residuals = self._residual_groups()
        drawn_res = set()

        for i, (spec, ly) in enumerate(zip(self.layers, lys)):
            lt = spec.get("type", "generic").lower()
            label = _name(lt, spec.get("label"))
            sub = spec.get("sublabel")
            is_add = lt == "add"

            if is_add:
                _plus(ax, cx, ly, r=self.PLUS_R)
            else:
                text_color = (
                    "white"
                    if lt
                    in {
                        "self_attn",
                        "mha",
                        "sdp",
                        "gated_attn",
                        "probsparse",
                        "cosine",
                        "local",
                    }
                    else COLORS["text"]
                )

                _box(
                    ax,
                    cx,
                    ly,
                    self.BOX_W,
                    self.BOX_H,
                    label,
                    color=_color(lt),
                    sublabel=sub,
                    fontsize=8.9,
                    text_color=text_color,
                    zorder=4,
                )

            if i > 0:
                prev_lt = self.layers[i - 1].get("type", "").lower()
                y_from = lys[i - 1] + (
                    self.PLUS_R if prev_lt == "add" else self.BOX_H / 2
                )
                y_to = ly - (self.PLUS_R if is_add else self.BOX_H / 2)
                _arrow(ax, cx, y_from, cx, y_to, lw=1.3, zorder=3)

            for grp in residuals:
                si, ei = grp
                if i == ei and grp not in drawn_res:
                    drawn_res.add(grp)
                    self._draw_residual(ax, cx, lys, si, ei, rx)

        if self.num_repeats is not None and inner_bounds:
            ix, iy, iw, ih = inner_bounds
            brace_x = ix - iw / 2 - 0.28
            _curly_brace(
                ax,
                brace_x,
                iy - ih / 2,
                iy + ih / 2,
                width=0.22,
                lw=1.8,
                facing="right",
                color=COLORS["text"],
                zorder=5,
            )
            ax.text(
                brace_x - 0.1,
                iy,
                f"{self.num_repeats}×",
                ha="right",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=COLORS["accent"],
                zorder=5,
            )

        return outer_y1 + (self.IO_H if self.show_output else 0.0)

    def _residual_groups(self) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        for ei, spec in enumerate(self.layers):
            if spec.get("type", "").lower() != "add":
                continue
            si = ei - 1
            while si > 0:
                t = self.layers[si].get("type", "").lower()
                if t in (
                    "add",
                    "embedding",
                    "tokenizer",
                    "query_input",
                    "memory",
                    "output",
                ):
                    break
                if self._is_final_norm(si):
                    break
                si -= 1
            groups.append((si, ei))
        return groups

    def _draw_residual(
        self,
        ax,
        cx: float,
        lys: Sequence[float],
        si: int,
        ei: int,
        rx: float,
    ):
        src_lt = self.layers[si].get("type", "").lower()
        is_add = src_lt == "add"
        y_tap = lys[si] + (self.PLUS_R if is_add else self.BOX_H / 2)
        y_dst = lys[ei]
        _residual_skip(
            ax,
            x_right=cx + self.BOX_W / 2,
            rx=rx,
            y_start=y_tap,
            y_end=y_dst,
            plus_r=self.PLUS_R,
        )

    def cross_attn_port(self) -> tuple[float, float] | None:
        for i, spec in enumerate(self.layers):
            if spec.get("type", "").lower() == "cross_attn":
                return (self._cx - self.BOX_W / 2, self._lys[i])
        return None

    def encoder_mem_port(self) -> tuple[float, float]:
        bw = self.block_width()
        return (self._cx + bw / 2, (self._y0 + self._y1) / 2)

    def stack_bounds(self) -> tuple[float, float, float, float] | None:
        return self._stack_bounds

    def inner_bounds(self) -> tuple[float, float, float, float] | None:
        return self._inner_bounds


# -----------------------------------------------------------------------------
# Optional FFN zoom panel
# -----------------------------------------------------------------------------


def _draw_ffn_zoom_panel(
    ax,
    x: float,
    y: float,
    *,
    title: str | None = None,
    ffn_type: str = "swiglu",
    hidden_dim_text: str | None = None,
    scale: float = 1.0,
):
    w = 3.1 * scale
    h = 2.35 * scale
    _container(
        ax,
        x,
        y,
        w,
        h,
        fc=COLORS["zoom_bg"],
        ec=COLORS["block_border"],
        lw=1.3,
        dashed=True,
        zorder=2,
    )

    bw = 1.15 * scale
    bh = 0.34 * scale
    left_x = x - 0.55 * scale
    right_x = x + 0.62 * scale
    top_y = y + 0.70 * scale
    mid_y = y + 0.12 * scale
    bot_y = y - 0.48 * scale
    mul_x = x + 0.10 * scale
    mul_y = y + 0.42 * scale

    _box(
        ax,
        left_x,
        bot_y,
        bw,
        bh,
        "Linear layer",
        COLORS["generic"],
        fontsize=8.5 * scale,
    )
    _box(
        ax,
        left_x,
        mid_y,
        bw,
        bh,
        "SiLU activation" if ffn_type in {"swiglu", "geglu"} else "Activation",
        COLORS["generic"],
        fontsize=8.5 * scale,
    )
    _box(
        ax,
        right_x,
        mid_y,
        bw,
        bh,
        "Linear layer",
        COLORS["generic"],
        fontsize=8.5 * scale,
    )
    _box(
        ax,
        left_x,
        top_y,
        bw,
        bh,
        "Linear layer",
        COLORS["generic"],
        fontsize=8.5 * scale,
    )

    c = Circle(
        (mul_x, mul_y),
        0.12 * scale,
        facecolor="#DDDDDD",
        edgecolor=COLORS["block_border"],
        linewidth=1.1,
        zorder=4,
    )
    ax.add_patch(c)
    ax.text(
        mul_x,
        mul_y,
        "×",
        ha="center",
        va="center",
        fontsize=10 * scale,
        color=COLORS["text"],
        zorder=5,
    )

    _arrow(ax, left_x, bot_y + bh / 2, left_x, mid_y - bh / 2, lw=1.3)
    _arrow(ax, left_x, mid_y + bh / 2, left_x, top_y - bh / 2, lw=1.3)
    ax.plot(
        [right_x - bw / 2, mul_x + 0.12 * scale],
        [mid_y, mid_y],
        color=COLORS["arrow"],
        lw=1.3,
        zorder=3,
    )
    ax.plot(
        [mul_x, mul_x],
        [mid_y, mul_y - 0.12 * scale],
        color=COLORS["arrow"],
        lw=1.3,
        zorder=3,
    )
    ax.plot(
        [mul_x, left_x],
        [mul_y + 0.12 * scale, mul_y + 0.12 * scale],
        color=COLORS["arrow"],
        lw=1.3,
        zorder=3,
    )
    _arrow(ax, mul_x, mul_y + 0.12 * scale, left_x, top_y - bh / 2, lw=1.3)

    if title:
        ax.text(
            x,
            y + h / 2 + 0.20,
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
            zorder=5,
        )

    if hidden_dim_text:
        ax.plot(
            [x - 0.05 * scale, x - 0.15 * scale],
            [y - h / 2, y - h / 2 - 0.65 * scale],
            color=COLORS["text"],
            lw=1.0,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
        _accent_text(
            ax,
            x - 0.90 * scale,
            y - h / 2 - 0.88 * scale,
            "Hidden layer\ndimension of ",
            hidden_dim_text,
            "",
            fontsize=12,
            ha="left",
            va="top",
        )

    return (x, y, w, h)


# -----------------------------------------------------------------------------
# Architecture extraction
# -----------------------------------------------------------------------------

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


def _best_choice(weights: Any, default: str = "unknown") -> str:
    if not isinstance(weights, dict) or not weights:
        return default
    return str(max(weights.items(), key=lambda item: float(item[1]))[0])


def _compact_choice(
    value: Any, mapping: dict[str, str] | None = None, default: str = "unknown"
) -> str:
    if value is None:
        return default
    key = str(value)
    if not key:
        return default
    return mapping.get(key, key.replace("_", " ")) if mapping else key.replace("_", " ")


def _top_level_transformer(module_obj: Any):
    if module_obj is None:
        return None
    return (
        getattr(module_obj, "transformer", None)
        or getattr(module_obj, "rnn", None)
        or module_obj
    )


def extract_selected_transformer_spec(model: Any) -> dict[str, Any]:
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
            "num_layers": int(getattr(enc, "num_layers", 1))
            if enc is not None
            else None,
            "tokenizer": _best_choice(
                weights.get("encoder_tokenizer"),
                default=getattr(
                    enc_wrapper,
                    "patching_mode",
                    getattr(enc, "patching_mode", "direct"),
                ),
            ),
            "self_attention": _best_choice(
                weights.get("encoder_self_attention"),
                default=getattr(
                    enc_wrapper,
                    "self_attention_type",
                    getattr(enc, "self_attention_type", "unknown"),
                ),
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
                default=getattr(
                    enc_wrapper, "ffn_mode", getattr(enc, "ffn_variant", "unknown")
                ),
            ),
        }

    decoder_spec = None
    if dec_wrapper is not None:
        decoder_spec = {
            "num_layers": int(getattr(dec, "num_layers", 1))
            if dec is not None
            else None,
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
                default=getattr(
                    dec_wrapper,
                    "self_attention_type",
                    getattr(dec, "self_attention_type", "unknown"),
                ),
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
                default=getattr(
                    dec_wrapper, "ffn_mode", getattr(dec, "ffn_variant", "unknown")
                ),
            ),
            "memory_queries": _best_choice(
                weights.get("decoder_memory_queries"),
                default="unknown",
            ),
        }

    return {
        "arch_mode": arch_mode,
        "tie_encoder_decoder_arch": bool(
            getattr(model, "tie_encoder_decoder_arch", False)
        ),
        "encoder": encoder_spec,
        "decoder": decoder_spec,
    }


def _summary_title(base: str, pairs: Iterable[tuple[str, str | None]]) -> str:
    parts = [f"{key}={value}" for key, value in pairs if value and value != "unknown"]
    if not parts:
        return base
    return f"{base}\n" + " | ".join(parts)


def make_encoder_layers_from_spec(spec: dict[str, Any]):
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
        embed_label="Input tokenizer",
        embed_sublabel=tokenizer,
        output_label="Encoder states",
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


def make_decoder_layers_from_spec(spec: dict[str, Any], *, encoder_available: bool):
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
    include_cross = bool(
        encoder_available and str(spec.get("cross_attention", "none")).lower() != "none"
    )

    embed_label = (
        "Parallel decoder input"
        if str(spec.get("style", "")).lower() == "informer"
        else "Decoder queries"
    )

    layers = make_decoder_layers(
        self_attn_type=str(spec.get("self_attention", "self_attn")),
        cross_attn=include_cross,
        ffn_type=str(spec.get("ffn", "ffn")),
        norm_type="rmsnorm",
        norm_pos="pre",
        self_attn_label=_name(str(spec.get("self_attention", "self_attn")))
        + " (masked)",
        ffn_label=_name(str(spec.get("ffn", "ffn"))),
        include_embed=True,
        include_output=True,
        embed_type="query_input",
        embed_label=embed_label,
        embed_sublabel=query_mode,
        output_label="Forecast head",
        output_sublabel=style,
        cross_attn_type="cross_attn",
        cross_attn_label="Cross-attention",
        cross_attn_sublabel=(f"{cross_attn} | {cross_pos}" if include_cross else None),
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


def draw_selected_transformer_architecture(
    model: Any, title: str | None = None, figsize=None
):
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
            bg_color=COLORS["model_bg"],
            figsize=figsize,
        )

    if arch_mode == "decoder_only" and decoder_spec is not None:
        decoder_layers, decoder_title, decoder_repeats, _ = (
            make_decoder_layers_from_spec(
                decoder_spec,
                encoder_available=False,
            )
        )
        return draw_single_block(
            decoder_layers,
            title=f"{overall_title}\n{decoder_title}",
            num_repeats=decoder_repeats,
            input_label="History-conditioned decoder input",
            bg_color=COLORS["model_bg"],
            figsize=figsize,
        )

    if (
        arch_mode == "encoder_decoder"
        and encoder_spec is not None
        and decoder_spec is not None
    ):
        encoder_layers, encoder_title, encoder_repeats = make_encoder_layers_from_spec(
            encoder_spec
        )
        decoder_layers, decoder_title, decoder_repeats, cross_label = (
            make_decoder_layers_from_spec(
                decoder_spec,
                encoder_available=True,
            )
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
    ax.set_facecolor(COLORS["page_bg"])
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Unsupported or non-transformer architecture: {arch_mode}",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["text"],
        transform=ax.transAxes,
    )
    return fig, ax


# -----------------------------------------------------------------------------
# High-level figure builders
# -----------------------------------------------------------------------------


def draw_single_block(
    layers: Sequence[dict[str, Any]],
    *,
    title: str = "Transformer block",
    num_repeats: int | None = None,
    input_label: str = "Input",
    bg_color: str | None = None,
    figsize=None,
    add_ffn_zoom: bool = False,
    ffn_zoom_type: str = "swiglu",
    ffn_hidden_dim_text: str | None = None,
    stack_label: str | None = None,
):
    block = TransformerBlock(
        title,
        layers,
        num_repeats=num_repeats,
        input_label=input_label,
        bg_color=bg_color or COLORS["model_bg"],
        stack_label=stack_label,
    )

    base_w = block.block_width() + 1.1
    base_h = block.total_height() + 1.5

    extra_w = 4.1 if add_ffn_zoom else 0.0
    bw = base_w + extra_w
    bh = base_h
    figsize = figsize or (max(6.5, bw), max(8.0, bh))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["page_bg"])
    ax.set_facecolor(COLORS["page_bg"])
    ax.set_xlim(0, bw)
    ax.set_ylim(-0.2, bh)
    ax.axis("off")

    cx = 0.5 * base_w
    block.draw(ax, cx, 0.45)

    if add_ffn_zoom:
        zoom_x = base_w + 1.75
        zoom_y = 0.68 * bh
        _draw_ffn_zoom_panel(
            ax,
            zoom_x,
            zoom_y,
            ffn_type=ffn_zoom_type,
            hidden_dim_text=ffn_hidden_dim_text,
            scale=1.0,
        )

        if block.inner_bounds() is not None:
            ix, iy, iw, ih = block.inner_bounds()
            target_x = cx + 0.55
            target_y = iy + 0.18
            ax.plot(
                [target_x, zoom_x - 1.6],
                [target_y, zoom_y + 0.05],
                color=COLORS["text"],
                lw=1.1,
                linestyle=(0, (2.0, 1.5)),
                zorder=3,
            )

    plt.tight_layout(pad=0.45)
    return fig, ax


def draw_encoder_decoder(
    *,
    encoder_layers: Sequence[dict[str, Any]],
    decoder_layers: Sequence[dict[str, Any]],
    encoder_title: str = "Encoder",
    decoder_title: str = "Decoder",
    encoder_repeats: int | None = 6,
    decoder_repeats: int | None = 6,
    encoder_input: str = "Source tokens",
    decoder_input: str = "Target tokens (shifted right)",
    title: str = "Transformer architecture",
    figsize=None,
    cross_label: str = "encoder memory\n(K, V)",
):
    enc = TransformerBlock(
        encoder_title,
        encoder_layers,
        num_repeats=encoder_repeats,
        input_label=encoder_input,
        bg_color=COLORS["model_bg"],
    )
    dec = TransformerBlock(
        decoder_title,
        decoder_layers,
        num_repeats=decoder_repeats,
        input_label=decoder_input,
        bg_color=COLORS["model_bg"],
    )

    gap = 2.3
    bw = max(enc.block_width(), dec.block_width())
    tw = 2 * bw + gap + 2.6
    th = max(enc.total_height(), dec.total_height()) + 2.0
    figsize = figsize or (tw * 1.02, th * 1.02)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["page_bg"])
    ax.set_facecolor(COLORS["page_bg"])
    ax.set_xlim(0, tw)
    ax.set_ylim(-0.2, th)
    ax.axis("off")

    cx_enc = tw / 2 - bw / 2 - gap / 2
    cx_dec = tw / 2 + bw / 2 + gap / 2

    enc.draw(ax, cx_enc, 0.45)
    dec.draw(ax, cx_dec, 0.45)

    ca_port = dec.cross_attn_port()
    if ca_port:
        mem_x = cx_enc + enc.block_width() / 2
        mem_y = (enc._y0 + enc._y1) / 2
        _arrow(
            ax,
            mem_x,
            mem_y,
            ca_port[0] - 0.02,
            ca_port[1],
            lw=1.5,
            color=COLORS["cross_arrow"],
            zorder=4,
        )

        mid_x = (mem_x + ca_port[0]) / 2
        mid_y = (mem_y + ca_port[1]) / 2 - 0.35
        ax.text(
            mid_x,
            mid_y,
            cross_label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLORS["text"],
            bbox=dict(
                fc="white",
                ec=COLORS["block_border"],
                boxstyle="round,pad=0.28",
                lw=1.0,
            ),
            zorder=5,
        )

    ax.text(
        tw / 2,
        th - 0.05,
        title,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
        zorder=6,
    )

    plt.tight_layout(pad=0.45)
    return fig, ax


# -----------------------------------------------------------------------------
# Raschka-like model summary figure
# -----------------------------------------------------------------------------


def draw_model_summary(
    *,
    title: str,
    layers: Sequence[dict[str, Any]],
    num_repeats: int,
    input_label: str = "Sample input text",
    tokenizer_label: str = "Tokenized text",
    top_vocab_text: str | None = None,
    embedding_dim_text: str | None = None,
    hidden_dim_text: str | None = None,
    context_len_text: str | None = None,
    ffn_zoom_type: str = "swiglu",
    figsize=None,
):
    block = TransformerBlock(
        title,
        layers,
        num_repeats=num_repeats,
        input_label=tokenizer_label,
        bg_color=COLORS["model_bg"],
    )

    base_w = block.block_width() + 4.8
    base_h = block.total_height() + 2.0
    figsize = figsize or (9.5, 11.0)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["page_bg"])
    ax.set_facecolor(COLORS["page_bg"])
    ax.set_xlim(0, base_w)
    ax.set_ylim(-0.2, base_h)
    ax.axis("off")

    cx = 3.6
    block.draw(ax, cx, 1.25)

    ax.text(
        cx,
        base_h - 0.28,
        title,
        ha="center",
        va="bottom",
        fontsize=28,
        fontweight="bold",
        color=COLORS["accent"],
        zorder=6,
    )

    ax.text(
        cx,
        0.15,
        input_label,
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["text"],
        zorder=6,
    )
    _arrow(ax, cx, 0.45, cx, 0.82, lw=1.4)

    if top_vocab_text:
        ox = block._cx + 1.25
        oy = block._y1 + 0.95
        ax.plot(
            [block._cx + 0.72, ox - 0.18],
            [block._y1 - 0.05, oy - 0.10],
            color=COLORS["text"],
            lw=1.0,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
        _accent_text(
            ax,
            ox,
            oy,
            "Vocabulary size of ",
            top_vocab_text,
            "",
            fontsize=15,
            ha="left",
            va="center",
        )

    if embedding_dim_text:
        ex = block._cx + 2.2
        ey = block._y0 + 1.18
        ax.plot(
            [block._cx + 0.62, ex - 0.15],
            [block._y0 + 1.00, ey + 0.05],
            color=COLORS["text"],
            lw=1.0,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
        _accent_text(
            ax,
            ex,
            ey,
            "Embedding\ndimension of ",
            embedding_dim_text,
            "",
            fontsize=15,
            ha="left",
            va="center",
        )

    if context_len_text:
        lx = block._cx - 3.0
        ly = block._y0 + 2.3
        ax.plot(
            [block._cx - 2.1, lx + 0.80],
            [block._y0 + 3.3, ly + 0.75],
            color=COLORS["text"],
            lw=1.0,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
        _accent_text(
            ax,
            lx,
            ly,
            "Supported\ncontext length\nof ",
            context_len_text,
            " tokens",
            fontsize=15,
            ha="left",
            va="center",
        )

    zoom_x = block._cx + 4.85
    zoom_y = block._y0 + 5.0
    _draw_ffn_zoom_panel(
        ax,
        zoom_x,
        zoom_y,
        ffn_type=ffn_zoom_type,
        hidden_dim_text=hidden_dim_text,
        scale=1.05,
    )

    if block.inner_bounds() is not None:
        ix, iy, iw, ih = block.inner_bounds()
        target_x = block._cx + 0.55
        target_y = iy + 0.25
        ax.plot(
            [target_x, zoom_x - 1.65],
            [target_y, zoom_y + 0.10],
            color=COLORS["text"],
            lw=1.0,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )

    plt.tight_layout(pad=0.45)
    return fig, ax


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    OUT = "outputs"
    os.makedirs(OUT, exist_ok=True)

    fig, _ = draw_encoder_decoder(
        encoder_layers=make_encoder_layers(
            attn_type="mha",
            ffn_type="feedforward",
            norm_type="layernorm",
            norm_pos="post",
        ),
        decoder_layers=make_decoder_layers(
            self_attn_type="mha",
            cross_attn=True,
            ffn_type="feedforward",
            norm_type="layernorm",
            norm_pos="post",
        ),
        encoder_title="Encoder",
        decoder_title="Decoder",
        encoder_repeats=6,
        decoder_repeats=6,
        encoder_input="Source tokens",
        decoder_input="Target tokens (shifted)",
        title="Classic Transformer",
        figsize=(13.5, 9.0),
    )
    fig.savefig(f"{OUT}/1_transformer_classic_raschka_style.png", bbox_inches="tight")
    print("Saved: 1_transformer_classic_raschka_style.png")

    fig, _ = draw_single_block(
        make_decoder_layers(
            self_attn_type="mha",
            cross_attn=False,
            ffn_type="swiglu",
            norm_type="rmsnorm",
            norm_pos="pre",
            embed_label="Token embedding layer",
            output_label="Linear output layer",
        ),
        title="GPT / LLaMA-style decoder",
        num_repeats=32,
        input_label="Tokenized text",
        bg_color=COLORS["model_bg"],
        add_ffn_zoom=True,
        ffn_zoom_type="swiglu",
        ffn_hidden_dim_text="11,008",
    )
    fig.savefig(f"{OUT}/2_gpt_llama_raschka_style.png", bbox_inches="tight")
    print("Saved: 2_gpt_llama_raschka_style.png")

    fig, _ = draw_single_block(
        make_encoder_layers(
            attn_type="mha",
            ffn_type="feedforward",
            norm_type="layernorm",
            norm_pos="post",
        ),
        title="BERT encoder",
        num_repeats=12,
        input_label="[CLS] token sequence",
        bg_color=COLORS["model_bg"],
    )
    fig.savefig(f"{OUT}/3_bert_encoder_raschka_style.png", bbox_inches="tight")
    print("Saved: 3_bert_encoder_raschka_style.png")

    fig, _ = draw_single_block(
        make_encoder_layers(
            attn_type="mha",
            ffn_type="moe",
            norm_type="rmsnorm",
            norm_pos="pre",
        ),
        title="MoE encoder",
        num_repeats=24,
        input_label="Tokenized text",
        bg_color=COLORS["model_bg"],
    )
    fig.savefig(f"{OUT}/4_moe_encoder_raschka_style.png", bbox_inches="tight")
    print("Saved: 4_moe_encoder_raschka_style.png")

    fig, _ = draw_single_block(
        make_hybrid_layers(
            linear_attn_type="deltanet",
            full_attn_type="gated_attn",
            ffn_type="moe",
            norm_type="rmsnorm",
            norm_pos="pre",
            ratio=3,
        ),
        title="Hybrid DeltaNet / MoE",
        num_repeats=24,
        input_label="Tokenized text",
        bg_color=COLORS["model_bg"],
    )
    fig.savefig(f"{OUT}/5_deltanet_moe_hybrid_raschka_style.png", bbox_inches="tight")
    print("Saved: 5_deltanet_moe_hybrid_raschka_style.png")

    fig, _ = draw_encoder_decoder(
        encoder_layers=make_encoder_layers(
            attn_type="mha",
            ffn_type="feedforward",
            norm_type="rmsnorm",
            norm_pos="pre",
            include_embed=False,
            include_output=False,
        ),
        decoder_layers=make_encoder_layers(
            attn_type="mha",
            ffn_type="feedforward",
            norm_type="layernorm",
            norm_pos="post",
            include_embed=False,
            include_output=False,
        ),
        encoder_title="Pre-norm",
        decoder_title="Post-norm",
        encoder_repeats=None,
        decoder_repeats=None,
        encoder_input="x (residual stream)",
        decoder_input="x (residual stream)",
        title="Pre-norm vs post-norm",
        figsize=(12.0, 7.5),
    )
    fig.savefig(f"{OUT}/6_prenorm_vs_postnorm_raschka_style.png", bbox_inches="tight")
    print("Saved: 6_prenorm_vs_postnorm_raschka_style.png")

    olmo_layers = make_decoder_layers(
        self_attn_type="mha",
        cross_attn=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        norm_pos="pre",
        embed_label="Token embedding layer",
        output_label="Linear output layer",
    )
    fig, _ = draw_model_summary(
        title="OLMo 2 7B",
        layers=olmo_layers,
        num_repeats=32,
        input_label="Sample input text",
        tokenizer_label="Tokenized text",
        top_vocab_text="100k",
        embedding_dim_text="4,096",
        hidden_dim_text="11,008",
        context_len_text="4k",
        ffn_zoom_type="swiglu",
        figsize=(11.0, 12.0),
    )
    fig.savefig(f"{OUT}/7_olmo_like_summary_raschka_style.png", bbox_inches="tight")
    print("Saved: 7_olmo_like_summary_raschka_style.png")

    plt.close("all")
    print(f"\nAll diagrams saved to {OUT}/")
