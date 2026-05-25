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

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path

from darts.config import DEFAULT_OP_FAMILIES

# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------

matplotlib.rcParams.update({
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
})


# -----------------------------------------------------------------------------
# Palette: flat, pastel, textbook-like
# -----------------------------------------------------------------------------

COLORS = {
    "page_bg": "#FFFFFF",
    "model_bg": "#F5F5F7",
    "stack_bg": "#E8CAD8",
    "inner_group_bg": "#E8CAD8",
    "zoom_bg": "#F4F4F6",
    "embedding": "#FCFCFD",
    "tokenizer": "#FCFCFD",
    "query_input": "#FCFCFD",
    "memory": "#FCFCFD",
    "norm": "#FCFCFD",
    "self_attn": "#6D7280",
    "cross_attn": "#FCFCFD",
    "linear_attn": "#D8DFF4",
    "feedforward": "#FCFCFD",
    "moe": "#FCFCFD",
    "swiglu": "#FCFCFD",
    "geglu": "#FCFCFD",
    "output": "#FCFCFD",
    "add": "#FFFFFF",
    "generic": "#FCFCFD",
    "block_border": "#202124",
    "text": "#202124",
    "dim_text": "#5F6368",
    "accent": "#B4235D",
    "arrow": "#202124",
    "residual": "#202124",
    "cross_arrow": "#202124",
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
        layers.append({
            "type": embed_type,
            "label": embed_label,
            "sublabel": embed_sublabel,
        })

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
        layers.append({
            "type": "output",
            "label": output_label,
            "sublabel": output_sublabel,
        })
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
        layers.append({
            "type": embed_type,
            "label": embed_label,
            "sublabel": embed_sublabel,
        })

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
        layers.append({
            "type": "output",
            "label": output_label,
            "sublabel": output_sublabel,
        })
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


def _edge_index(node_idx: int, input_idx: int) -> int:
    return sum(range(node_idx)) + input_idx


def _edge_source_target(edge_idx: int) -> tuple[int, int]:
    edge_idx_remaining = int(edge_idx)
    target = 1
    while edge_idx_remaining >= target:
        edge_idx_remaining -= target
        target += 1
    return edge_idx_remaining, target


def _safe_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list):
            return [float(v) for v in data]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return []


def _best_edge_choice(edge: Any) -> tuple[str | None, float | None]:
    available_ops = list(getattr(edge, "available_ops", []) or [])
    if not available_ops:
        return None, None

    getter = getattr(edge, "get_alphas", None)
    if not callable(getter):
        return available_ops[0], None

    try:
        logits = _safe_float_list(getter())
    except Exception:
        logits = []

    if not logits:
        return available_ops[0], None

    num_ops = min(len(logits), len(available_ops))
    best_idx = max(range(num_ops), key=lambda idx: logits[idx])
    max_logit = max(logits[:num_ops])
    exp_vals = [math.exp(value - max_logit) for value in logits[:num_ops]]
    denom = sum(exp_vals) or 1.0
    return available_ops[best_idx], exp_vals[best_idx] / denom


def _edge_selection_weights_for_diagram(edge: Any):
    if not hasattr(edge, "available_ops"):
        return None

    if (
        hasattr(edge, "use_hierarchical")
        and edge.use_hierarchical
        and hasattr(edge, "_get_weights")
        and hasattr(edge, "ops")
    ):
        try:
            routed = edge._get_weights(top_k=None)
            if routed:
                probs = torch.zeros(len(edge.ops), dtype=routed[0][1].dtype)
                for op_idx, weight in routed:
                    probs[int(op_idx)] = probs[int(op_idx)] + weight.detach().cpu()
                return probs
        except Exception:
            pass

    getter = getattr(edge, "get_alphas", None)
    if not callable(getter):
        return None

    try:
        alpha_like = getter()
    except Exception:
        return None

    if not isinstance(alpha_like, torch.Tensor) or alpha_like.numel() == 0:
        return None

    with torch.no_grad():
        flat = alpha_like.detach().reshape(-1).cpu()
        finite_ok = torch.isfinite(flat).all().item()
        in_range = flat.min().item() >= -1e-6 and flat.max().item() <= 1.0 + 1e-6
        sum_close = abs(flat.sum().item() - 1.0) <= 1e-4
        looks_like_probs = finite_ok and in_range and sum_close
        if looks_like_probs:
            probs = flat.clamp_min(1e-8)
            return probs / probs.sum().clamp_min(1e-8)
        return torch.softmax(flat, dim=0)


def _fixed_edge_operation_name(edge: Any) -> str | None:
    op = getattr(edge, "op", None)
    if op is None:
        return None
    class_name = type(op).__name__
    if class_name.endswith("Op"):
        class_name = class_name[:-2]
    return class_name or None


def _edge_operation_names(edge: Any) -> list[str]:
    available_ops = getattr(edge, "available_ops", None)
    if available_ops is not None:
        return [str(name) for name in available_ops]
    fixed_name = _fixed_edge_operation_name(edge)
    if fixed_name is not None:
        return [fixed_name]
    return []


def _assign_cell_edges_with_diversity_for_diagram(
    cell: Any,
    edge_weight_list: list[torch.Tensor | None],
    edge_importance: list[float] | None = None,
) -> tuple[list[int], dict[str, int]]:
    n_edges = len(getattr(cell, "edges", []))
    if n_edges == 0:
        return [], {}
    if edge_importance is None or len(edge_importance) != n_edges:
        edge_importance = [1.0] * n_edges

    assignments: list[int | None] = [None] * n_edges
    op_counts_by_name: dict[str, int] = {}
    used_unique_ops = set()

    available_names = set()
    for edge in cell.edges:
        edge_names = getattr(edge, "available_ops", None)
        if edge_names is not None:
            available_names.update(edge_names)

    unique_pool = [name for name in available_names if name != "Identity"]
    pool_size = len(unique_pool) if unique_pool else len(available_names)
    target_unique = min(max(2, int(math.ceil(n_edges * 0.5))), max(pool_size, 1))

    candidates = []
    for edge_idx, weights in enumerate(edge_weight_list):
        if weights is None or weights.numel() == 0:
            continue
        for op_idx, weight in enumerate(weights):
            score = float(weight.item()) * (
                0.65 + 0.35 * float(edge_importance[edge_idx])
            )
            candidates.append((score, edge_idx, int(op_idx)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, edge_idx, op_idx in candidates:
        if len(used_unique_ops) >= target_unique:
            break
        if assignments[edge_idx] is not None:
            continue
        edge = cell.edges[edge_idx]
        edge_op_names = _edge_operation_names(edge)
        if not edge_op_names:
            continue
        op_name = edge_op_names[op_idx]
        if op_name in used_unique_ops:
            continue
        if op_name == "Identity" and len(used_unique_ops) < max(target_unique - 1, 1):
            continue
        assignments[edge_idx] = op_idx
        used_unique_ops.add(op_name)
        op_counts_by_name[op_name] = op_counts_by_name.get(op_name, 0) + 1

    remaining_edges = [idx for idx in range(n_edges) if assignments[idx] is None]
    remaining_edges.sort(
        key=lambda edge_idx: (
            float(edge_weight_list[edge_idx].max().item())
            * float(edge_importance[edge_idx])
            if edge_weight_list[edge_idx] is not None
            and edge_weight_list[edge_idx].numel() > 0
            else -1e9
        ),
        reverse=True,
    )

    max_per_op = max(1, int(math.ceil(n_edges / max(target_unique, 1))))
    repeat_penalty = 0.30
    identity_extra_penalty = 0.40

    for edge_idx in remaining_edges:
        edge = cell.edges[edge_idx]
        weights = edge_weight_list[edge_idx]
        if weights is None or weights.numel() == 0:
            chosen_idx = 0
        else:
            adjusted = weights.clone()
            edge_op_names = _edge_operation_names(edge)
            for op_idx, op_name in enumerate(edge_op_names):
                count = op_counts_by_name.get(op_name, 0)
                penalty = repeat_penalty * float(count)
                if op_name == "Identity":
                    penalty += identity_extra_penalty * float(count)
                    if float(edge_importance[edge_idx]) < 0.35:
                        penalty -= 0.15
                adjusted[op_idx] = adjusted[op_idx] - penalty

            ranked = torch.argsort(adjusted, descending=True).tolist()
            chosen_idx = int(ranked[0])
            for candidate_idx in ranked:
                candidate_name = edge_op_names[int(candidate_idx)]
                count = op_counts_by_name.get(candidate_name, 0)
                if count < max_per_op or int(candidate_idx) == ranked[0]:
                    chosen_idx = int(candidate_idx)
                    break

        assignments[edge_idx] = chosen_idx
        chosen_name = _edge_operation_names(edge)[chosen_idx]
        op_counts_by_name[chosen_name] = op_counts_by_name.get(chosen_name, 0) + 1

    max_repeat = max(1, int(math.ceil(n_edges * 0.40)))
    for _ in range(n_edges * 2):
        changed = False
        by_freq = sorted(
            op_counts_by_name.items(), key=lambda item: item[1], reverse=True
        )
        for op_name, count in by_freq:
            if count <= max_repeat:
                continue

            holders = []
            for edge_idx, op_idx in enumerate(assignments):
                if op_idx is None:
                    continue
                edge = cell.edges[edge_idx]
                edge_op_names = _edge_operation_names(edge)
                if not edge_op_names:
                    continue
                chosen_name = edge_op_names[int(op_idx)]
                if chosen_name != op_name:
                    continue
                weights = edge_weight_list[edge_idx]
                confidence = (
                    float(weights[int(op_idx)].item())
                    if weights is not None and weights.numel() > int(op_idx)
                    else -1e9
                )
                holders.append((
                    confidence * float(edge_importance[edge_idx]),
                    edge_idx,
                ))

            holders.sort(key=lambda item: item[0])
            replaced = False
            for _, edge_idx in holders:
                edge = cell.edges[edge_idx]
                weights = edge_weight_list[edge_idx]
                edge_op_names = _edge_operation_names(edge)
                ranked = (
                    list(range(len(edge_op_names)))
                    if weights is None or weights.numel() == 0
                    else torch.argsort(weights, descending=True).tolist()
                )

                for cand_idx in ranked:
                    cand_idx = int(cand_idx)
                    cand_name = edge_op_names[cand_idx]
                    if cand_name == op_name:
                        continue
                    cand_count = op_counts_by_name.get(cand_name, 0)
                    if cand_count >= max_repeat:
                        continue
                    if (
                        cand_name == "Identity"
                        and float(edge_importance[edge_idx]) > 0.65
                    ):
                        continue

                    old_idx = int(assignments[edge_idx])
                    old_name = edge_op_names[old_idx]
                    assignments[edge_idx] = cand_idx
                    op_counts_by_name[old_name] = op_counts_by_name.get(old_name, 1) - 1
                    if op_counts_by_name[old_name] <= 0:
                        op_counts_by_name.pop(old_name, None)
                    op_counts_by_name[cand_name] = cand_count + 1
                    replaced = True
                    changed = True
                    break

                if replaced:
                    break

        if not changed:
            break

    final_assignments = [int(idx) if idx is not None else 0 for idx in assignments]
    return final_assignments, op_counts_by_name


def extract_logged_cell_specs(model: Any, log_text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"Cell\s+(?P<cell>\d+),\s+Edge\s+(?P<edge>\d+):\s+(?P<op>[A-Za-z0-9_]+)Op(?:\s+\(weight:\s*(?P<weight>[-+]?[0-9]*\.?[0-9]+)\))?"
    )
    matches = list(pattern.finditer(log_text or ""))
    if not matches:
        return []

    cells = list(getattr(model, "cells", []) or [])
    grouped: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        cell_idx = int(match.group("cell"))
        edge_idx = int(match.group("edge"))
        if cells and cell_idx >= len(cells):
            continue

        source, target = _edge_source_target(edge_idx)

        weight_str = match.group("weight")
        grouped.setdefault(cell_idx, []).append({
            "edge_idx": edge_idx,
            "source": source,
            "target": target,
            "operation": match.group("op"),
            "op_weight": float(weight_str) if weight_str is not None else None,
            "importance": 1.0,
        })

    specs: list[dict[str, Any]] = []
    for cell_idx, selected_edges in sorted(grouped.items()):
        cell = cells[cell_idx] if cell_idx < len(cells) else None
        inferred_num_nodes = (
            max(int(edge_spec["target"]) for edge_spec in selected_edges) + 1
        )
        num_nodes = int(getattr(cell, "num_nodes", 0) or inferred_num_nodes)
        selected_edges.sort(
            key=lambda edge_spec: (edge_spec["target"], edge_spec["source"])
        )
        node_roles = []
        for node_idx in range(1, num_nodes):
            incoming = [edge for edge in selected_edges if edge["target"] == node_idx]
            label, sublabel = _node_role_from_edges(node_idx, incoming)
            node_roles.append({
                "node_idx": node_idx,
                "label": label,
                "sublabel": sublabel,
            })
        specs.append({
            "cell_idx": cell_idx,
            "num_nodes": num_nodes,
            "selected_edges": selected_edges,
            "selected_ops": sorted({edge["operation"] for edge in selected_edges}),
            "node_roles": node_roles,
        })

    return specs


def extract_selected_cell_specs(model: Any) -> list[dict[str, Any]]:
    cells = list(getattr(model, "cells", []) or [])
    if not cells:
        return []

    specs: list[dict[str, Any]] = []
    for cell_idx, cell in enumerate(cells):
        num_nodes = int(getattr(cell, "num_nodes", 0) or 0)
        if num_nodes < 2:
            continue

        stored_selected_edges = getattr(cell, "selected_edge_specs", None)
        if stored_selected_edges:
            selected_edges = [
                {
                    "edge_idx": int(edge_spec.get("edge_idx", idx)),
                    "source": int(edge_spec["source"]),
                    "target": int(edge_spec["target"]),
                    "operation": str(edge_spec["operation"]),
                    "op_weight": edge_spec.get("op_weight"),
                    "importance": float(edge_spec.get("importance", 1.0)),
                }
                for idx, edge_spec in enumerate(stored_selected_edges)
                if "source" in edge_spec
                and "target" in edge_spec
                and "operation" in edge_spec
            ]
            selected_edges.sort(
                key=lambda edge_spec: (edge_spec["target"], edge_spec["source"])
            )
            node_roles = []
            for node_idx in range(1, num_nodes):
                target_edges = [
                    edge_spec
                    for edge_spec in selected_edges
                    if int(edge_spec["target"]) == node_idx
                ]
                label, sublabel = _node_role_from_edges(node_idx, target_edges)
                node_roles.append({
                    "node_idx": node_idx,
                    "label": label,
                    "sublabel": sublabel,
                })
            specs.append({
                "cell_idx": cell_idx,
                "num_nodes": num_nodes,
                "selected_edges": selected_edges,
                "selected_ops": sorted({edge["operation"] for edge in selected_edges}),
                "node_roles": node_roles,
            })
            continue

        importance_vals = []
        edge_importance = getattr(cell, "edge_importance", None)
        if edge_importance is not None and hasattr(edge_importance, "sigmoid"):
            importance_vals = _safe_float_list(edge_importance.sigmoid())

        selected_edges: list[dict[str, Any]] = []
        cell_edges = list(getattr(cell, "edges", []))
        if cell_edges and all(
            not hasattr(edge, "available_ops") for edge in cell_edges
        ):
            selected_indices = [0] * len(cell_edges)
            edge_weight_list = [None] * len(cell_edges)
        else:
            edge_weight_list = [
                _edge_selection_weights_for_diagram(edge) for edge in cell_edges
            ]
            selected_indices, _ = _assign_cell_edges_with_diversity_for_diagram(
                cell,
                edge_weight_list,
                edge_importance=importance_vals,
            )

        for edge_idx, op_idx in enumerate(selected_indices):
            edge = cell.edges[edge_idx]
            op_idx = int(op_idx)
            edge_op_names = _edge_operation_names(edge)
            if not edge_op_names:
                continue
            op_name = str(edge_op_names[op_idx])
            weights = edge_weight_list[edge_idx]
            op_weight = (
                float(weights[op_idx].item())
                if weights is not None and weights.numel() > op_idx
                else None
            )

            source, target = _edge_source_target(edge_idx)
            importance = (
                float(importance_vals[edge_idx])
                if edge_idx < len(importance_vals)
                else 1.0
            )

            selected_edges.append({
                "edge_idx": edge_idx,
                "source": source,
                "target": target,
                "operation": op_name,
                "op_weight": op_weight,
                "importance": importance,
            })

        selected_edges.sort(
            key=lambda edge_spec: (edge_spec["target"], edge_spec["source"])
        )
        node_roles = []
        for node_idx in range(1, num_nodes):
            target_edges = [
                edge_spec
                for edge_spec in selected_edges
                if int(edge_spec["target"]) == node_idx
            ]
            label, sublabel = _node_role_from_edges(node_idx, target_edges)
            node_roles.append({
                "node_idx": node_idx,
                "label": label,
                "sublabel": sublabel,
            })
        specs.append({
            "cell_idx": cell_idx,
            "num_nodes": num_nodes,
            "selected_edges": selected_edges,
            "selected_ops": sorted({edge["operation"] for edge in selected_edges}),
            "node_roles": node_roles,
        })

    return specs


def _selected_edge_summary(selected_edges: Sequence[dict[str, Any]]) -> str:
    parts = []
    for edge_spec in sorted(
        selected_edges,
        key=lambda item: int(item.get("edge_idx", 0)),
    ):
        op_name = str(edge_spec.get("operation", "op"))
        weight = edge_spec.get("op_weight")
        weight_value = float(weight) if weight is not None else None
        weight_text = (
            f" w={weight_value:.3f}"
            if weight_value is not None and math.isfinite(weight_value)
            else ""
        )
        parts.append(
            f"E{int(edge_spec.get('edge_idx', 0))} "
            f"{int(edge_spec.get('source', 0))}->{int(edge_spec.get('target', 0))}: "
            f"{op_name}{weight_text}"
        )
    return " | ".join(parts)


def _draw_selected_operation_node_panel(
    ax,
    *,
    cell_x: float,
    cell_y: float,
    cell_w: float,
    cell_h: float,
    cell_spec: dict[str, Any],
    single_cell: bool,
) -> bool:
    selected_edges = sorted(
        list(cell_spec.get("selected_edges", [])),
        key=lambda edge_spec: int(edge_spec.get("edge_idx", 0)),
    )
    if not selected_edges:
        return False

    max_target = max(int(edge_spec.get("target", 1)) for edge_spec in selected_edges)
    center_y = cell_y - 0.10
    usable_w = cell_w - (0.92 if single_cell else 1.10)
    input_x = cell_x - cell_w / 2 + (0.46 if single_cell else 0.55)
    output_x = input_x + usable_w
    span = output_x - input_x

    target_groups: dict[int, list[dict[str, Any]]] = {}
    for edge_spec in selected_edges:
        target_groups.setdefault(int(edge_spec.get("target", 1)), []).append(edge_spec)

    op_positions: dict[int, tuple[float, float]] = {}
    for target, target_edges in sorted(target_groups.items()):
        target_edges.sort(
            key=lambda edge_spec: (
                int(edge_spec.get("source", 0)),
                int(edge_spec.get("edge_idx", 0)),
            )
        )
        x_pos = input_x + span * (target / max(max_target + 1, 1))
        if len(target_edges) == 1:
            offsets = [0.0]
        else:
            step = min(0.62, 1.18 / max(len(target_edges) - 1, 1))
            offsets = [
                (len(target_edges) - 1) * step / 2 - idx * step
                for idx in range(len(target_edges))
            ]
        for edge_spec, offset in zip(target_edges, offsets, strict=False):
            op_positions[int(edge_spec.get("edge_idx", 0))] = (x_pos, center_y + offset)

    incoming_by_state: dict[int, list[dict[str, Any]]] = {}
    for edge_spec in selected_edges:
        incoming_by_state.setdefault(int(edge_spec.get("target", 0)), []).append(
            edge_spec
        )

    ax.add_patch(
        Circle(
            (input_x, center_y),
            0.12,
            facecolor="#FFFFFF",
            edgecolor=COLORS["block_border"],
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.text(
        input_x,
        center_y + 0.18,
        "Input",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=COLORS["text"],
        zorder=6,
    )

    ax.add_patch(
        Circle(
            (output_x, center_y),
            0.12,
            facecolor="#FFFFFF",
            edgecolor=COLORS["block_border"],
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.text(
        output_x,
        center_y + 0.18,
        "Cell out",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=COLORS["text"],
        zorder=6,
    )

    def draw_conn(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
        dashed: bool = False,
        rad: float = 0.0,
        lw: float = 1.4,
        zorder: int = 3,
    ) -> None:
        ax.annotate(
            "",
            xy=(end[0] - 0.12, end[1]),
            xytext=(start[0] + 0.12, start[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=10,
                linestyle=(0, (3.0, 2.0)) if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )

    for edge_spec in selected_edges:
        edge_idx = int(edge_spec.get("edge_idx", 0))
        source = int(edge_spec.get("source", 0))
        op_name = str(edge_spec.get("operation", "op"))
        color = _edge_op_color(op_name)
        op_pos = op_positions[edge_idx]
        if source == 0:
            sources = [(input_x, center_y)]
        else:
            sources = [
                op_positions[int(source_edge.get("edge_idx", 0))]
                for source_edge in incoming_by_state.get(source, [])
                if int(source_edge.get("edge_idx", 0)) in op_positions
            ]
        if not sources:
            sources = [(input_x, center_y)]

        for source_rank, source_pos in enumerate(sources):
            rad = 0.08 * (source_rank - (len(sources) - 1) / 2)
            draw_conn(
                source_pos,
                op_pos,
                color=color,
                dashed=op_name == "Identity",
                rad=rad,
                lw=1.4 if op_name == "Identity" else 2.1,
            )

    for edge_spec in incoming_by_state.get(max_target, []):
        edge_idx = int(edge_spec.get("edge_idx", 0))
        op_name = str(edge_spec.get("operation", "op"))
        if edge_idx not in op_positions:
            continue
        draw_conn(
            op_positions[edge_idx],
            (output_x, center_y),
            color=COLORS["arrow"],
            rad=0.05 * (edge_idx - len(selected_edges) / 2),
            lw=1.25,
            zorder=2,
        )

    for edge_spec in selected_edges:
        edge_idx = int(edge_spec.get("edge_idx", 0))
        op_name = str(edge_spec.get("operation", "op"))
        op_x, op_y = op_positions[edge_idx]
        color = _edge_op_color(op_name)
        ax.add_patch(
            Circle(
                (op_x, op_y),
                0.13,
                facecolor="#FFFFFF",
                edgecolor=color,
                linewidth=1.25,
                linestyle=(0, (3.0, 2.0)) if op_name == "Identity" else "-",
                zorder=5,
            )
        )
        weight = edge_spec.get("op_weight")
        weight_value = float(weight) if weight is not None else None
        weight_line = (
            f"w={weight_value:.3f}"
            if weight_value is not None and math.isfinite(weight_value)
            else None
        )
        label_lines = [f"E{edge_idx}", op_name]
        if weight_line:
            label_lines.append(weight_line)
        label_above = op_y >= center_y - 0.02
        label_x = op_x if label_above else op_x + 0.20
        label_y = op_y + 0.20 if label_above else op_y
        ax.text(
            label_x,
            label_y,
            "\n".join(label_lines),
            ha="center" if label_above else "left",
            va="bottom" if label_above else "center",
            fontsize=6.9,
            color=COLORS["text"],
            linespacing=1.0,
            bbox=dict(
                fc="#FFFFFF",
                ec="#D7DCE3",
                boxstyle="round,pad=0.18",
                lw=0.8,
            ),
            zorder=6,
        )

    ax.plot(
        [input_x + 0.10, output_x - 0.10],
        [center_y - 0.38, center_y - 0.38],
        color="#C7CCD3",
        lw=0.85,
        linestyle=(0, (2.0, 1.5)),
        zorder=2,
    )
    ax.text(
        (input_x + output_x) / 2,
        center_y - 0.52,
        "global residual",
        ha="center",
        va="center",
        fontsize=7.3,
        color=COLORS["dim_text"],
        zorder=3,
    )

    ops_text = ", ".join(cell_spec.get("selected_ops", [])) or "n/a"
    ax.text(
        cell_x,
        cell_y + cell_h / 2 - (0.16 if single_cell else 0.38),
        f"selected primitives: {ops_text}",
        ha="center",
        va="top",
        fontsize=7.6,
        color=COLORS["dim_text"],
        zorder=4,
    )
    return True


def _draw_darts_cell_panel(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    cell_specs: Sequence[dict[str, Any]],
    heading: str | None = None,
):
    if not cell_specs:
        return

    if heading:
        ax.text(
            x,
            y + h / 2 + 0.28,
            heading,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=COLORS["text"],
            zorder=6,
        )
        ax.text(
            x,
            y + h / 2 + 0.10,
            "faint edges show the full cell DAG; saturated edges show the discretized selection",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=COLORS["dim_text"],
            zorder=6,
        )

    _container(
        ax,
        x,
        y,
        w,
        h,
        fc="#FBFBFC",
        lw=1.0,
        rounding=0.18,
        zorder=1,
    )

    inner_margin = 0.28
    gap = 0.34
    n_cells = max(len(cell_specs), 1)
    cell_w = (w - 2 * inner_margin - gap * (n_cells - 1)) / n_cells
    cell_h = h - 0.66
    left = x - w / 2 + inner_margin + cell_w / 2
    single_cell = n_cells == 1

    for idx, cell_spec in enumerate(cell_specs):
        cell_x = left + idx * (cell_w + gap)
        cell_y = y - 0.03
        if not single_cell:
            _container(
                ax,
                cell_x,
                cell_y,
                cell_w,
                cell_h,
                fc="#FFFFFF",
                lw=0.95,
                rounding=0.16,
                zorder=2,
            )
        if not single_cell:
            ax.text(
                cell_x,
                cell_y + cell_h / 2 - 0.16,
                f"Cell {cell_spec['cell_idx']}",
                ha="center",
                va="top",
                fontsize=10.4,
                fontweight="bold",
                color=COLORS["text"],
                zorder=6,
            )

        if _draw_selected_operation_node_panel(
            ax,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_w=cell_w,
            cell_h=cell_h,
            cell_spec=cell_spec,
            single_cell=single_cell,
        ):
            continue

        num_nodes = int(cell_spec.get("num_nodes", 0))
        internal_nodes = max(num_nodes - 1, 1)
        node_roles = {
            int(role["node_idx"]): role for role in cell_spec.get("node_roles", [])
        }
        node_labels = ["Node 0"]
        node_sublabels = [None]
        for node_idx in range(1, num_nodes):
            role = node_roles.get(node_idx, {})
            role_label = str(role.get("label") or "hidden")
            node_labels.append(f"Node {node_idx}")
            node_sublabels.append(
                role_label
                if role.get("sublabel") is None
                else f"{role_label}\n{role.get('sublabel')}"
            )
        node_labels.append("Cell out")
        node_sublabels.append(None)
        n_positions = len(node_labels)
        usable_w = cell_w - (0.92 if single_cell else 1.10)
        x0 = cell_x - cell_w / 2 + (0.46 if single_cell else 0.55)
        x_positions = [
            x0 + usable_w * (i / max(n_positions - 1, 1)) for i in range(n_positions)
        ]

        center_y = cell_y - 0.12
        spread = min(0.48, 0.20 * max(internal_nodes - 1, 0))
        y_positions = [center_y]
        for i in range(internal_nodes):
            if internal_nodes == 1:
                yi = center_y
            else:
                frac = i / max(internal_nodes - 1, 1)
                yi = center_y + spread * (0.5 - frac)
            y_positions.append(yi)
        y_positions.append(center_y)

        for node_idx in range(1, num_nodes):
            for input_idx in range(node_idx):
                x1, y1 = x_positions[input_idx], y_positions[input_idx]
                x2, y2 = x_positions[node_idx], y_positions[node_idx]
                rad = 0.06 * (node_idx - input_idx)
                ax.annotate(
                    "",
                    xy=(x2 - 0.10, y2),
                    xytext=(x1 + 0.10, y1),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="#D0D4DA",
                        lw=0.9,
                        mutation_scale=8,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=2,
                )

        edges_by_target: dict[int, list[dict[str, Any]]] = {}
        for edge_spec in cell_spec.get("selected_edges", []):
            edges_by_target.setdefault(int(edge_spec["target"]), []).append(edge_spec)

        selected_non_identity = [
            edge_spec
            for edge_spec in cell_spec.get("selected_edges", [])
            if edge_spec.get("operation") != "Identity"
        ]
        non_identity_ops_by_target = {
            target: {
                str(edge_spec["operation"])
                for edge_spec in target_edges
                if edge_spec.get("operation") != "Identity"
            }
            for target, target_edges in edges_by_target.items()
        }

        for target, target_edges in sorted(edges_by_target.items()):
            target_edges.sort(key=lambda edge_spec: edge_spec["source"])
            for edge_rank, edge_spec in enumerate(target_edges):
                src_idx = int(edge_spec["source"])
                dst_idx = int(edge_spec["target"])
                x1, y1 = x_positions[src_idx], y_positions[src_idx]
                x2, y2 = x_positions[dst_idx], y_positions[dst_idx]
                color = _edge_op_color(str(edge_spec["operation"]))
                lw = 1.2 + 1.8 * float(edge_spec.get("importance", 0.5))
                rad = 0.06 * (dst_idx - src_idx) + 0.03 * (
                    edge_rank - (len(target_edges) - 1) / 2
                )
                ax.annotate(
                    "",
                    xy=(x2 - 0.11, y2),
                    xytext=(x1 + 0.11, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=lw,
                        mutation_scale=10,
                        linestyle=(0, (3.0, 2.0))
                        if edge_spec["operation"] == "Identity"
                        else "-",
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=3,
                )

                role = node_roles.get(dst_idx, {})
                target_non_identity_ops = non_identity_ops_by_target.get(dst_idx, set())
                show_edge_label = edge_spec["operation"] != "Identity" and (
                    len(target_non_identity_ops) > 1
                    or str(role.get("label") or "") != str(edge_spec["operation"])
                )

                if show_edge_label:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx = x2 - x1
                    dy = y2 - y1
                    edge_len = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                    normal_x = -dy / edge_len
                    normal_y = dx / edge_len
                    curve_sign = 1.0 if rad >= 0 else -1.0
                    offset = 0.10 * (edge_rank - (len(target_edges) - 1) / 2)
                    label_lift = 0.18 + 0.45 * abs(rad)
                    mid_x += curve_sign * normal_x * label_lift
                    mid_y += curve_sign * normal_y * label_lift + offset
                    edge_label = str(edge_spec["operation"])
                    if edge_spec.get("op_weight") is not None:
                        edge_label = (
                            f"{edge_label}\nw={float(edge_spec['op_weight']):.3f}"
                        )
                    ax.text(
                        mid_x,
                        mid_y,
                        edge_label,
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=COLORS["text"],
                        linespacing=1.0,
                        bbox=dict(
                            fc="#FFFFFF",
                            ec=color,
                            boxstyle="round,pad=0.18",
                            lw=0.85,
                        ),
                        zorder=4,
                    )

        last_internal_idx = max(1, n_positions - 2)
        _arrow(
            ax,
            x_positions[last_internal_idx] + 0.12,
            y_positions[last_internal_idx],
            x_positions[-1] - 0.12,
            y_positions[-1],
            lw=1.35,
            color=COLORS["arrow"],
            zorder=3,
        )
        ax.plot(
            [x_positions[0] + 0.10, x_positions[-1] - 0.10],
            [y_positions[0] - 0.22, y_positions[-1] - 0.22],
            color="#C7CCD3",
            lw=0.85,
            linestyle=(0, (2.0, 1.5)),
            zorder=2,
        )
        ax.text(
            (x_positions[0] + x_positions[-1]) / 2,
            center_y - 0.36,
            "global residual",
            ha="center",
            va="center",
            fontsize=7.3,
            color=COLORS["dim_text"],
            zorder=3,
        )

        for node_x, node_y, node_label, node_sublabel in zip(
            x_positions,
            y_positions,
            node_labels,
            node_sublabels,
            strict=False,
        ):
            is_cell_out = node_label == "Cell out"
            is_input_node = node_label == "Node 0"
            is_terminal = is_input_node or is_cell_out
            ax.add_patch(
                Circle(
                    (node_x, node_y),
                    0.12 if is_terminal else 0.105,
                    facecolor="#FFFFFF" if is_terminal else "#F5F7FA",
                    edgecolor=COLORS["block_border"],
                    linewidth=1.0,
                    zorder=5,
                )
            )
            if is_terminal:
                terminal_sublabel = "input" if is_input_node else None
                label_text = (
                    node_label
                    if terminal_sublabel is None
                    else f"{node_label}\n{terminal_sublabel}"
                )
                ax.text(
                    node_x,
                    node_y + 0.18,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color=COLORS["text"],
                    linespacing=1.05,
                    zorder=6,
                )
            else:
                label_lines = [str(node_label)]
                if node_sublabel:
                    label_lines.append(str(node_sublabel))
                ax.text(
                    node_x,
                    node_y + 0.20,
                    "\n".join(label_lines),
                    ha="center",
                    va="bottom",
                    fontsize=6.7,
                    color=COLORS["text"],
                    linespacing=1.05,
                    bbox=dict(
                        fc="#FFFFFF",
                        ec="#D7DCE3",
                        boxstyle="round,pad=0.18",
                        lw=0.8,
                    ),
                    zorder=6,
                )

        ops_text = ", ".join(cell_spec.get("selected_ops", [])) or "n/a"
        ax.text(
            cell_x,
            cell_y + cell_h / 2 - (0.16 if single_cell else 0.38),
            f"selected primitives: {ops_text}",
            ha="center",
            va="top",
            fontsize=7.6,
            color=COLORS["dim_text"],
            zorder=4,
        )
        edge_summary = _selected_edge_summary(cell_spec.get("selected_edges", []))
        if edge_summary:
            ax.text(
                cell_x,
                cell_y - cell_h / 2 + 0.26,
                edge_summary,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=COLORS["text"],
                bbox=dict(
                    fc="#FFFFFF",
                    ec="#D7DCE3",
                    boxstyle="round,pad=0.22",
                    lw=0.8,
                ),
                zorder=6,
            )

        if not selected_non_identity:
            ax.text(
                cell_x,
                center_y + 0.48,
                "identity-dominant selection",
                ha="center",
                va="center",
                fontsize=7.8,
                color=COLORS["dim_text"],
                bbox=dict(
                    fc="#F7F8FA",
                    ec="#C7CCD3",
                    boxstyle="round,pad=0.20",
                    lw=0.8,
                ),
                zorder=4,
            )


def _summary_title(base: str, pairs: Iterable[tuple[str, str | None]]) -> str:
    parts = [f"{key}={value}" for key, value in pairs if value and value != "unknown"]
    if not parts:
        return base
    return f"{base}\n" + " | ".join(parts)


def _summary_lines(pairs: Iterable[tuple[str, str | None]]) -> list[str]:
    return [f"{key}: {value}" for key, value in pairs if value and value != "unknown"]


def _draw_side_legend(
    ax,
    x: float,
    y: float,
    *,
    heading: str,
    lines: Sequence[str],
    align: str,
):
    if not lines:
        return

    rows = [
        TextArea(
            heading,
            textprops=dict(
                color=COLORS["text"],
                fontsize=11.2,
                fontweight="bold",
            ),
        )
    ]

    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            row = HPacker(
                children=[
                    TextArea(
                        f"{key}:",
                        textprops=dict(
                            color=COLORS["text"],
                            fontsize=10.2,
                            fontweight="bold",
                        ),
                    ),
                    TextArea(
                        f" {value.strip()}",
                        textprops=dict(
                            color=COLORS["text"],
                            fontsize=10.2,
                        ),
                    ),
                ],
                align="baseline",
                pad=0,
                sep=2,
            )
        else:
            row = TextArea(
                line,
                textprops=dict(
                    color=COLORS["text"],
                    fontsize=10.2,
                ),
            )
        rows.append(row)

    panel = VPacker(children=rows, align="left", pad=0, sep=5)
    box_alignment = (1.0, 0.5) if align == "right" else (0.0, 0.5)
    annotation = AnnotationBbox(
        panel,
        (x, y),
        xycoords="data",
        box_alignment=box_alignment,
        frameon=True,
        bboxprops=dict(
            fc="#FAFAFB",
            ec=COLORS["block_border"],
            boxstyle="round,pad=0.62,rounding_size=0.34",
            lw=1.0,
        ),
        zorder=5,
    )
    ax.add_artist(annotation)


def _edge_op_color(op_name: str) -> str:
    if op_name == "Identity":
        return "#9AA0A6"
    if op_name in {"TimeConv", "TCN", "ConvMixer", "MultiScaleConv", "PyramidConv"}:
        return COLORS["accent"]
    if op_name in {"Fourier", "Wavelet"}:
        return "#6B8E23"
    return COLORS["text"]


_OP_TO_FAMILY = {
    op_name: family_name
    for family_name, ops in DEFAULT_OP_FAMILIES.items()
    for op_name in ops
}


def _family_display_name(family_name: str | None) -> str | None:
    if not family_name:
        return None
    mapping = {
        "conv": "Convolutional",
        "frequency": "Frequency",
        "attention": "Attention",
        "mlp": "MLP",
    }
    return mapping.get(family_name, str(family_name).title())


def _node_role_from_edges(
    target_idx: int,
    target_edges: Sequence[dict[str, Any]],
) -> tuple[str, str | None]:
    non_identity_ops = []
    for edge_spec in target_edges:
        op_name = str(edge_spec.get("operation", "Identity"))
        if op_name != "Identity" and op_name not in non_identity_ops:
            non_identity_ops.append(op_name)

    if not non_identity_ops:
        return f"Skip node {target_idx}", "identity"

    families = []
    for op_name in non_identity_ops:
        family_name = _family_display_name(_OP_TO_FAMILY.get(op_name))
        if family_name and family_name not in families:
            families.append(family_name)

    if len(non_identity_ops) == 1:
        sublabel = families[0] if families else None
        return non_identity_ops[0], sublabel

    if len(families) == 1:
        return f"{families[0]} node", ", ".join(non_identity_ops)

    return f"Mixed node {target_idx}", ", ".join(non_identity_ops)


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
    summary_lines = _summary_lines(
        [
            ("tok", tokenizer),
            ("attn", self_attn),
            ("pos", position),
            ("ffn", ffn),
        ],
    )
    return layers, "Encoder", int(spec.get("num_layers") or 1), summary_lines


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

    summary_lines = _summary_lines(
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

    return (
        layers,
        "Decoder",
        int(spec.get("num_layers") or 1),
        cross_label,
        summary_lines,
    )


def draw_selected_transformer_architecture(
    model: Any, title: str | None = None, figsize=None
):
    spec = extract_selected_transformer_spec(model)
    arch_mode = spec["arch_mode"]
    encoder_spec = spec.get("encoder")
    decoder_spec = spec.get("decoder")
    overall_title = title or f"Selected Transformer Architecture ({arch_mode})"

    if arch_mode == "encoder_only" and encoder_spec is not None:
        encoder_layers, encoder_title, encoder_repeats, encoder_summary_lines = (
            make_encoder_layers_from_spec(encoder_spec)
        )
        single_title = overall_title
        if encoder_summary_lines:
            single_title += "\n" + "\n".join(encoder_summary_lines)
        return draw_single_block(
            encoder_layers,
            title=single_title,
            num_repeats=encoder_repeats,
            input_label="History / endogenous series",
            bg_color=COLORS["model_bg"],
            figsize=figsize,
        )

    if arch_mode == "decoder_only" and decoder_spec is not None:
        decoder_layers, decoder_title, decoder_repeats, _, decoder_summary_lines = (
            make_decoder_layers_from_spec(
                decoder_spec,
                encoder_available=False,
            )
        )
        single_title = overall_title
        if decoder_summary_lines:
            single_title += "\n" + "\n".join(decoder_summary_lines)
        return draw_single_block(
            decoder_layers,
            title=single_title,
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
        encoder_layers, encoder_title, encoder_repeats, encoder_summary_lines = (
            make_encoder_layers_from_spec(encoder_spec)
        )
        (
            decoder_layers,
            decoder_title,
            decoder_repeats,
            cross_label,
            decoder_summary_lines,
        ) = make_decoder_layers_from_spec(
            decoder_spec,
            encoder_available=True,
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
            encoder_summary_lines=encoder_summary_lines,
            decoder_summary_lines=decoder_summary_lines,
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
    encoder_summary_lines: Sequence[str] | None = None,
    decoder_summary_lines: Sequence[str] | None = None,
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
    side_w = 3.15
    bw = max(enc.block_width(), dec.block_width())
    tw = 2 * bw + gap + 2.6 + 2 * side_w
    th = max(enc.total_height(), dec.total_height()) + 1.2
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

    enc_summary_x = cx_enc - enc.block_width() / 2 - 0.55
    dec_summary_x = cx_dec + dec.block_width() / 2 + 0.55
    enc_summary_y = (enc._y0 + enc._y1) / 2
    dec_summary_y = (dec._y0 + dec._y1) / 2

    _draw_side_legend(
        ax,
        enc_summary_x,
        enc_summary_y,
        heading=encoder_title,
        lines=encoder_summary_lines or [],
        align="right",
    )

    _draw_side_legend(
        ax,
        dec_summary_x,
        dec_summary_y,
        heading=decoder_title,
        lines=decoder_summary_lines or [],
        align="left",
    )

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
        th - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
        zorder=6,
    )

    plt.tight_layout(pad=0.45)
    return fig, ax


def draw_selected_darts_cell_topology(
    model: Any,
    title: str | None = None,
    figsize=None,
    cell_specs_override: list[dict[str, Any]] | None = None,
    log_text: str | None = None,
):
    cell_specs = cell_specs_override
    if cell_specs is None and log_text:
        cell_specs = extract_logged_cell_specs(model, log_text)
    if not cell_specs:
        cell_specs = extract_selected_cell_specs(model)

    fig, ax = plt.subplots(figsize=figsize or (9.5, 3.8))
    fig.patch.set_facecolor(COLORS["page_bg"])
    ax.set_facecolor(COLORS["page_bg"])
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    if not cell_specs:
        ax.text(
            4.75,
            1.9,
            "No DARTS cell topology available for this model.",
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["text"],
        )
        return fig, ax

    _draw_darts_cell_panel(
        ax,
        x=4.75,
        y=1.9,
        w=8.7,
        h=3.0,
        cell_specs=cell_specs,
        heading=title,
    )

    plt.tight_layout(pad=0.35)
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
