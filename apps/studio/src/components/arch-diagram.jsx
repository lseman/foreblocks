import { useMemo } from "react";

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

const W = 340;
const CX = W / 2;
const BW = 268;          // regular block width
const CW = BW + 20;      // container width (10px extra each side)
const ARROW_H = 22;
const GAP = 6;
const CONT_PAD_T = 30;   // inner-container top clearance (title area)
const CONT_PAD_B = 12;   // inner-container bottom clearance

// --------------------------------------------------------------------------
// Label maps
// --------------------------------------------------------------------------

const ATTN_LABEL = {
    standard: "Self-Attention",
    linear: "Linear Attention",
    sype: "SyPE Attention",
    hybrid: "Hybrid Attention",
    kimi: "Kimi Attention",
    hybrid_kimi: "Hybrid Kimi",
    kimi_3to1: "Kimi 3:1",
    gated_delta: "Gated Delta",
    hybrid_gdn: "Hybrid GDN",
    gdn_3to1: "GDN 3:1",
};

const FFN_LABEL = {
    standard: "Feed Forward",
    swiglu: "SwiGLU FFN",
    moe: "MoE SwiGLU FFN",
};

const NORM_LABEL = {
    rms: "RMSNorm",
    layernorm: "LayerNorm",
    rmsnorm: "RMSNorm",
};

// --------------------------------------------------------------------------
// Tone → fill / stroke / text colours (all CSS vars)
// --------------------------------------------------------------------------

const TONES = {
    embed: ["var(--accent-soft)", "rgba(59,130,246,0.35)", "var(--accent-strong)"],
    attn: ["rgba(59,130,246,0.08)", "rgba(59,130,246,0.30)", "var(--accent-strong)"],
    ffn: ["var(--warm-soft)", "rgba(14,165,164,0.30)", "var(--warm)"],
    norm: ["var(--cool-soft)", "rgba(100,116,139,0.30)", "var(--cool)"],
    cross: ["var(--secondary-soft)", "rgba(99,102,241,0.30)", "var(--secondary)"],
    revin: ["var(--secondary-soft)", "rgba(99,102,241,0.30)", "var(--secondary)"],
    output: ["var(--warm-soft)", "rgba(14,165,164,0.30)", "var(--warm)"],
    context: ["var(--cool-soft)", "rgba(100,116,139,0.30)", "var(--cool)"],
};

// --------------------------------------------------------------------------
// Layout builder
// --------------------------------------------------------------------------

function buildDiagram(config) {
    const m = config.model;
    const p = config.prep;
    const family = m.family;
    const encoderFfnType = m.encFfnType ?? m.ffnType ?? "swiglu";
    const decoderFfnType = m.decFfnType ?? m.ffnType ?? "swiglu";
    const encoderFfDim = m.encFfDim ?? m.ffDim ?? m.hiddenSize * 4;
    const decoderFfDim = m.decFfDim ?? m.ffDim ?? m.hiddenSize * 4;
    const encoderNumExperts = m.encNumExperts ?? m.numExperts ?? 4;
    const decoderNumExperts = m.decNumExperts ?? m.numExperts ?? 4;

    let y = 14;
    let kn = 0;
    const K = () => kn++;

    const blocks = [];      // SVG specs, rendered on top
    const containers = [];  // SVG specs, rendered behind blocks

    // -- helpers --

    function bH(hasSub) { return hasSub ? 50 : 36; }

    function arrow(h = ARROW_H) {
        const y1 = y, y2 = y + h;
        blocks.push({ kind: "arrow", k: K(), x1: CX, y1, x2: CX, y2: y2 - 5 });
        y = y2;
    }

    function ioBlock(label, sub) {
        const h = bH(!!sub);
        blocks.push({ kind: "io", k: K(), x: CX - BW / 2, y, w: BW, h, label, sub });
        y += h;
    }

    function block(label, sub, tone) {
        const [fill, stroke, text] = TONES[tone] || ["var(--panel-elevated)", "var(--border-strong)", "var(--text)"];
        const h = bH(!!sub);
        blocks.push({ kind: "block", k: K(), x: CX - BW / 2, y, w: BW, h, label, sub, fill, stroke, text });
        y += h;
    }

    function gap(n = GAP) { y += n; }

    function beginSection(title, badge) {
        const startY = y;
        y += CONT_PAD_T;
        return { startY, title, badge };
    }

    function endSection(sect, variant = false) {
        const endY = y + CONT_PAD_B;
        let fill, stroke, tcolor;
        if (variant === "heads") {
            fill = "rgba(14,165,164,0.05)";
            stroke = "var(--warm)";
            tcolor = "var(--warm)";
        } else if (variant) {
            fill = "rgba(99,102,241,0.05)";
            stroke = "var(--secondary)";
            tcolor = "var(--secondary)";
        } else {
            fill = "rgba(59,130,246,0.05)";
            stroke = "var(--accent)";
            tcolor = "var(--accent)";
        }
        containers.push({
            k: containers.length,
            x: CX - CW / 2, y: sect.startY, w: CW, h: endY - sect.startY,
            title: sect.title.toUpperCase(), badge: sect.badge,
            fill, stroke, tcolor,
        });
        y = endY;
    }

    // ---------- common: input + head composer stack ----------
    ioBlock("Input sequence", `${p.windowSize} time steps`);
    arrow();

    const h = m.heads ?? {};
    const activeHeadBlocks = [
        h.useRevIN && { label: "RevINHead", sub: "serial · invertible norm", tone: "revin" },
        h.useDecomposition && { label: "DecompositionBlock", sub: `kernel = ${h.decompositionKernelSize ?? 25}`, tone: "norm" },
        h.useMultiScaleConv && { label: "MultiScaleConvHead", sub: `${h.multiScaleNumScales ?? 5} scales · pyramid`, tone: "ffn" },
    ].filter(Boolean);
    const headSect = beginSection("Head Stack", null);
    if (activeHeadBlocks.length === 0) {
        block("Identity Path", "no optional heads enabled", "context");
    } else {
        for (let hi = 0; hi < activeHeadBlocks.length; hi++) {
            if (hi > 0) gap();
            block(activeHeadBlocks[hi].label, activeHeadBlocks[hi].sub, activeHeadBlocks[hi].tone);
        }
        gap();
        block("HeadComposer", `serial · ${activeHeadBlocks.length} active head${activeHeadBlocks.length === 1 ? "" : "s"}`, "output");
    }
    endSection(headSect, "heads");
    arrow();

    // ---------- transformer ----------
    if (family === "transformer") {
        const attnLabel = ATTN_LABEL[m.attentionMode] || "Self-Attention";
        const encoderFfnLabel = FFN_LABEL[encoderFfnType] || "Feed Forward";
        const decoderFfnLabel = FFN_LABEL[decoderFfnType] || "Feed Forward";
        const normLabel = NORM_LABEL[m.normType] || "LayerNorm";
        const attnSub = `${m.nheads} heads · ${m.attentionBackend || "torch"}`;
        const decAttnSub = `${m.decNheads ?? m.nheads} heads`;
        const encoderFfnSub = encoderFfnType === "moe"
            ? `${encoderNumExperts} experts · d_ff = ${encoderFfDim}`
            : `d_ff = ${encoderFfDim}`;
        const decoderFfnSub = decoderFfnType === "moe"
            ? `${decoderNumExperts} experts · d_ff = ${decoderFfDim}`
            : `d_ff = ${decoderFfDim}`;

        if (m.patchEncoder) {
            const nP = m.patchLen && m.patchStride
                ? Math.floor((p.windowSize - m.patchLen) / m.patchStride) + 1
                : null;
            block(
                "Patch Embedding",
                `L=${m.patchLen}, S=${m.patchStride}${nP ? ` → ${nP} patches` : ""}${m.channelIndep ? "  CI" : ""}`,
                "embed",
            );
        } else {
            block("Linear Projection", `d_model = ${m.hiddenSize}`, "embed");
        }

        arrow();

        const encSect = beginSection("Encoder", m.encLayers);
        block(attnLabel, attnSub, "attn");
        gap();
        block(encoderFfnLabel, encoderFfnSub, "ffn");
        gap();
        block(normLabel, null, "norm");
        endSection(encSect, false);

        arrow(28);

        const decSect = beginSection("Decoder", m.decLayers);
        block("Masked Self-Attention", decAttnSub, "attn");
        gap();
        block("Cross-Attention", "← encoder context", "cross");
        gap();
        block(decoderFfnLabel, decoderFfnSub, "ffn");
        gap();
        block(normLabel, null, "norm");
        endSection(decSect, true);

        arrow();
        block("Output Projection", `horizon = ${p.horizon}`, "output");
        arrow();
        ioBlock("Forecast output", `${p.horizon} steps ahead`);

        // ---------- recurrent ----------
    } else if (family === "lstm" || family === "gru") {
        const FU = family.toUpperCase();

        block("Input Projection", `d_model = ${m.hiddenSize}`, "embed");
        arrow();

        const encSect = beginSection(`${FU} Encoder`, m.encLayers);
        block(`${FU} recurrent cells`,
            `hidden = ${m.hiddenSize}${m.dropout > 0 ? ` · drop ${m.dropout}` : ""}`, "attn");
        endSection(encSect, false);

        arrow();
        block("Context vector", "hidden state", "context");
        arrow();

        const decSect = beginSection(`${FU} Decoder`, m.decLayers);
        block(`${FU} recurrent cells`, `hidden = ${m.hiddenSize}`, "attn");
        if (m.useAttentionModule) {
            gap();
            block("Attention Bridge", "external attention layer", "cross");
        }
        endSection(decSect, true);

        arrow();
        block("Output Projection", `horizon = ${p.horizon}`, "output");
        arrow();
        ioBlock("Forecast output", `${p.horizon} steps ahead`);

        // ---------- direct ----------
    } else {
        block("Flatten", "T × C → feature vector", "embed");
        arrow();

        const mlpSect = beginSection("MLP Head", null);
        block("Linear layers", `d = ${m.hiddenSize}`, "attn");
        gap();
        block("Activation", "ReLU", "ffn");
        if (m.dropout > 0) {
            gap();
            block("Dropout", `p = ${m.dropout}`, "norm");
        }
        endSection(mlpSect, false);

        arrow();
        block("Output Layer", `horizon = ${p.horizon}`, "output");
        arrow();
        ioBlock("Forecast output", `${p.horizon} steps ahead`);
    }

    y += 14; // bottom padding

    // ---------- render specs → React SVG nodes ----------

    const containerNodes = containers.map((c) => (
        <g key={`C${c.k}`}>
            <rect
                x={c.x} y={c.y} width={c.w} height={c.h}
                rx={12} fill={c.fill}
                stroke={c.stroke} strokeWidth={1.5} strokeDasharray="6 3"
            />
            <text
                x={c.x + 12} y={c.y + CONT_PAD_T / 2}
                dominantBaseline="middle"
                fontSize={9.5} fontWeight={700} fill={c.tcolor}
                letterSpacing="0.08em"
            >
                {c.title}
            </text>
            {c.badge != null && (
                <g>
                    <rect
                        x={c.x + c.w - 34} y={c.y + CONT_PAD_T / 2 - 10}
                        width={30} height={20} rx={10} fill={c.stroke}
                    />
                    <text
                        x={c.x + c.w - 19} y={c.y + CONT_PAD_T / 2}
                        textAnchor="middle" dominantBaseline="middle"
                        fontSize={11} fontWeight={700} fill="white"
                    >
                        ×{c.badge}
                    </text>
                </g>
            )}
        </g>
    ));

    const blockNodes = blocks.map((b) => {
        if (b.kind === "arrow") {
            return (
                <line
                    key={`A${b.k}`}
                    x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
                    stroke="var(--muted)" strokeWidth={1.5}
                    markerEnd="url(#arch-arrow)"
                />
            );
        }
        if (b.kind === "io") {
            return (
                <g key={`IO${b.k}`}>
                    <rect
                        x={b.x} y={b.y} width={b.w} height={b.h} rx={b.h / 2}
                        fill="var(--panel-elevated)" stroke="var(--border-strong)" strokeWidth={1.5}
                    />
                    <text
                        x={CX} y={b.y + (b.sub ? b.h * 0.36 : b.h / 2)}
                        textAnchor="middle" dominantBaseline="middle"
                        fontSize={12} fontWeight={700} fill="var(--text)"
                    >
                        {b.label}
                    </text>
                    {b.sub && (
                        <text
                            x={CX} y={b.y + b.h * 0.72}
                            textAnchor="middle" dominantBaseline="middle"
                            fontSize={9} fontStyle="italic" fill="var(--muted)"
                        >
                            {b.sub}
                        </text>
                    )}
                </g>
            );
        }
        return (
            <g key={`B${b.k}`}>
                <rect
                    x={b.x} y={b.y} width={b.w} height={b.h} rx={8}
                    fill={b.fill} stroke={b.stroke} strokeWidth={1.5}
                />
                <text
                    x={CX} y={b.y + (b.sub ? b.h * 0.36 : b.h / 2)}
                    textAnchor="middle" dominantBaseline="middle"
                    fontSize={12} fontWeight={600} fill={b.text}
                >
                    {b.label}
                </text>
                {b.sub && (
                    <text
                        x={CX} y={b.y + b.h * 0.72}
                        textAnchor="middle" dominantBaseline="middle"
                        fontSize={9} fontStyle="italic" fill="var(--subtext)"
                    >
                        {b.sub}
                    </text>
                )}
            </g>
        );
    });

    return { nodes: [...containerNodes, ...blockNodes], totalH: y };
}

// --------------------------------------------------------------------------
// Public component
// --------------------------------------------------------------------------

export function ArchDiagram({ config }) {
    const { nodes, totalH } = useMemo(() => buildDiagram(config), [config]);

    return (
        <div className="arch-diagram">
            <div className="arch-diagram-title">Architecture Preview</div>
            <svg
                viewBox={`0 0 ${W} ${totalH}`}
                width="100%"
                style={{ height: totalH, display: "block" }}
                xmlns="http://www.w3.org/2000/svg"
                strokeLinecap="round"
                strokeLinejoin="round"
            >
                <defs>
                    <marker
                        id="arch-arrow"
                        markerWidth="8" markerHeight="8"
                        refX="5" refY="4"
                        orient="auto-start-reverse"
                    >
                        <path
                            d="M1,1 L5,4 L1,7"
                            stroke="var(--muted)" strokeWidth="1.5" fill="none"
                            strokeLinecap="round" strokeLinejoin="round"
                        />
                    </marker>
                </defs>
                {nodes}
            </svg>
        </div>
    );
}


