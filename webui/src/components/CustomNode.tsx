import { Trash2, Activity, Box, Database, Cpu, Layers, Play, CheckCircle2, AlertCircle, Terminal, Settings2, BarChart3, HelpCircle } from "lucide-react";
import React, { useMemo } from "react";
import {
    Handle,
    Position,
    type Edge as RFEdge,
    type Node as RFNode,
} from "reactflow";
import { motion } from "framer-motion";
import type {
    Connection,
    ExecutionResult,
    ExecutionResultsMap,
    ExecutionStateMap,
    NodeData,
    NodeTypeMap,
} from "../types/types";

/* ──────────────────────────────────────────────────────────────────────────
   Helpers for category icons
   ────────────────────────────────────────────────────────────────────────── */
const CategoryIcons: Record<string, React.ReactNode> = {
    'Data': <Database size={16} />,
    'Encoder': <Layers size={16} />,
    'Decoder': <Layers size={16} />,
    'Training': <Cpu size={16} />,
    'Evaluation': <BarChart3 size={16} />,
    'Output': <Terminal size={16} />,
    'Model': <Box size={16} />,
    'Utility': <Settings2 size={16} />,
    'Sampling': <Activity size={16} />,
};

const CategoryAccents: Record<string, { border: string; glow: string; chip: string }> = {
    Data: { border: "#0ea5e9", glow: "rgba(14,165,233,0.35)", chip: "rgba(14,165,233,0.16)" },
    Encoder: { border: "#8b5cf6", glow: "rgba(139,92,246,0.35)", chip: "rgba(139,92,246,0.18)" },
    Decoder: { border: "#a855f7", glow: "rgba(168,85,247,0.35)", chip: "rgba(168,85,247,0.18)" },
    Training: { border: "#22c55e", glow: "rgba(34,197,94,0.35)", chip: "rgba(34,197,94,0.18)" },
    Evaluation: { border: "#f59e0b", glow: "rgba(245,158,11,0.35)", chip: "rgba(245,158,11,0.18)" },
    Output: { border: "#f43f5e", glow: "rgba(244,63,94,0.35)", chip: "rgba(244,63,94,0.18)" },
    Model: { border: "#6366f1", glow: "rgba(99,102,241,0.35)", chip: "rgba(99,102,241,0.18)" },
    Utility: { border: "#64748b", glow: "rgba(100,116,139,0.35)", chip: "rgba(100,116,139,0.18)" },
    Sampling: { border: "#06b6d4", glow: "rgba(6,182,212,0.35)", chip: "rgba(6,182,212,0.18)" },
};

function getCategoryIcon(category?: string) {
    if (!category) return <HelpCircle size={16} />;
    return CategoryIcons[category] || <Box size={16} />;
}

/* ──────────────────────────────────────────────────────────────────────────
   Helpers for optional color override
   ────────────────────────────────────────────────────────────────────────── */
function colorClasses(c?: string) {
    if (!c) return "";
    if (
        /\bbg-/.test(c) ||
        /\bfrom-/.test(c) ||
        /\bto-/.test(c) ||
        /\bvia-/.test(c) ||
        /\bgradient\b/.test(c)
    ) {
        return c;
    }
    return "";
}
function colorStyle(c?: string) {
    if (!c) return undefined;
    if (/^#|^rgb|^hsl|^linear-gradient|^radial-gradient/i.test(c)) {
        return { background: c };
    }
    return undefined;
}

/* ──────────────────────────────────────────────────────────────────────────
   RF mapping
   ────────────────────────────────────────────────────────────────────────── */
export function toRFNodes(
    nodes: NodeData[],
    nodeTypes: NodeTypeMap,
    exec: ExecutionStateMap,
    results: ExecutionResultsMap,
): RFNode[] {
    return nodes.map((n) => ({
        id: n.id,
        type: "timeseriesNode",
        position: { x: n.position.x, y: n.position.y },
        parentNode: n.config?.composer_parent_id || undefined,
        extent: n.config?.composer_parent_id ? "parent" : undefined,
        data: {
            node: n,
            nodeTypes,
            execState: exec[n.id],
            result: results[n.id],
        },
        selectable: true,
    }));
}

export function toRFEdges(conns: Connection[]): RFEdge[] {
    return conns.map((c) => ({
        id: c.id,
        source: c.from,
        target: c.to,
        sourceHandle: c.fromPort,
        targetHandle: c.toPort,
        animated: false,
        type: "bezier",
        style: {
            strokeWidth: 2,
            stroke: "#334155", // slate-700
        },
    }));
}

/* ──────────────────────────────────────────────────────────────────────────
   Custom Node (Premium Architecture)
   ────────────────────────────────────────────────────────────────────────── */
type CardProps = {
    data: {
        node: NodeData;
        nodeTypes: NodeTypeMap;
        execState?: "running" | "success" | "error";
        result?: ExecutionResult;
        onDelete?: (id: string) => void;
        onOpenResult?: (id: string, result: ExecutionResult) => void;
        onDoubleOpenConfig?: (nodeId: string) => void;
    };
    selected: boolean;
};

export const HEADER_H = 48;
export const BODY_PAD_Y = 12;
export const ROW_BOX_H = 28;
export const ROW_GAP = 6; // Tailwind gap-1.5
export const ROW_H = ROW_BOX_H + ROW_GAP; // step between consecutive rows
export const COMPOSER_W = 420;
export const COMPOSER_H = 340;
const HANDLE_SIZE = 11;
const HANDLE_NUDGE_Y = 10;
const LABEL_H = 16; // Tailwind h-4
const LABEL_MB = 4; // Tailwind mb-1
const BODY_SECTION_GAP = 16; // Tailwind gap-4 between chip row and I/O grid
const CHIP_ROW_H = 24; // category/id badges row visual height
const FIRST_ROW_CENTER_OFFSET =
    CHIP_ROW_H +
    BODY_SECTION_GAP +
    LABEL_H +
    LABEL_MB +
    ROW_BOX_H / 2;

const TimeseriesRFNodeBase: React.FC<CardProps> = ({ data, selected }) => {
    const {
        node,
        nodeTypes,
        execState,
        result,
        onDelete,
        onDoubleOpenConfig,
    } = data;

    const def = nodeTypes[node.type] || {};
    const category = (def as any).category as string | undefined;
    const color = (def as any).color as string | undefined;
    const tailwindColor = colorClasses(color);
    const inlineColor = colorStyle(color);
    const accent = CategoryAccents[category || ""] || CategoryAccents.Utility;
    const inputs: string[] = def.inputs || [];
    const outputs: string[] = def.outputs || [];
    const isComposerContainer = node.type === "head_composer";

    const handleTop = (i: number) =>
        HEADER_H + BODY_PAD_Y + FIRST_ROW_CENTER_OFFSET + i * ROW_H + HANDLE_NUDGE_Y;

    const statusGlow = useMemo(() => {
        if (execState === "running") return "ring-2 ring-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.5)]";
        if (execState === "success") return "ring-2 ring-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.5)]";
        if (execState === "error") return "ring-2 ring-rose-500/50 shadow-[0_0_20px_rgba(244,63,94,0.5)]";
        return "";
    }, [execState]);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`
                relative rounded-2xl overflow-hidden backdrop-blur-xl transition-all duration-300
                ${selected ? "ring-2 ring-sky-400 scale-[1.015] shadow-2xl" : "shadow-xl border border-white/5"}
                ${statusGlow}
            `}
            style={{
                width: isComposerContainer ? COMPOSER_W : 292,
                minHeight: isComposerContainer ? COMPOSER_H : undefined,
                borderColor: selected ? accent.border : undefined,
                background: "linear-gradient(165deg, rgba(27,27,35,0.92) 0%, rgba(10,10,16,0.98) 100%)",
                boxShadow: selected
                    ? `0 18px 45px rgba(0,0,0,0.55), 0 0 0 1px ${accent.border}, 0 0 28px ${accent.glow}`
                    : "0 16px 35px rgba(0,0,0,0.45)",
            }}
            onDoubleClick={() => onDoubleOpenConfig?.(node.id)}
        >
            <div
                className="absolute top-0 left-0 right-0 h-[2px]"
                style={{ background: `linear-gradient(90deg, ${accent.border}, rgba(255,255,255,0.2))` }}
            />
            {/* AMBIENT GLOW */}
            {selected && (
                <div className="absolute -inset-2 blur-2xl pointer-events-none rounded-full" style={{ background: accent.glow }} />
            )}

            {/* HEADER */}
            <div
                className={`border-b border-white/10 relative z-10 flex items-center justify-between px-3
                    ${tailwindColor || "bg-gradient-to-br from-neutral-800 to-neutral-900"}
                `}
                style={{ height: HEADER_H, ...(inlineColor || {}) }}
            >
                <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-black/25 flex items-center justify-center text-white/90 shadow-inner border border-white/10">
                        {getCategoryIcon(category)}
                    </div>
                    <div>
                        <div className="font-bold text-[11px] uppercase tracking-[0.12em] text-white/90 truncate max-w-[148px]">
                            {def.name || node.type}
                        </div>
                        <div className="text-[10px] text-white/45 font-medium">
                            {category || 'Node'}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-1.5">
                    {execState === "running" && <Activity size={14} className="text-blue-400 animate-spin" />}
                    {execState === "success" && <CheckCircle2 size={14} className="text-emerald-400" />}
                    {execState === "error" && <AlertCircle size={14} className="text-rose-400" />}

                    <button
                        onClick={(e) => { e.stopPropagation(); onDelete?.(node.id); }}
                        className="w-7 h-7 rounded-lg hover:bg-white/10 flex items-center justify-center text-white/40 hover:text-white/90 transition-colors"
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>

            {/* BODY */}
            <div className="px-3 pb-4 pt-3 flex flex-col gap-4">
                <div className="flex items-center gap-2">
                    <span
                        className="px-2 py-1 rounded-md text-[10px] font-semibold tracking-wide uppercase text-white/90 border border-white/10"
                        style={{ background: accent.chip }}
                    >
                        {category || "node"}
                    </span>
                    <span className="px-2 py-1 rounded-md text-[10px] font-mono text-white/50 border border-white/10 bg-white/[0.03]">
                        {node.id}
                    </span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    {/* INPUTS COLUMN */}
                    <div className="flex flex-col gap-1.5">
                        <div className="h-4 leading-4 text-[9px] uppercase tracking-widest text-white/30 font-bold px-1 mb-1">Inputs</div>
                        {inputs.length === 0 ? (
                            <div className="text-[10px] text-white/10 italic px-1">Open</div>
                        ) : (
                            inputs.map((name) => (
                                <div key={name} className="h-7 flex items-center px-2 rounded-md bg-white/[0.04] border border-white/10 text-[11px] text-white/75">
                                    <span className="truncate">{name}</span>
                                </div>
                            ))
                        )}
                    </div>

                    {/* OUTPUTS COLUMN */}
                    <div className="flex flex-col gap-1.5 text-right">
                        <div className="h-4 leading-4 text-[9px] uppercase tracking-widest text-white/30 font-bold px-1 mb-1">Outputs</div>
                        {outputs.length === 0 ? (
                            <div className="text-[10px] text-white/10 italic px-1">End</div>
                        ) : (
                            outputs.map((name) => (
                                <div key={name} className="h-7 flex items-center justify-end px-2 rounded-md bg-white/[0.04] border border-white/10 text-[11px] text-white/75">
                                    <span className="truncate">{name}</span>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* NODE SUBTYPE TAG */}
                {node.subtype && (
                    <div className="px-2 py-1 rounded bg-sky-500/10 border border-sky-500/20 text-[10px] text-sky-300 font-mono inline-block self-start">
                        {node.subtype}
                    </div>
                )}

                {isComposerContainer && (
                    <div className="grid grid-cols-2 gap-2 -mt-1">
                        <div className="rounded-lg border border-cyan-400/20 bg-cyan-400/5 p-2 min-h-[86px]">
                            <div className="text-[9px] uppercase tracking-widest font-bold text-cyan-200/85 mb-1">Parallel</div>
                            <div className="text-[10px] text-cyan-100/55">Drop head blocks here</div>
                        </div>
                        <div className="rounded-lg border border-fuchsia-400/20 bg-fuchsia-400/5 p-2 min-h-[86px]">
                            <div className="text-[9px] uppercase tracking-widest font-bold text-fuchsia-200/85 mb-1">Serial</div>
                            <div className="text-[10px] text-fuchsia-100/55">Drop head blocks here</div>
                        </div>
                    </div>
                )}
            </div>

            {/* HANDLES */}
            {inputs.map((name, i) => (
                <Handle
                    key={`in-${name}`}
                    id={name}
                    type="target"
                    position={Position.Left}
                    style={{
                        width: HANDLE_SIZE, height: HANDLE_SIZE, left: -6,
                        top: handleTop(i),
                        background: accent.border,
                        border: "2px solid #0f172a",
                        boxShadow: `0 0 0 2px rgba(15,23,42,0.7), 0 0 10px ${accent.glow}`,
                    }}
                    className="hover:!w-4 hover:!h-4 transition-all"
                />
            ))}
            {outputs.map((name, i) => (
                <Handle
                    key={`out-${name}`}
                    id={name}
                    type="source"
                    position={Position.Right}
                    style={{
                        width: HANDLE_SIZE, height: HANDLE_SIZE, right: -6,
                        top: handleTop(i),
                        background: "#8b5cf6",
                        border: "2px solid #0f172a",
                        boxShadow: "0 0 0 2px rgba(15,23,42,0.7), 0 0 10px rgba(139,92,246,0.35)",
                    }}
                    className="hover:!w-4 hover:!h-4 transition-all"
                />
            ))}
        </motion.div>
    );
};

export const TimeseriesRFNode = React.memo(TimeseriesRFNodeBase);
export const rfNodeTypes = { timeseriesNode: TimeseriesRFNode };
