import React from "react";
import { X } from "lucide-react";
import type { NodeData, NodeTypeMap } from "../types/types";

export const ConfigWindow: React.FC<{
    node: NodeData | null;
    rect: { x: number; y: number } | null;
    dragging: boolean;
    onStartDrag: (e: React.MouseEvent) => void;
    onClose: () => void;
    onChangeConfig: (nodeId: string, key: string, value: any) => void;
    onChangeSubtype: (nodeId: string, subtype: string) => void;
    nodeTypes: NodeTypeMap;
}> = ({
    node,
    rect,
    dragging,
    onStartDrag,
    onClose,
    onChangeConfig,
    onChangeSubtype,
    nodeTypes,
}) => {
    if (!node || !rect) return null;
    const def = nodeTypes[node.type] || {
        name: node.type,
        subtypes: [],
        color: "bg-slate-700/60",
    };

    return (
        <div
            data-config-window // 👈 marker so global hotkeys can detect focus inside config
            className="absolute bg-slate-900/95 backdrop-blur-xl rounded-xl shadow-2xl border border-slate-700/50 overflow-hidden"
            style={{
                left: rect.x,
                top: rect.y,
                width: 320,
                maxHeight: 600,
                zIndex: 1000,
            }}
            onClick={(e) => e.stopPropagation()}
            /* 👇 HARD STOP: keep keystrokes/scroll/mouse from reaching RF + window hotkeys */
            onKeyDownCapture={(e) => e.stopPropagation()}
            onKeyUpCapture={(e) => e.stopPropagation()}
            onKeyPressCapture={(e) => e.stopPropagation()}
            onWheelCapture={(e) => e.stopPropagation()}
            onMouseDownCapture={(e) => e.stopPropagation()}
            onMouseUpCapture={(e) => e.stopPropagation()}
        >
            <div
                className="p-3 border-b border-slate-700/50 bg-gradient-to-r from-slate-800 to-slate-700 cursor-move select-none"
                onMouseDown={onStartDrag}
                style={{ cursor: dragging ? "grabbing" : "grab" }}
            >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="flex gap-1">
                            <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                            <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
                        </div>
                        <h2 className="font-semibold text-sm ml-2">
                            Properties
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="hover:bg-white/10 p-1.5 rounded-lg transition"
                    >
                        <X size={16} />
                    </button>
                </div>
            </div>

            <div className="overflow-y-auto max-h-[520px] p-3 space-y-3">
                <div className="p-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50">
                    <label className="block text-xs font-medium mb-1.5 text-slate-400 uppercase tracking-wider">
                        Node ID
                    </label>
                    <input
                        type="text"
                        value={node.id}
                        disabled
                        className="w-full px-2.5 py-1.5 bg-slate-900/50 rounded-lg text-xs border border-slate-700/50 text-slate-400"
                    />
                </div>

                <div className="p-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50">
                    <label className="block text-xs font-medium mb-1.5 text-slate-400 uppercase tracking-wider">
                        Node Type
                    </label>
                    <div className="text-sm font-medium text-white">
                        {def.name}
                    </div>
                </div>

                {def.subtypes && def.subtypes.length > 0 && (
                    <div className="p-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <label className="block text-xs font-medium mb-1.5 text-slate-400 uppercase tracking-wider">
                            Subtype
                        </label>
                        <select
                            value={node.subtype || ""}
                            onChange={(e) =>
                                onChangeSubtype(
                                    node.id,
                                    (e.target as HTMLSelectElement).value,
                                )
                            }
                            className="w-full px-2.5 py-1.5 bg-slate-900/50 rounded-lg text-xs border border-slate-700/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                            onKeyDown={(e) => e.stopPropagation()} // extra safety inside inputs
                        >
                            {def.subtypes.map((st) => (
                                <option key={st} value={st}>
                                    {st}
                                </option>
                            ))}
                        </select>
                    </div>
                )}

                {Object.entries(node.config).map(([key, value]) => {
                    const isSel =
                        key === "forecasting_strategy" ||
                        key === "model_type" ||
                        key === "att_type" ||
                        key === "custom_norm" ||
                        key === "ci_aggregate" ||
                        key === "optimizer" ||
                        key === "criterion" ||
                        key === "combine" ||
                        key === "position";

                    return (
                        <div
                            key={key}
                            className="p-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50"
                        >
                            <label className="block text-xs font-medium mb-1.5 text-slate-400 uppercase tracking-wider">
                                {key.replace(/_/g, " ")}
                            </label>

                            {typeof value === "boolean" ? (
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={value}
                                        onChange={(e) =>
                                            onChangeConfig(
                                                node.id,
                                                key,
                                                (e.target as HTMLInputElement)
                                                    .checked,
                                            )
                                        }
                                        className="w-4 h-4 rounded bg-slate-900/50 border-slate-700/50"
                                        onKeyDown={(e) => e.stopPropagation()}
                                    />
                                    <span className="text-xs font-medium">
                                        {value ? "Enabled" : "Disabled"}
                                    </span>
                                </label>
                            ) : typeof value === "number" ? (
                                <input
                                    type="number"
                                    value={value}
                                    onChange={(e) =>
                                        onChangeConfig(
                                            node.id,
                                            key,
                                            parseFloat(
                                                (e.target as HTMLInputElement)
                                                    .value,
                                            ) || 0,
                                        )
                                    }
                                    step={
                                        key.includes("ratio") ||
                                        key.includes("dropout")
                                            ? "0.1"
                                            : "1"
                                    }
                                    className="w-full px-2.5 py-1.5 bg-slate-900/50 rounded-lg text-xs border border-slate-700/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                    onKeyDown={(e) => e.stopPropagation()}
                                />
                            ) : isSel ? (
                                <select
                                    value={value as any}
                                    onChange={(e) =>
                                        onChangeConfig(
                                            node.id,
                                            key,
                                            (e.target as HTMLSelectElement)
                                                .value,
                                        )
                                    }
                                    className="w-full px-2.5 py-1.5 bg-slate-900/50 rounded-lg text-xs border border-slate-700/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                    onKeyDown={(e) => e.stopPropagation()}
                                >
                                    {key === "forecasting_strategy" &&
                                        [
                                            "seq2seq",
                                            "autoregressive",
                                            "direct",
                                            "transformer_seq2seq",
                                        ].map((o) => (
                                            <option key={o} value={o}>
                                                {o}
                                            </option>
                                        ))}
                                    {key === "model_type" &&
                                        [
                                            "lstm",
                                            "transformer",
                                            "informer-like",
                                        ].map((o) => (
                                            <option key={o} value={o}>
                                                {o}
                                            </option>
                                        ))}
                                    {key === "att_type" &&
                                        ["standard", "prob", "full"].map(
                                            (o) => (
                                                <option key={o} value={o}>
                                                    {o}
                                                </option>
                                            ),
                                        )}
                                    {key === "custom_norm" &&
                                        ["layer", "batch", "rms", "none"].map(
                                            (o) => (
                                                <option key={o} value={o}>
                                                    {o}
                                                </option>
                                            ),
                                        )}
                                    {key === "ci_aggregate" &&
                                        ["mean", "linear", "concat"].map(
                                            (o) => (
                                                <option key={o} value={o}>
                                                    {o}
                                                </option>
                                            ),
                                        )}
                                    {key === "optimizer" &&
                                        ["Adam", "SGD", "AdamW", "RMSprop"].map(
                                            (o) => (
                                                <option key={o} value={o}>
                                                    {o}
                                                </option>
                                            ),
                                        )}
                                    {key === "criterion" &&
                                        [
                                            "MSELoss",
                                            "L1Loss",
                                            "SmoothL1Loss",
                                            "HuberLoss",
                                        ].map((o) => (
                                            <option key={o} value={o}>
                                                {o}
                                            </option>
                                        ))}
                                    {key === "combine" &&
                                        ["auto", "invert", "add", "none"].map(
                                            (o) => (
                                                <option key={o} value={o}>
                                                    {o}
                                                </option>
                                            ),
                                        )}
                                    {key === "position" &&
                                        [
                                            "pre_encoder",
                                            "post_encoder",
                                            "post_decoder",
                                        ].map((o) => (
                                            <option key={o} value={o}>
                                                {o}
                                            </option>
                                        ))}
                                </select>
                            ) : (
                                <input
                                    type="text"
                                    value={(value ?? "") as any}
                                    onChange={(e) =>
                                        onChangeConfig(
                                            node.id,
                                            key,
                                            (e.target as HTMLInputElement)
                                                .value,
                                        )
                                    }
                                    className="w-full px-2.5 py-1.5 bg-slate-900/50 rounded-lg text-xs border border-slate-700/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                    onKeyDown={(e) => e.stopPropagation()}
                                />
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
