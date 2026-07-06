import { Settings, Trash2 } from "lucide-react";
import React from "react";
import type {
    Connection,
    ExecutionResult,
    NodeData,
    NodeTypeMap,
} from "../types/types";
import { tailwindClassToGradientStyle } from "../utils/ui";
import {
    HEADER_H,
    PORT_DIAMETER,
    PORT_OUTSET,
    PORT_ROW_H,
} from "../utils/ports";

export const NodeCard: React.FC<{
    node: NodeData;
    isSelected: boolean;
    execState: "running" | "success" | "error" | undefined;
    hasResult: ExecutionResult | undefined;
    onMouseDown: (e: React.MouseEvent) => void;
    onClick: (e: React.MouseEvent) => void;
    onDelete: (id: string) => void;
    onStartConnection: (
        e: React.MouseEvent,
        nodeId: string,
        portType: "output",
        portName: string,
    ) => void;
    onCompleteConnection?: (
        e: React.MouseEvent,
        nodeId: string,
        portType: "input" | "output",
        portName: string,
    ) => void;
    connections: Connection[];
    nodeTypes: NodeTypeMap;
}> = ({
    node,
    isSelected,
    execState,
    hasResult,
    onMouseDown,
    onClick,
    onDelete,
    onStartConnection,
    onCompleteConnection,
    connections,
    nodeTypes,
}) => {
    const nodeType = nodeTypes[node.type] || {};
    const inputs = nodeType.inputs || [];
    const outputs = nodeType.outputs || [];

    const headerStatus =
        execState === "running"
            ? "ring-2 ring-yellow-400 animate-pulse"
            : execState === "success"
              ? "ring-2 ring-green-400"
              : execState === "error"
                ? "ring-2 ring-red-400"
                : "";

    return (
        <div
            data-node-card
            className={`absolute backdrop-blur-sm ${
                isSelected
                    ? "ring-2 ring-blue-400 shadow-2xl shadow-blue-500/50"
                    : "shadow-xl"
            } ${headerStatus}`}
            style={{
                left: node.position.x,
                top: node.position.y,
                width: 260,
                cursor: "grab",
            }}
            onMouseDown={onMouseDown}
            onClick={onClick}
        >
            <div
                style={{
                    background: tailwindClassToGradientStyle(nodeType.color),
                }}
                className="rounded-xl overflow-hidden border border-white/10"
            >
                {/* HEADER (fixed height for stable port geometry) */}
                <div
                    className="border-b border-white/10 bg-black/20"
                    style={{ height: HEADER_H }}
                >
                    <div className="h-full px-3 flex items-center justify-between">
                        <div className="flex-1 min-w-0 flex items-center gap-2">
                            {execState === "running" && (
                                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin flex-shrink-0" />
                            )}
                            {execState === "success" && (
                                <div className="w-3 h-3 bg-green-400 rounded-full flex-shrink-0" />
                            )}
                            {execState === "error" && (
                                <div className="w-3 h-3 bg-red-400 rounded-full flex-shrink-0" />
                            )}
                            <div className="flex-1 min-w-0">
                                <div className="font-semibold text-sm truncate">
                                    {nodeType.name || node.type}
                                </div>
                                {node.subtype && (
                                    <div className="text-xs opacity-75 truncate">
                                        {node.subtype}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="flex gap-1">
                            {hasResult && (
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        window.dispatchEvent(
                                            new CustomEvent(
                                                "show-node-result",
                                                {
                                                    detail: {
                                                        nodeId: node.id,
                                                        result: hasResult,
                                                    },
                                                },
                                            ),
                                        );
                                    }}
                                    className="hover:bg-white/20 p-1.5 rounded-lg transition"
                                    title="View Results"
                                >
                                    <Settings size={14} />
                                </button>
                            )}

                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDelete(node.id);
                                }}
                                className="hover:bg-white/20 p-1.5 rounded-lg flex-shrink-0 transition"
                            >
                                <Trash2 size={14} />
                            </button>
                        </div>
                    </div>
                </div>

                {/* PORTS */}
                <div className="px-3 py-3 space-y-2 bg-black/10">
                    {/* INPUT PORT ROWS */}
                    {inputs.map((inputName) => {
                        const isConnected = connections.some(
                            (c) => c.to === node.id && c.toPort === inputName,
                        );

                        return (
                            <div
                                key={`in-${inputName}`}
                                className="relative flex items-center group"
                                style={{ height: PORT_ROW_H }}
                            >
                                {/* CIRCLE PORT */}
                                <button
                                    type="button"
                                    aria-label={`Input port ${inputName}`}
                                    className={`absolute top-1/2 -translate-y-1/2 rounded-full block transition-all ring-2 ring-white/50 pointer-events-auto
                    ${
                        isConnected
                            ? "bg-gradient-to-r from-blue-400 to-purple-400 scale-110 shadow-lg shadow-blue-500/50"
                            : "bg-slate-600 hover:bg-gradient-to-r hover:from-blue-400 hover:to-purple-400 hover:scale-110"
                    }`}
                                    style={{
                                        width: PORT_DIAMETER,
                                        height: PORT_DIAMETER,
                                        left: -PORT_OUTSET,
                                    }}
                                    onMouseUp={(e) =>
                                        onCompleteConnection?.(
                                            e,
                                            node.id,
                                            "input",
                                            inputName,
                                        )
                                    }
                                    onClick={(e) => e.stopPropagation()}
                                    data-port="input"
                                    data-port-name={inputName}
                                    title={`Input: ${inputName}`}
                                />

                                <span className="text-xs ml-1 group-hover:text-white transition-colors font-medium">
                                    {inputName}
                                </span>
                            </div>
                        );
                    })}

                    {/* OUTPUT PORT ROWS */}
                    {outputs.map((outputName) => {
                        const isConnected = connections.some(
                            (c) =>
                                c.from === node.id && c.fromPort === outputName,
                        );

                        return (
                            <div
                                key={`out-${outputName}`}
                                className="relative flex items-center justify-end group"
                                style={{ height: PORT_ROW_H }}
                            >
                                <span className="text-xs mr-1 group-hover:text-white transition-colors font-medium">
                                    {outputName}
                                </span>

                                {/* CIRCLE PORT */}
                                <button
                                    type="button"
                                    aria-label={`Output port ${outputName}`}
                                    className={`absolute top-1/2 -translate-y-1/2 rounded-full block cursor-crosshair transition-all ring-2 ring-white/50 pointer-events-auto
                    ${
                        isConnected
                            ? "bg-gradient-to-r from-blue-400 to-purple-400 scale-110 shadow-lg shadow-blue-500/50"
                            : "bg-slate-600 hover:bg-gradient-to-r hover:from-green-400 hover:to-emerald-400 hover:scale-125"
                    }`}
                                    style={{
                                        width: PORT_DIAMETER,
                                        height: PORT_DIAMETER,
                                        right: -PORT_OUTSET,
                                    }}
                                    onMouseDown={(e) =>
                                        onStartConnection(
                                            e,
                                            node.id,
                                            "output",
                                            outputName,
                                        )
                                    }
                                    onClick={(e) => e.stopPropagation()}
                                    data-port="output"
                                    data-port-name={outputName}
                                    title={`Output: ${outputName} (Click + Drag)`}
                                />
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
