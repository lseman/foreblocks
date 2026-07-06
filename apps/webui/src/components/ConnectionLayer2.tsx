// src/components/ConnectionLayer.tsx
import React, { useMemo } from "react";
import type { Connection, NodeData, NodeTypeMap } from "../types/types";
import { getPortCenter } from "../utils/ports";

type Pt = { x: number; y: number };
type EdgeStyle = "bezier" | "beveled";

export const ConnectionLayer: React.FC<{
    nodes: NodeData[];
    connections: Connection[];
    connecting: {
        nodeId: string;
        portType: "output";
        portName: string;
        startX: number;
        startY: number;
    } | null;
    mousePos: { x: number; y: number };
    nearestPort: {
        nodeId: string;
        portName: string;
        x: number;
        y: number;
    } | null;
    onRemove: (id: string) => void;
    nodeTypes: NodeTypeMap;

    /** Optional: choose cable style ("bezier" | "beveled"). Default: "bezier". */
    edgeStyle?: EdgeStyle;
    /** Optional: bezier curvature (0..1). Default: 0.45. */
    curvature?: number;
    /** Optional: beveled corner radius (px). Default: 12. */
    radius?: number;
}> = ({
    nodes,
    connections,
    connecting,
    mousePos,
    nearestPort,
    onRemove,
    nodeTypes,
    edgeStyle = "bezier",
    curvature = 0.45,
    radius = 12,
}) => {
    // ---- helpers -------------------------------------------------------------

    /** Smooth left→right Bézier with adjustable curvature (0..1). */
    const bezierPath = (p1: Pt, p2: Pt, k = 0.5): string => {
        const dx = p2.x - p1.x || 1; // avoid 0 to keep handles usable
        const c1 = { x: p1.x + Math.abs(dx) * k, y: p1.y };
        const c2 = { x: p2.x - Math.abs(dx) * k, y: p2.y };
        return `M ${p1.x} ${p1.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${p2.x} ${p2.y}`;
    };

    /**
     * Orthogonal "beveled" cable: horizontal → rounded corner → vertical → rounded corner → horizontal.
     * Uses two arcs to create soft elbows. Radius auto-clamped to avoid overshoot.
     */
    const beveledPath = (p1: Pt, p2: Pt, rIn = 12): string => {
        const midX = (p1.x + p2.x) / 2;
        const dx = Math.abs(p2.x - p1.x);
        const dy = Math.abs(p2.y - p1.y);
        const r = Math.max(0, Math.min(rIn, dx / 2, dy / 2));

        // First elbow (around midX near p1.y)
        const sweep = p2.y >= p1.y ? 1 : 0;

        const x1 = midX - r;
        const y1 = p1.y;
        const x2 = midX;
        const y2 = p1.y + (sweep ? r : -r);

        // Second elbow (around midX near p2.y)
        const x3 = midX;
        const y3 = p2.y - (sweep ? r : -r);
        const x4 = midX + r;
        const y4 = p2.y;

        return [
            `M ${p1.x} ${p1.y}`,
            `L ${x1} ${y1}`,
            `A ${r} ${r} 0 0 ${sweep} ${x2} ${y2}`,
            `L ${x3} ${y3}`,
            `A ${r} ${r} 0 0 ${sweep} ${x4} ${y4}`,
            `L ${p2.x} ${p2.y}`,
        ].join(" ");
    };

    const buildPath = (p1: Pt, p2: Pt): string => {
        if (edgeStyle === "beveled") return beveledPath(p1, p2, radius);
        return bezierPath(p1, p2, curvature);
    };

    // -------------------------------------------------------------------------

    const nodeById = useMemo(
        () => Object.fromEntries(nodes.map((n) => [n.id, n])),
        [nodes],
    );

    const paths: React.ReactNode[] = connections.map((conn) => {
        const fromNode = nodeById[conn.from];
        const toNode = nodeById[conn.to];
        if (!fromNode || !toNode) return null;

        const fromType = nodeTypes[fromNode.type] || {};
        const toType = nodeTypes[toNode.type] || {};

        const fromPortIndex = (fromType.outputs || []).indexOf(conn.fromPort);
        const toPortIndex = (toType.inputs || []).indexOf(conn.toPort);
        if (fromPortIndex < 0 || toPortIndex < 0) return null;

        // Compute exact port centers
        const p1 = getPortCenter(
            fromNode,
            "output",
            fromPortIndex,
            (fromType.inputs || []).length,
        );
        const p2 = getPortCenter(toNode, "input", toPortIndex, 0);

        const d = buildPath(p1, p2);

        return (
            <path
                key={conn.id}
                d={d}
                stroke="url(#connectionGradient)"
                strokeWidth="3"
                fill="none"
                filter="url(#glow)"
                strokeLinecap="round"
                className="hover:opacity-80 cursor-pointer transition-opacity pointer-events-auto"
                onClick={(e) => {
                    e.stopPropagation();
                    onRemove(conn.id);
                }}
            />
        );
    });

    if (connecting) {
        const targetX = nearestPort ? nearestPort.x : mousePos.x;
        const targetY = nearestPort ? nearestPort.y : mousePos.y;

        const p1 = { x: connecting.startX, y: connecting.startY };
        const p2 = { x: targetX, y: targetY };

        const d = buildPath(p1, p2);

        paths.push(
            <path
                key="active-connection"
                d={d}
                stroke={nearestPort ? "#10b981" : "url(#connectionGradient)"}
                strokeWidth="3"
                strokeDasharray={nearestPort ? "0" : "8,4"}
                fill="none"
                filter="url(#glow)"
                strokeLinecap="round"
                className="pointer-events-none"
            />,
        );
    }

    return (
        <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ overflow: "visible" }}
        >
            <defs>
                <linearGradient
                    id="connectionGradient"
                    x1="0%"
                    y1="0%"
                    x2="100%"
                >
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>

                {/* Soft outer glow */}
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>

                {/* Optional: arrowhead (uncomment markerEnd below to use) */}
                {/*
        <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6" />
        </marker>
        */}
            </defs>

            {paths}
        </svg>
    );
};
