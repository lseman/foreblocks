// ─────────────────────────────────────────────────────────────────────────────
// Auto layout with Dagre (size cache for speed)
// ─────────────────────────────────────────────────────────────────────────────
import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef
} from "react";
import {
    Background,
    BackgroundVariant,
    Controls,
    MarkerType,
    MiniMap,
    ReactFlow,
    useEdgesState,
    useNodesState,
    useReactFlow,
    type Connection as RFConnection,
    type Edge as RFEdge,
    type Node as RFNode,
    type OnNodesChange,
    type OnEdgesChange
} from "reactflow";

// @ts-ignore
import dagre from "dagre";

import { Connection, ExecutionResult, NodeData, NodeTypeMap } from "@/types/types";
import { BODY_PAD_Y, HEADER_H, ROW_BOX_H, ROW_GAP } from "./CustomNode";
const sizeCache = new Map<string, { width: number; height: number }>();

function estimateNodeSize(n: NodeData, nodeTypes: NodeTypeMap) {
    const def = nodeTypes[n.type] || {};
    const key = `${n.type}:${(def.inputs || []).length}:${(def.outputs || []).length}`;
    const cached = sizeCache.get(key);
    if (cached) return cached;
    const inputs = (def.inputs || []).length;
    const outputs = (def.outputs || []).length;
    const rows = Math.max(inputs, outputs);
    const labelBlock = 16 + 4 + ROW_GAP; // h-4 + mb-1 + gap-1.5
    const rowsBlock =
        rows > 0
            ? rows * ROW_BOX_H + Math.max(0, rows - 1) * ROW_GAP
            : 14;
    const body = BODY_PAD_Y + labelBlock + rowsBlock + 16; // + pb-4
    const height = HEADER_H + Math.max(110 - HEADER_H, body);
    const v = { width: 292, height };
    sizeCache.set(key, v);
    return v;
}

export function layoutWithDagre(
    nodes: NodeData[],
    connections: Connection[],
    nodeTypes: NodeTypeMap,
    direction: "LR" | "TB" = "LR",
) {
    const g = new dagre.graphlib.Graph();
    g.setGraph({
        rankdir: direction,
        nodesep: 40,
        ranksep: 100,
        edgesep: 20,
        marginx: 40,
        marginy: 40,
        ranker: "network-simplex",
    });
    g.setDefaultEdgeLabel(() => ({}));

    nodes.forEach((n) => {
        const { width, height } = estimateNodeSize(n, nodeTypes);
        g.setNode(n.id, { width, height });
    });
    connections.forEach((c) => {
        g.setEdge(c.from, c.to);
    });

    dagre.layout(g);
    const posById: Record<string, { x: number; y: number }> = {};
    nodes.forEach((n) => {
        const p = g.node(n.id);
        posById[n.id] = {
            x: Math.round(p.x - (p.width ?? 0) / 2),
            y: Math.round(p.y - (p.height ?? 0) / 2),
        };
    });
    return posById;
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphCanvas (under ReactFlowProvider so we can use useReactFlow())
// ─────────────────────────────────────────────────────────────────────────────
export type GraphCanvasProps = React.PropsWithChildren<{
    flowNodes: RFNode[];
    flowEdges: RFEdge[];
    nodeTypesMap: Record<string, React.FC<any>>;
    handleNodesChange: OnNodesChange;
    onEdgesChange: OnEdgesChange;
    onConnect: (c: RFConnection) => void;
    onSelectionChange: (p: { nodes: RFNode[]; edges: RFEdge[] }) => void;
    contextMenuHandler: (e: React.MouseEvent) => void;
    deleteNode: (id: string) => void;
    openResult: (id: string, r: ExecutionResult) => void;
    openNodeConfig: (nodeId: string) => void;
    autoFitTick: number; // triggers fitView
    focusNodeId: string | null;
    focusNodeTick: number;
}>;

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
    flowNodes,
    flowEdges,
    nodeTypesMap,
    handleNodesChange,
    onEdgesChange,
    onConnect,
    onSelectionChange,
    contextMenuHandler,
    deleteNode,
    openResult,
    openNodeConfig,
    autoFitTick,
    focusNodeId,
    focusNodeTick,
    children
}) => {
    const { fitView, getViewport, setViewport, getNode, setCenter, getZoom } = useReactFlow();

    // Debounced, gentle fit
    const fitTimer = useRef<number | null>(null);
    const fitNicely = useCallback(() => {
        if (fitTimer.current) window.clearTimeout(fitTimer.current);
        fitTimer.current = window.setTimeout(() => {
            fitView({ padding: 0.2, includeHiddenNodes: true, duration: 300 });
            const vp = getViewport();
            setViewport({
                x: Math.round(vp.x),
                y: Math.round(vp.y),
                zoom: Math.min(
                    1,
                    Math.max(0.15, Math.round(vp.zoom * 10) / 10),
                ),
            });
            fitTimer.current = null;
        }, 80);
    }, [fitView, getViewport, setViewport]);

    useEffect(() => {
        fitNicely();
    }, [autoFitTick, fitNicely]);

    useEffect(() => {
        return () => {
            if (fitTimer.current) window.clearTimeout(fitTimer.current);
        };
    }, []);

    useEffect(() => {
        if (!focusNodeId) return;
        const n = getNode(focusNodeId);
        if (!n) return;
        const x = n.positionAbsolute?.x ?? n.position.x;
        const y = n.positionAbsolute?.y ?? n.position.y;
        const width = n.width ?? 280;
        const height = n.height ?? 180;
        setCenter(x + width / 2, y + height / 2, {
            zoom: Math.max(getZoom(), 0.75),
            duration: 250,
        });
    }, [focusNodeId, focusNodeTick, getNode, setCenter, getZoom]);

    // Stable callbacks injected into nodes (prevents child re-renders)
    const cbDelete = useCallback((id: string) => deleteNode(id), [deleteNode]);
    const cbOpenResult = useCallback(
        (id: string, r: ExecutionResult) => openResult(id, r),
        [openResult],
    );
    const cbOpenConfig = useCallback(
        (n: any) => openNodeConfig(typeof n === 'string' ? n : n.id),
        [openNodeConfig],
    );

    const mappedNodes = useMemo(
        () =>
            flowNodes.map((n) => ({
                ...n,
                data: {
                    ...(n.data || {}),
                    onDelete: cbDelete,
                    onOpenResult: cbOpenResult,
                    onDoubleOpenConfig: cbOpenConfig,
                },
            })),
        [flowNodes, cbDelete, cbOpenConfig, cbOpenResult],
    );

    return (
        <div className="absolute inset-0" onContextMenu={contextMenuHandler}>
            <ReactFlow
                nodes={mappedNodes}
                edges={flowEdges}
                nodeTypes={nodeTypesMap}
                onNodesChange={handleNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onSelectionChange={onSelectionChange}
                fitView
                panOnScroll
                selectionOnDrag
                defaultEdgeOptions={{
                    type: "smoothstep",
                    style: {
                        strokeWidth: 2.2,
                        stroke: "#64748b",
                    },
                    markerEnd: {
                        type: MarkerType.ArrowClosed,
                        width: 16,
                        height: 16,
                        color: "#64748b",
                    },
                }}
                deleteKeyCode={null}
                proOptions={{ hideAttribution: true }}
                defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            >
                {children}
                <Background
                    variant={BackgroundVariant.Lines}
                    gap={28}
                    size={0.8}
                    color="#1f2937"
                    className="opacity-45"
                />
                <Background
                    variant={BackgroundVariant.Dots}
                    gap={28}
                    size={1.8}
                    color="#334155"
                    className="opacity-55"
                />
                <MiniMap
                    pannable
                    zoomable
                    className="!bg-slate-900/92 !border-slate-600/50 rounded-xl shadow-2xl"
                    maskColor="rgba(15, 23, 42, 0.88)"
                    nodeColor="#7dd3fc"
                />
                <Controls className="!bg-slate-900/92 !border-slate-600/50 rounded-xl shadow-2xl" />
            </ReactFlow>
        </div>
    );
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: cycle check for connect & single-occupancy target port
// ─────────────────────────────────────────────────────────────────────────────
export function wouldCreateCycle(from: string, to: string, conns: Connection[]) {
    const adj = new Map<string, string[]>();
    for (const c of conns) {
        const arr = adj.get(c.from) || [];
        arr.push(c.to);
        adj.set(c.from, arr);
    }
    const stack = [to];
    const seen = new Set<string>();
    while (stack.length) {
        const u = stack.pop()!;
        if (u === from) return true;
        if (seen.has(u)) continue;
        seen.add(u);
        for (const v of adj.get(u) || []) stack.push(v);
    }
    return false;
}
