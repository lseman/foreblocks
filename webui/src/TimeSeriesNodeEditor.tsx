import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import {
    ReactFlowProvider,
    addEdge,
    useEdgesState,
    useNodesState,
    type Connection as RFConnection,
    type Edge as RFEdge,
    type Node as RFNode
} from "reactflow";
import "reactflow/dist/style.css";

import { ContextMenu } from "./components/ContextMenu";
import { CodeModal } from "./components/modals/CodeModal";
import { ResultsModal } from "./components/modals/ResultsModal";
import { ShortcutsHelp } from "./components/ZoomHelp";
import { EditorToolbar } from "./components/EditorToolbar";
import { ResultsPanel } from "./components/ResultsPanel";
import { CommandPalette } from "./components/CommandPalette";
import { PreflightPanel } from "./components/PreflightPanel";
import { WorkspaceSidebar } from "./components/WorkspaceSidebar";
import { InspectorSidebar } from "./components/InspectorSidebar";

import { useStore } from "./store/store";
import { useExecution } from "./hooks/useExecution";

import type {
    CategoryMap,
    Connection,
    ExecutionResult,
    NodeData,
    NodeTypeMap,
} from "./types/types";
import { fetchNodeDefs } from "./utils/api";
import { COMPOSER_H, COMPOSER_W, rfNodeTypes, toRFEdges, toRFNodes } from "./components/CustomNode";
import { GraphCanvas, layoutWithDagre, wouldCreateCycle } from "./components/AutoAlign";
import { uid } from "./utils/ids";
import { generateCode } from "./utils/codegen";
import {
    createBlankWorkflow,
    createWorkflowId,
    loadCurrentWorkflowId,
    loadSavedWorkflows,
    saveCurrentWorkflowId,
    saveSavedWorkflows,
    type SavedWorkflow,
} from "./utils/workflows";

type HeadBlockStage = "serial" | "parallel";

type HeadBlockConfig = {
    head_node_id: string;
    name?: string;
    stage: HeadBlockStage;
    combine: "invert" | "add" | "none";
    add_project: boolean;
    alpha_mode: "off" | "gate" | "soft";
    alpha_mix_style: "blend" | "residual";
    alpha_init: number;
    alpha_trainable: boolean;
    weight_carry: boolean;
};

const RECENT_NODE_TYPES_KEY = "foreblocks_webui_recent_node_types_v1";
const SIDEBAR_COLLAPSED_KEY = "foreblocks_webui_sidebar_collapsed_v1";
const SIDEBAR_SECTION_KEY = "foreblocks_webui_sidebar_section_v1";
const LEFT_PANEL_WIDTH_KEY = "foreblocks_webui_left_panel_width_v1";
const RIGHT_PANEL_WIDTH_KEY = "foreblocks_webui_right_panel_width_v1";
const BOTTOM_PANEL_HEIGHT_KEY = "foreblocks_webui_bottom_panel_height_v1";
const MAX_RECENT_NODE_TYPES = 10;
const NODE_ID_PATTERN = /^node_(\d+)$/;
const DEFAULT_LEFT_PANEL_WIDTH = 308;
const DEFAULT_RIGHT_PANEL_WIDTH = 396;
const DEFAULT_BOTTOM_PANEL_HEIGHT = 280;
const MIN_LEFT_PANEL_WIDTH = 240;
const MAX_LEFT_PANEL_WIDTH = 520;
const MIN_RIGHT_PANEL_WIDTH = 300;
const MAX_RIGHT_PANEL_WIDTH = 560;
const MIN_BOTTOM_PANEL_HEIGHT = 180;
const MAX_BOTTOM_PANEL_HEIGHT = 460;

type SidebarSection = "workflows" | "recent" | "library";
type ResizeTarget = "left" | "right" | "bottom";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

export const TimeSeriesNodeEditor: React.FC = () => {
    // Canonical state from store
    const nodeIdCounter = useRef<number>(0);
    const {
        nodes,
        connections,
        nodeTypes,
        nodeCategories,
        executionState,
        executionResults,
        executionLogs,
        executionArtifacts,
        isExecuting,
        focusNodeId,
        focusNodeTick,
        setNodeDefs,
        setGraph,
        updateNodes,
        updateConnections,
        undo,
        redo,
        setSelectedNodeId,
        selectedNodes,
        setSelectedNodes,
        setShowConfigPanel,
    } = useStore();

    const { executeWorkflow } = useExecution();

    // UI state local to main editor
    const [searchQuery, setSearchQuery] = useState("");
    const [workflowName, setWorkflowName] = useState("Untitled Workflow");
    const [savedWorkflows, setSavedWorkflows] = useState<SavedWorkflow[]>([]);
    const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(null);
    const [recentNodeTypes, setRecentNodeTypes] = useState<string[]>([]);
    const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(true);
    const [leftSidebarSection, setLeftSidebarSection] = useState<SidebarSection>("library");
    const [leftPanelWidth, setLeftPanelWidth] = useState(DEFAULT_LEFT_PANEL_WIDTH);
    const [rightPanelWidth, setRightPanelWidth] = useState(DEFAULT_RIGHT_PANEL_WIDTH);
    const [bottomPanelHeight, setBottomPanelHeight] = useState(DEFAULT_BOTTOM_PANEL_HEIGHT);
    const [activeResizeTarget, setActiveResizeTarget] = useState<ResizeTarget | null>(null);
    const [contextMenu, setContextMenu] = useState<{
        x: number;
        y: number;
        screenX: number;
        screenY: number;
    } | null>(null);

    // Modals
    const [showCodeModal, setShowCodeModal] = useState(false);
    const [generatedCode, setGeneratedCode] = useState("");
    const [showResultModal, setShowResultModal] = useState<{
        nodeId: string;
        result: ExecutionResult;
    } | null>(null);

    // Fit trigger for RF
    const [autoFitTick, setAutoFitTick] = useState(0);
    const hydratedWorkflowRef = useRef(false);
    const shellRef = useRef<HTMLDivElement | null>(null);

    // Load node defs
    useEffect(() => {
        let cancel = false;
        (async () => {
            const defs = await fetchNodeDefs();
            if (!cancel) {
                setNodeDefs(defs.nodes, defs.categories);
            }
        })();
        return () => {
            cancel = true;
        };
    }, [setNodeDefs]);

    useEffect(() => {
        try {
            const raw = window.localStorage.getItem(RECENT_NODE_TYPES_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) {
                    setRecentNodeTypes(parsed.map((value) => String(value)).slice(0, MAX_RECENT_NODE_TYPES));
                }
            }
        } catch {
            // ignore storage failures
        }
    }, []);

    useEffect(() => {
        try {
            const collapsed = window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
            if (collapsed != null) {
                setLeftSidebarCollapsed(collapsed === "1");
            }
            const section = window.localStorage.getItem(SIDEBAR_SECTION_KEY);
            if (section === "workflows" || section === "recent" || section === "library") {
                setLeftSidebarSection(section);
            }
            const savedLeft = window.localStorage.getItem(LEFT_PANEL_WIDTH_KEY);
            if (savedLeft != null) {
                setLeftPanelWidth(clamp(Number(savedLeft) || DEFAULT_LEFT_PANEL_WIDTH, MIN_LEFT_PANEL_WIDTH, MAX_LEFT_PANEL_WIDTH));
            }
            const savedRight = window.localStorage.getItem(RIGHT_PANEL_WIDTH_KEY);
            if (savedRight != null) {
                setRightPanelWidth(clamp(Number(savedRight) || DEFAULT_RIGHT_PANEL_WIDTH, MIN_RIGHT_PANEL_WIDTH, MAX_RIGHT_PANEL_WIDTH));
            }
            const savedBottom = window.localStorage.getItem(BOTTOM_PANEL_HEIGHT_KEY);
            if (savedBottom != null) {
                setBottomPanelHeight(clamp(Number(savedBottom) || DEFAULT_BOTTOM_PANEL_HEIGHT, MIN_BOTTOM_PANEL_HEIGHT, MAX_BOTTOM_PANEL_HEIGHT));
            }
        } catch {
            // ignore storage failures
        }
    }, []);

    useEffect(() => {
        if (typeof window === "undefined") return;
        try {
            window.localStorage.setItem(
                RECENT_NODE_TYPES_KEY,
                JSON.stringify(recentNodeTypes.slice(0, MAX_RECENT_NODE_TYPES)),
            );
        } catch {
            // ignore storage failures
        }
    }, [recentNodeTypes]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        try {
            window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, leftSidebarCollapsed ? "1" : "0");
            window.localStorage.setItem(SIDEBAR_SECTION_KEY, leftSidebarSection);
            window.localStorage.setItem(LEFT_PANEL_WIDTH_KEY, String(leftPanelWidth));
            window.localStorage.setItem(RIGHT_PANEL_WIDTH_KEY, String(rightPanelWidth));
            window.localStorage.setItem(BOTTOM_PANEL_HEIGHT_KEY, String(bottomPanelHeight));
        } catch {
            // ignore storage failures
        }
    }, [bottomPanelHeight, leftPanelWidth, leftSidebarCollapsed, leftSidebarSection, rightPanelWidth]);

    useEffect(() => {
        if (hydratedWorkflowRef.current) return;
        const workflows = loadSavedWorkflows();
        const currentId = loadCurrentWorkflowId();
        const fallbackWorkflow =
            workflows.find((workflow) => workflow.id === currentId) ||
            workflows[0] ||
            createBlankWorkflow();
        const nextWorkflows =
            workflows.length > 0
                ? workflows
                : [fallbackWorkflow];

        setSavedWorkflows(nextWorkflows);
        setCurrentWorkflowId(fallbackWorkflow.id);
        setWorkflowName(fallbackWorkflow.name);
        setGraph(fallbackWorkflow.nodes, fallbackWorkflow.connections);
        nodeIdCounter.current = fallbackWorkflow.nodes.reduce((maxId, node) => {
            const match = NODE_ID_PATTERN.exec(node.id);
            return match ? Math.max(maxId, Number(match[1]) + 1) : maxId;
        }, 0);
        saveCurrentWorkflowId(fallbackWorkflow.id);
        if (workflows.length === 0) {
            saveSavedWorkflows(nextWorkflows);
        }
        hydratedWorkflowRef.current = true;
    }, [setGraph]);

    // Groups for menu
    const groupsForQuery = useMemo(() => {
        if (!searchQuery) return nodeCategories;
        const q = searchQuery.toLowerCase();
        const out: CategoryMap = {};
        Object.entries(nodeCategories).forEach(([cat, types]) => {
            const filtered = types.filter((t) => {
                const info = nodeTypes[t] || {};
                return (
                    (info.name || t).toLowerCase().includes(q) ||
                    t.toLowerCase().includes(q)
                );
            });
            if (filtered.length) out[cat] = filtered;
        });
        return out;
    }, [searchQuery, nodeCategories, nodeTypes]);

    const isHeadLikeNode = useCallback((n: NodeData) => {
        const outs = nodeTypes[n.type]?.outputs || [];
        return outs.some((o) => String(o).toLowerCase().includes("head"));
    }, [nodeTypes]);

    const toAbsPos = useCallback((n: NodeData, all: NodeData[]) => {
        const parentId = String(n.config?.composer_parent_id || "").trim();
        if (!parentId) return { ...n.position };
        const parent = all.find((x) => x.id === parentId);
        if (!parent) return { ...n.position };
        return {
            x: parent.position.x + n.position.x,
            y: parent.position.y + n.position.y,
        };
    }, []);

    const syncComposerBlocks = useCallback((allNodes: NodeData[]) => {
        return allNodes.map((n) => {
            if (n.type !== "head_composer") return n;

            const previous = Array.isArray(n.config?.head_blocks)
                ? (n.config.head_blocks as HeadBlockConfig[])
                : [];
            const prevMap = new Map(previous.map((b) => [b.head_node_id, b]));

            const children = allNodes
                .filter((c) => String(c.config?.composer_parent_id || "") === n.id && isHeadLikeNode(c))
                .sort((a, b) => (a.position.y - b.position.y) || (a.position.x - b.position.x));

            const nextBlocks: HeadBlockConfig[] = children.map((child, idx) => {
                const prev = prevMap.get(child.id);
                const stage = child.config?.composer_stage === "parallel" ? "parallel" : "serial";
                if (prev) {
                    return {
                        ...prev,
                        head_node_id: child.id,
                        stage,
                        name: prev.name || `${child.id}_${stage}_${idx + 1}`,
                    };
                }
                return {
                    head_node_id: child.id,
                    name: `${child.id}_${stage}_${idx + 1}`,
                    stage,
                    combine: "none",
                    add_project: true,
                    alpha_mode: "off",
                    alpha_mix_style: "blend",
                    alpha_init: 0,
                    alpha_trainable: true,
                    weight_carry: true,
                };
            });

            return {
                ...n,
                config: {
                    ...n.config,
                    head_blocks: nextBlocks,
                    composer_mode:
                        nextBlocks.some((b) => b.stage === "parallel")
                            ? (nextBlocks.some((b) => b.stage === "serial") ? "hybrid" : "parallel")
                            : "serial",
                },
            };
        });
    }, [isHeadLikeNode]);

    const persistWorkflow = useCallback((workflowId: string, nextName: string, nextNodes: NodeData[], nextConnections: Connection[]) => {
        const cleanedName = nextName.trim() || "Untitled Workflow";
        const updatedWorkflow: SavedWorkflow = {
            id: workflowId,
            name: cleanedName,
            description: `${nextNodes.length} nodes · ${nextConnections.length} connections`,
            updatedAt: new Date().toISOString(),
            nodes: nextNodes,
            connections: nextConnections,
        };
        setSavedWorkflows((prev) => {
            const next = [
                updatedWorkflow,
                ...prev.filter((item) => item.id !== workflowId),
            ].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
            saveSavedWorkflows(next);
            return next;
        });
        saveCurrentWorkflowId(workflowId);
        setCurrentWorkflowId(workflowId);
        setWorkflowName(cleanedName);
    }, []);

    const createNewWorkflow = useCallback(() => {
        const workflow = createBlankWorkflow(`Workflow ${savedWorkflows.length + 1}`);
        persistWorkflow(workflow.id, workflow.name, workflow.nodes, workflow.connections);
        setGraph([], []);
        nodeIdCounter.current = 0;
        setSelectedNodes(new Set());
        setSelectedNodeId(null);
        setShowConfigPanel(false);
    }, [persistWorkflow, savedWorkflows.length, setGraph, setSelectedNodeId, setSelectedNodes, setShowConfigPanel]);

    const loadWorkflow = useCallback((workflowId: string) => {
        const workflow = savedWorkflows.find((item) => item.id === workflowId);
        if (!workflow) return;
        setCurrentWorkflowId(workflow.id);
        setWorkflowName(workflow.name);
        setGraph(workflow.nodes, workflow.connections);
        nodeIdCounter.current = workflow.nodes.reduce((maxId, node) => {
            const match = NODE_ID_PATTERN.exec(node.id);
            return match ? Math.max(maxId, Number(match[1]) + 1) : maxId;
        }, 0);
        setSelectedNodes(new Set());
        setSelectedNodeId(null);
        setShowConfigPanel(false);
        saveCurrentWorkflowId(workflow.id);
    }, [savedWorkflows, setGraph, setSelectedNodeId, setSelectedNodes, setShowConfigPanel]);

    const deleteWorkflow = useCallback((workflowId: string) => {
        const remaining = savedWorkflows.filter((item) => item.id !== workflowId);
        if (remaining.length === 0) {
            const blank = createBlankWorkflow();
            setSavedWorkflows([blank]);
            saveSavedWorkflows([blank]);
            setCurrentWorkflowId(blank.id);
            setWorkflowName(blank.name);
            setGraph(blank.nodes, blank.connections);
            nodeIdCounter.current = 0;
            saveCurrentWorkflowId(blank.id);
            return;
        }
        setSavedWorkflows(remaining);
        saveSavedWorkflows(remaining);
        if (currentWorkflowId === workflowId) {
            const fallback = remaining[0];
            setCurrentWorkflowId(fallback.id);
            setWorkflowName(fallback.name);
            setGraph(fallback.nodes, fallback.connections);
            nodeIdCounter.current = fallback.nodes.reduce((maxId, node) => {
                const match = NODE_ID_PATTERN.exec(node.id);
                return match ? Math.max(maxId, Number(match[1]) + 1) : maxId;
            }, 0);
            saveCurrentWorkflowId(fallback.id);
        }
    }, [currentWorkflowId, savedWorkflows, setGraph]);

    const saveActiveWorkflow = useCallback(() => {
        const workflowId = currentWorkflowId || createWorkflowId();
        persistWorkflow(workflowId, workflowName, nodes, connections);
    }, [connections, currentWorkflowId, nodes, persistWorkflow, workflowName]);

    // Add node
    const addNode = useCallback(
        (type: string, position?: { x: number; y: number }) => {
            const def = nodeTypes[type];
            if (!def) return;
            const baseConfig = { ...(def.config || {}) } as any;
            if (type === "transformer_decoder") {
                baseConfig.informer_like = true;
            }
            if (type === "forecasting_model") {
                if (baseConfig.forecasting_strategy == null) {
                    baseConfig.forecasting_strategy = "seq2seq";
                }
                if (baseConfig.model_type == null) {
                    baseConfig.model_type = "informer-like";
                }
            }
            const newNode: NodeData = {
                id: `node_${nodeIdCounter.current++}`,
                type,
                position: position || { x: 100, y: 100 + nodes.length * 120 },
                config: baseConfig,
                subtype: def.subtypes?.length ? def.subtypes[0] : null,
            };
            setGraph([...nodes, newNode], connections);
            setContextMenu(null);
            setSearchQuery("");
            setRecentNodeTypes((prev) => [type, ...prev.filter((item) => item !== type)].slice(0, MAX_RECENT_NODE_TYPES));
        },
        [nodeTypes, nodes, connections, setGraph],
    );


    const deleteNode = useCallback(
        (nodeId: string) => {
            const parent = nodes.find((n) => n.id === nodeId);
            setGraph(
                nodes
                    .filter((n) => n.id !== nodeId)
                    .map((n) => {
                        const parentId = String(n.config?.composer_parent_id || "").trim();
                        if (!parent || parent.type !== "head_composer" || parentId !== nodeId) {
                            return n;
                        }
                        return {
                            ...n,
                            position: {
                                x: parent.position.x + n.position.x,
                                y: parent.position.y + n.position.y,
                            },
                            config: {
                                ...n.config,
                                composer_parent_id: null,
                                composer_stage: null,
                            },
                        };
                    }),
                connections.filter((c) => c.from !== nodeId && c.to !== nodeId),
            );
            setSelectedNodeId(null);
            setShowConfigPanel(false);
            setSelectedNodes(new Set([...selectedNodes].filter(id => id !== nodeId)));
        },
        [nodes, connections, selectedNodes, setGraph, setSelectedNodeId, setShowConfigPanel, setSelectedNodes],
    );

    // Context menu
    const handleContextMenu = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setContextMenu({
            x: e.nativeEvent.offsetX,
            y: e.nativeEvent.offsetY,
            screenX: e.clientX,
            screenY: e.clientY,
        });
    }, []);

    useEffect(() => {
        const close = () => setContextMenu(null);
        window.addEventListener("click", close);
        return () => window.removeEventListener("click", close);
    }, []);

    const onViewCode = useCallback(() => {
        setGeneratedCode(generateCode(nodes, connections, nodeTypes));
        setShowCodeModal(true);
    }, [nodes, connections, nodeTypes]);

    // ── React Flow bridge state ────────────────────────────────────────────────
    const rfNodes = useMemo(
        () => toRFNodes(nodes, nodeTypes, executionState, executionResults),
        [nodes, nodeTypes, executionState, executionResults],
    );
    const rfEdges = useMemo(() => toRFEdges(connections), [connections]);

    const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(rfNodes);
    const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState(rfEdges);

    useEffect(() => {
        setFlowNodes((prev) =>
            rfNodes.map((newNode) => ({
                ...newNode,
                selected: selectedNodes.has(newNode.id),
            })),
        );
    }, [rfNodes, selectedNodes, setFlowNodes]);

    useEffect(() => setFlowEdges(rfEdges), [rfEdges, setFlowEdges]);

    // Persist position changes back to canonical
    const handleNodesChange = useCallback(
        (changes: Parameters<typeof onNodesChange>[0]) => {
            onNodesChange(changes);
            const moved = new Map<string, { x: number; y: number }>();
            changes.forEach((c: any) => {
                // Persist all position changes; some drag flows don't emit a final non-dragging event.
                if (c.type === "position" && c.position)
                    moved.set(c.id, c.position);
            });
            if (moved.size > 0) {
                updateNodes((ns) => {
                    let next = ns.map((n) =>
                        moved.has(n.id)
                            ? {
                                ...n,
                                position: {
                                    x: moved.get(n.id)!.x,
                                    y: moved.get(n.id)!.y,
                                },
                            }
                            : n,
                    );

                    const composers = next.filter((n) => n.type === "head_composer");

                    next = next.map((n) => {
                        if (!moved.has(n.id) || n.type === "head_composer" || !isHeadLikeNode(n)) {
                            return n;
                        }

                        const parentId = String(n.config?.composer_parent_id || "").trim();
                        const abs = toAbsPos(n, next);

                        const hitComposer = composers.find((cp) => {
                            if (cp.id === n.id) return false;
                            const x0 = cp.position.x;
                            const y0 = cp.position.y;
                            return abs.x >= x0 && abs.x <= x0 + COMPOSER_W && abs.y >= y0 && abs.y <= y0 + COMPOSER_H;
                        });

                        if (hitComposer) {
                            const relX = Math.max(8, Math.min(COMPOSER_W - 120, abs.x - hitComposer.position.x));
                            const relY = Math.max(56, Math.min(COMPOSER_H - 60, abs.y - hitComposer.position.y));
                            const stage: HeadBlockStage = relX < COMPOSER_W * 0.5 ? "parallel" : "serial";
                            return {
                                ...n,
                                position: { x: relX, y: relY },
                                config: {
                                    ...n.config,
                                    composer_parent_id: hitComposer.id,
                                    composer_stage: stage,
                                },
                            };
                        }

                        if (parentId) {
                            return {
                                ...n,
                                position: { x: abs.x, y: abs.y },
                                config: {
                                    ...n.config,
                                    composer_parent_id: null,
                                    composer_stage: null,
                                },
                            };
                        }

                        return n;
                    });

                    return syncComposerBlocks(next);
                });
            }
        },
        [onNodesChange, updateNodes, isHeadLikeNode, toAbsPos, syncComposerBlocks],
    );

    // Connect edges
    const onConnect = useCallback(
        (conn: RFConnection) => {
            if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return;
            if (connections.some((c) => c.from === conn.source && c.to === conn.target && c.fromPort === conn.sourceHandle && c.toPort === conn.targetHandle)) return;
            if (connections.some((c) => c.to === conn.target && c.toPort === conn.targetHandle)) {
                console.warn("Target port already occupied");
                return;
            }
            if (wouldCreateCycle(conn.source, conn.target, connections)) {
                console.warn("Blocked edge that would create a cycle.");
                return;
            }
            const newConn: Connection = {
                id: uid("conn"),
                from: conn.source,
                to: conn.target,
                fromPort: conn.sourceHandle,
                toPort: conn.targetHandle,
            };
            updateConnections((cs) => [...cs, newConn]);
            setFlowEdges((eds) => addEdge(conn, eds));
        },
        [connections, updateConnections, setFlowEdges],
    );

    // Selection sync
    const onSelectionChange = useCallback(
        (params: { nodes: RFNode[]; edges: RFEdge[] }) => {
            const ids = new Set(params.nodes.map((n) => n.id));
            setSelectedNodes(ids);
            const firstId = params.nodes[0]?.id || null;
            setSelectedNodeId(firstId);
            if (firstId) setShowConfigPanel(true);
        },
        [setSelectedNodes, setSelectedNodeId, setShowConfigPanel],
    );

    const openNodeConfig = useCallback((nodeId: string) => {
        setSelectedNodeId(nodeId);
        setShowConfigPanel(true);
    }, [setSelectedNodeId, setShowConfigPanel]);

    // Keyboard shortcuts
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            const active = document.activeElement;
            const isEditing = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || (active as HTMLElement).isContentEditable);
            if (isEditing) return;

            if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
                e.preventDefault();
                undo();
            } else if ((e.ctrlKey || e.metaKey) && ((e.shiftKey && e.key === "z") || e.key === "y")) {
                e.preventDefault();
                redo();
            } else if (e.key === "Delete" || e.key === "Backspace") {
                if (selectedNodes.size) {
                    e.preventDefault();
                    setGraph(
                        nodes.filter((n) => !selectedNodes.has(n.id)),
                        connections.filter((c) => !selectedNodes.has(c.from) && !selectedNodes.has(c.to)),
                    );
                    setSelectedNodes(new Set());
                    setSelectedNodeId(null);
                    setShowConfigPanel(false);
                }
            } else if ((e.ctrlKey || e.metaKey) && e.key === "e") {
                e.preventDefault();
                executeWorkflow();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [undo, redo, nodes, connections, selectedNodes, setGraph, setSelectedNodes, setSelectedNodeId, setShowConfigPanel, executeWorkflow]);

    // Auto layout
    const autoLayout = useCallback(
        (dir: "LR" | "TB" = "LR") => {
            if (nodes.length === 0) return;
            const pos = layoutWithDagre(nodes, connections, nodeTypes, dir);
            const laidOut = nodes.map((n) => ({
                ...n,
                position: pos[n.id] ? { x: pos[n.id].x, y: pos[n.id].y } : n.position,
            }));
            setGraph(laidOut, [...connections]);
            setAutoFitTick((t) => t + 1);
        },
        [nodes, connections, nodeTypes, setGraph],
    );

    // Template loader
    const loadTemplate = useCallback((name: string) => {
        if (name === "basic_transformer") {
            const safe = (k: string) => ({ ...(nodeTypes[k]?.config || {}) });
            const n: NodeData[] = [
                { id: "data_1", type: "data_input", position: { x: 50, y: 50 }, config: safe("data_input"), subtype: null },
                { id: "enc_1", type: "transformer_encoder", position: { x: 350, y: 50 }, config: safe("transformer_encoder"), subtype: null },
                {
                    id: "dec_1",
                    type: "transformer_decoder",
                    position: { x: 350, y: 250 },
                    config: { ...safe("transformer_decoder"), informer_like: true },
                    subtype: null,
                },
                { id: "sched_1", type: "scheduled_sampling", position: { x: 350, y: 450 }, config: safe("scheduled_sampling"), subtype: null },
                {
                    id: "model_1",
                    type: "forecasting_model",
                    position: { x: 650, y: 150 },
                    config: {
                        ...safe("forecasting_model"),
                        forecasting_strategy: "seq2seq",
                        model_type: "informer-like",
                    },
                    subtype: null,
                },
                { id: "trainer_1", type: "trainer", position: { x: 950, y: 150 }, config: safe("trainer"), subtype: null },
                { id: "output_1", type: "output", position: { x: 1250, y: 150 }, config: safe("output"), subtype: null },
            ].filter((x) => nodeTypes[x.type]);
            const c: Connection[] = [
                { id: "c1", from: "data_1", fromPort: "X_train", to: "trainer_1", toPort: "X_train" },
                { id: "c2", from: "data_1", fromPort: "Y_train", to: "trainer_1", toPort: "Y_train" },
                { id: "c3", from: "enc_1", fromPort: "encoder", to: "model_1", toPort: "encoder" },
                { id: "c4", from: "dec_1", fromPort: "decoder", to: "model_1", toPort: "decoder" },
                { id: "c5", from: "sched_1", fromPort: "sampling_fn", to: "model_1", toPort: "scheduled_sampling_fn" },
                { id: "c6", from: "model_1", fromPort: "model", to: "trainer_1", toPort: "model" },
                { id: "c7", from: "trainer_1", fromPort: "trained_model", to: "output_1", toPort: "trained_model" },
            ].filter((cc) => n.find((nn) => nn.id === cc.from) && n.find((nn) => nn.id === cc.to));
            setGraph(n, c);
            nodeIdCounter.current = 10;
            setTimeout(() => autoLayout("LR"), 50);
        }
    }, [nodeTypes, setGraph, autoLayout]);

    useEffect(() => {
        if (!hydratedWorkflowRef.current || !currentWorkflowId) return;
        const timer = window.setTimeout(() => {
            persistWorkflow(currentWorkflowId, workflowName, nodes, connections);
        }, 500);
        return () => window.clearTimeout(timer);
    }, [connections, currentWorkflowId, nodes, persistWorkflow, workflowName]);

    const hasBottomPanelContent =
        isExecuting ||
        executionLogs.length > 0 ||
        executionArtifacts.length > 0 ||
        Object.keys(executionResults).length > 0;

    const startResize = useCallback((target: ResizeTarget) => {
        setActiveResizeTarget(target);
    }, []);

    useEffect(() => {
        if (!activeResizeTarget) return;

        const handleMove = (event: MouseEvent) => {
            const shellRect = shellRef.current?.getBoundingClientRect();
            const viewportWidth = shellRect?.width || window.innerWidth;
            const viewportHeight = shellRect?.height || window.innerHeight;
            const leftOffset = shellRect?.left || 0;
            const topOffset = shellRect?.top || 0;
            const pointerX = event.clientX - leftOffset;
            const pointerY = event.clientY - topOffset;
            const collapsedLeftWidth = 78;
            const reservedMainMinWidth = 520;

            if (activeResizeTarget === "left") {
                const maxLeft = Math.max(MIN_LEFT_PANEL_WIDTH, viewportWidth - rightPanelWidth - reservedMainMinWidth);
                setLeftPanelWidth(clamp(pointerX, MIN_LEFT_PANEL_WIDTH, Math.min(MAX_LEFT_PANEL_WIDTH, maxLeft)));
                return;
            }

            if (activeResizeTarget === "right") {
                const maxRight = Math.max(MIN_RIGHT_PANEL_WIDTH, viewportWidth - (leftSidebarCollapsed ? collapsedLeftWidth : leftPanelWidth) - reservedMainMinWidth);
                const nextRight = viewportWidth - pointerX;
                setRightPanelWidth(clamp(nextRight, MIN_RIGHT_PANEL_WIDTH, Math.min(MAX_RIGHT_PANEL_WIDTH, maxRight)));
                return;
            }

            const availableHeight = Math.max(MIN_BOTTOM_PANEL_HEIGHT, viewportHeight - 220);
            const nextBottom = viewportHeight - pointerY;
            setBottomPanelHeight(clamp(nextBottom, MIN_BOTTOM_PANEL_HEIGHT, Math.min(MAX_BOTTOM_PANEL_HEIGHT, availableHeight)));
        };

        const stopResize = () => {
            setActiveResizeTarget(null);
        };

        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", stopResize);
        document.body.style.userSelect = "none";
        document.body.style.cursor =
            activeResizeTarget === "bottom" ? "row-resize" : "col-resize";

        return () => {
            window.removeEventListener("mousemove", handleMove);
            window.removeEventListener("mouseup", stopResize);
            document.body.style.userSelect = "";
            document.body.style.cursor = "";
        };
    }, [activeResizeTarget, leftPanelWidth, leftSidebarCollapsed, rightPanelWidth]);

    return (
        <ReactFlowProvider>
            <div className="flex h-screen w-full flex-col overflow-hidden bg-[#060b12] text-white font-sans selection:bg-cyan-500/30">
                <EditorToolbar
                    workflowName={workflowName}
                    onWorkflowNameChange={setWorkflowName}
                    onViewCode={onViewCode}
                    loadTemplate={loadTemplate}
                    autoLayout={autoLayout}
                    onSaveWorkflow={saveActiveWorkflow}
                    onCreateWorkflow={createNewWorkflow}
                />

                <div
                    ref={shellRef}
                    className="grid min-h-0 flex-1 overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.08),transparent_26%),radial-gradient(circle_at_top_right,rgba(59,130,246,0.06),transparent_22%),linear-gradient(180deg,#08101a_0%,#060b12_100%)]"
                    style={{
                        gridTemplateColumns: `${leftSidebarCollapsed ? 78 : leftPanelWidth}px ${leftSidebarCollapsed ? 0 : 10}px minmax(0,1fr) 10px ${rightPanelWidth}px`,
                        gridTemplateRows: hasBottomPanelContent
                            ? `minmax(0,1fr) 10px ${bottomPanelHeight}px`
                            : "minmax(0,1fr) 0px 0px",
                    }}
                >
                    <div className="col-start-1 row-start-1 row-span-3 min-h-0">
                        <WorkspaceSidebar
                            nodeSearch={searchQuery}
                            setNodeSearch={setSearchQuery}
                            groups={groupsForQuery}
                            nodeTypes={nodeTypes}
                            onAddNode={addNode}
                            recentNodeTypes={recentNodeTypes}
                            workflows={savedWorkflows}
                            currentWorkflowId={currentWorkflowId}
                            onSelectWorkflow={loadWorkflow}
                            onCreateWorkflow={createNewWorkflow}
                            onDeleteWorkflow={deleteWorkflow}
                            collapsed={leftSidebarCollapsed}
                            activeSection={leftSidebarSection}
                            onToggleCollapsed={() => setLeftSidebarCollapsed((prev) => !prev)}
                            onSectionChange={setLeftSidebarSection}
                        />
                    </div>

                    {!leftSidebarCollapsed && (
                        <div
                            className={`col-start-2 row-start-1 row-span-3 min-h-0 cursor-col-resize bg-transparent transition hover:bg-cyan-400/10 ${activeResizeTarget === "left" ? "bg-cyan-400/16" : ""}`}
                            onMouseDown={() => startResize("left")}
                            title="Resize left panel"
                        />
                    )}

                    <div className="relative col-start-3 row-start-1 min-h-0 bg-[#070d15] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-center justify-between px-5 py-4">
                            <div className="pointer-events-auto rounded-[22px] bg-[#0c111b]/70 px-4 py-3 shadow-[0_10px_40px_rgba(2,8,23,0.28),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-xl">
                                <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                                    Canvas
                                </div>
                                <div className="mt-1 text-sm font-semibold text-white">
                                    Build forecasting graphs visually
                                </div>
                            </div>
                            <ShortcutsHelp />
                        </div>

                        <GraphCanvas
                            flowNodes={flowNodes}
                            flowEdges={flowEdges}
                            nodeTypesMap={rfNodeTypes}
                            handleNodesChange={handleNodesChange}
                            onEdgesChange={onEdgesChange}
                            onConnect={onConnect}
                            onSelectionChange={onSelectionChange}
                            contextMenuHandler={handleContextMenu}
                            deleteNode={deleteNode}
                            openResult={(id, r) => setShowResultModal({ nodeId: id, result: r })}
                            openNodeConfig={openNodeConfig}
                            autoFitTick={autoFitTick}
                            focusNodeId={focusNodeId}
                            focusNodeTick={focusNodeTick}
                        />

                        <ContextMenu
                            ctx={contextMenu}
                            groups={groupsForQuery}
                            onAdd={addNode}
                            searchQuery={searchQuery}
                            setSearchQuery={setSearchQuery}
                            onClose={() => setContextMenu(null)}
                            nodeTypes={nodeTypes}
                        />
                    </div>

                    <div
                        className={`col-start-4 row-start-1 row-span-3 min-h-0 cursor-col-resize bg-transparent transition hover:bg-cyan-400/10 ${activeResizeTarget === "right" ? "bg-cyan-400/16" : ""}`}
                        onMouseDown={() => startResize("right")}
                        title="Resize right panel"
                    />

                    <div className="col-start-5 row-start-1 row-span-3 min-h-0">
                        <InspectorSidebar workflowName={workflowName} />
                    </div>

                    {hasBottomPanelContent && (
                        <div
                            className={`col-start-3 row-start-2 min-h-0 cursor-row-resize bg-transparent transition hover:bg-cyan-400/10 ${activeResizeTarget === "bottom" ? "bg-cyan-400/16" : ""}`}
                            onMouseDown={() => startResize("bottom")}
                            title="Resize bottom panel"
                        />
                    )}

                    {hasBottomPanelContent && (
                        <div className="col-start-3 row-start-3 min-h-0 overflow-hidden bg-[#07101a]">
                            <ResultsPanel onShowResultModal={(id, r) => setShowResultModal({ nodeId: id, result: r })} />
                        </div>
                    )}
                </div>

                <PreflightPanel />
                <CommandPalette onAddNode={(type) => addNode(type)} />

                <CodeModal
                    open={showCodeModal}
                    code={generatedCode}
                    onClose={() => setShowCodeModal(false)}
                />
                <ResultsModal
                    modal={showResultModal}
                    onClose={() => setShowResultModal(null)}
                />
            </div>
        </ReactFlowProvider>
    );
};
