import { create } from "zustand";
import {
    NodeData,
    Connection,
    NodeTypeMap,
    CategoryMap,
    ExecutionStateMap,
    ExecutionResultsMap,
    ExecutionResult,
    WorkflowPreflightResult,
} from "../types/types";

interface GraphState {
    nodes: NodeData[];
    connections: Connection[];
}

const HISTORY_LIMIT = 200;

const samePos = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    a.x === b.x && a.y === b.y;

const shallowEqualRecord = (a: Record<string, any>, b: Record<string, any>) => {
    if (a === b) return true;
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) return false;
    for (const k of aKeys) {
        if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
        if (!Object.is(a[k], b[k])) return false;
    }
    return true;
};

const sameNode = (a: NodeData, b: NodeData) =>
    a.id === b.id &&
    a.type === b.type &&
    a.subtype === b.subtype &&
    samePos(a.position, b.position) &&
    shallowEqualRecord(a.config, b.config);

const sameConnection = (a: Connection, b: Connection) =>
    a.id === b.id &&
    a.from === b.from &&
    a.fromPort === b.fromPort &&
    a.to === b.to &&
    a.toPort === b.toPort;

const sameGraph = (
    nodesA: NodeData[],
    connsA: Connection[],
    nodesB: NodeData[],
    connsB: Connection[],
) => {
    if (nodesA.length !== nodesB.length || connsA.length !== connsB.length) return false;
    for (let i = 0; i < nodesA.length; i += 1) {
        if (!sameNode(nodesA[i], nodesB[i])) return false;
    }
    for (let i = 0; i < connsA.length; i += 1) {
        if (!sameConnection(connsA[i], connsB[i])) return false;
    }
    return true;
};

interface EditorState {
    // Graph State
    nodes: NodeData[];
    connections: Connection[];

    // History
    history: GraphState[];
    historyIndex: number;

    // Node Definitions
    nodeTypes: NodeTypeMap;
    nodeCategories: CategoryMap;

    // Execution State
    executionState: ExecutionStateMap;
    executionResults: ExecutionResultsMap;
    executionLogs: string[];
    executionArtifacts: string[];
    isExecuting: boolean;
    progressMsg: string;
    taskId: string | null;

    // UI State
    selectedNodeId: string | null;
    selectedNodes: Set<string>;
    showConfigPanel: boolean;
    showResultsPanel: boolean;
    showPreflightPanel: boolean;
    preflightIssues: WorkflowPreflightResult;
    focusNodeId: string | null;
    focusNodeTick: number;

    // Actions
    setNodeDefs: (nodes: NodeTypeMap, categories: CategoryMap) => void;
    setGraph: (nodes: NodeData[], connections: Connection[]) => void;
    pushHistory: (state: GraphState) => void;
    undo: () => void;
    redo: () => void;

    updateNodes: (fn: (ns: NodeData[]) => NodeData[]) => void;
    updateConnections: (fn: (cs: Connection[]) => Connection[]) => void;

    setSelectedNodeId: (id: string | null) => void;
    setSelectedNodes: (ids: Set<string>) => void;
    setShowConfigPanel: (show: boolean) => void;
    setShowResultsPanel: (show: boolean) => void;
    setShowPreflightPanel: (show: boolean) => void;
    setPreflightIssues: (issues: WorkflowPreflightResult) => void;
    clearPreflightIssues: () => void;
    requestFocusNode: (id: string) => void;

    setExecutionState: (state: ExecutionStateMap) => void;
    updateExecutionState: (fn: (prev: ExecutionStateMap) => ExecutionStateMap) => void;
    setExecutionResults: (results: ExecutionResultsMap) => void;
    updateExecutionResults: (fn: (prev: ExecutionResultsMap) => ExecutionResultsMap) => void;
    setExecutionLogs: (logs: string[]) => void;
    appendExecutionLogs: (logs: string[]) => void;
    setExecutionArtifacts: (artifacts: string[]) => void;
    appendExecutionArtifact: (artifact: string) => void;
    clearExecutionTrace: () => void;
    setIsExecuting: (isExecuting: boolean) => void;
    setProgressMsg: (msg: string) => void;
    setTaskId: (id: string | null) => void;
}

export const useStore = create<EditorState>((set, get) => ({
    nodes: [],
    connections: [],
    history: [{ nodes: [], connections: [] }],
    historyIndex: 0,

    nodeTypes: {},
    nodeCategories: {},

    executionState: {},
    executionResults: {},
    executionLogs: [],
    executionArtifacts: [],
    isExecuting: false,
    progressMsg: "",
    taskId: null,

    selectedNodeId: null,
    selectedNodes: new Set(),
    showConfigPanel: false,
    showResultsPanel: false,
    showPreflightPanel: false,
    preflightIssues: { errors: [], warnings: [] },
    focusNodeId: null,
    focusNodeTick: 0,

    setNodeDefs: (nodeTypes, nodeCategories) => set({ nodeTypes, nodeCategories }),

    setGraph: (nodes, connections) => {
        const currentNodes = get().nodes;
        const currentConnections = get().connections;
        if (sameGraph(currentNodes, currentConnections, nodes, connections)) return;
        get().pushHistory({ nodes, connections });
        set({ nodes, connections });
    },

    pushHistory: (state) => {
        const { history, historyIndex } = get();
        const nextHistory = history.slice(0, historyIndex + 1);
        nextHistory.push(state);
        const trimmedHistory =
            nextHistory.length > HISTORY_LIMIT
                ? nextHistory.slice(nextHistory.length - HISTORY_LIMIT)
                : nextHistory;
        set({ history: trimmedHistory, historyIndex: trimmedHistory.length - 1 });
    },

    undo: () => {
        const { history, historyIndex } = get();
        if (historyIndex > 0) {
            const prevState = history[historyIndex - 1];
            set({
                nodes: prevState.nodes,
                connections: prevState.connections,
                historyIndex: historyIndex - 1
            });
        }
    },

    redo: () => {
        const { history, historyIndex } = get();
        if (historyIndex < history.length - 1) {
            const nextState = history[historyIndex + 1];
            set({
                nodes: nextState.nodes,
                connections: nextState.connections,
                historyIndex: historyIndex + 1
            });
        }
    },

    updateNodes: (fn) => {
        const newNodes = fn(get().nodes);
        get().setGraph(newNodes, get().connections);
    },

    updateConnections: (fn) => {
        const newConnections = fn(get().connections);
        get().setGraph(get().nodes, newConnections);
    },

    setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),
    setSelectedNodes: (selectedNodes) => set({ selectedNodes }),
    setShowConfigPanel: (showConfigPanel) => set({ showConfigPanel }),
    setShowResultsPanel: (showResultsPanel) => set({ showResultsPanel }),
    setShowPreflightPanel: (showPreflightPanel) => set({ showPreflightPanel }),
    setPreflightIssues: (preflightIssues) => set({ preflightIssues }),
    clearPreflightIssues: () => set({ preflightIssues: { errors: [], warnings: [] } }),
    requestFocusNode: (id) => set((state) => ({
        focusNodeId: id,
        focusNodeTick: state.focusNodeTick + 1,
    })),

    setExecutionState: (executionState) => set({ executionState }),
    updateExecutionState: (fn) => set((state) => ({ executionState: fn(state.executionState) })),
    setExecutionResults: (executionResults) => set({ executionResults }),
    updateExecutionResults: (fn) => set((state) => ({ executionResults: fn(state.executionResults) })),
    setExecutionLogs: (executionLogs) => set({ executionLogs }),
    appendExecutionLogs: (logs) => set((state) => ({
        executionLogs: [...state.executionLogs, ...logs],
    })),
    setExecutionArtifacts: (executionArtifacts) => set({ executionArtifacts }),
    appendExecutionArtifact: (artifact) => set((state) => ({
        executionArtifacts: state.executionArtifacts.includes(artifact)
            ? state.executionArtifacts
            : [...state.executionArtifacts, artifact],
    })),
    clearExecutionTrace: () => set({
        executionLogs: [],
        executionArtifacts: [],
        executionResults: {},
        executionState: {},
    }),
    setIsExecuting: (isExecuting) => set({ isExecuting }),
    setProgressMsg: (progressMsg) => set({ progressMsg }),
    setTaskId: (taskId) => set({ taskId }),
}));
