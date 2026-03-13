import type { Connection, NodeData } from "../types/types";

const STORAGE_WORKFLOWS_KEY = "foreblocks_webui_workflows_v1";
const STORAGE_CURRENT_WORKFLOW_KEY = "foreblocks_webui_current_workflow_v1";

export type SavedWorkflow = {
  id: string;
  name: string;
  description?: string;
  updatedAt: string;
  nodes: NodeData[];
  connections: Connection[];
};

export function createWorkflowId(): string {
  return `wf_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function createBlankWorkflow(name = "Untitled Workflow"): SavedWorkflow {
  return {
    id: createWorkflowId(),
    name,
    description: "New forecasting workflow",
    updatedAt: new Date().toISOString(),
    nodes: [],
    connections: [],
  };
}

export function loadSavedWorkflows(): SavedWorkflow[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_WORKFLOWS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item) => item && typeof item.id === "string" && typeof item.name === "string")
      .map((item) => ({
        id: String(item.id),
        name: String(item.name),
        description: item.description ? String(item.description) : "",
        updatedAt: item.updatedAt ? String(item.updatedAt) : new Date().toISOString(),
        nodes: Array.isArray(item.nodes) ? item.nodes : [],
        connections: Array.isArray(item.connections) ? item.connections : [],
      }))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  } catch {
    return [];
  }
}

export function saveSavedWorkflows(workflows: SavedWorkflow[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_WORKFLOWS_KEY, JSON.stringify(workflows));
  } catch {
    // ignore storage failures
  }
}

export function loadCurrentWorkflowId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(STORAGE_CURRENT_WORKFLOW_KEY);
  } catch {
    return null;
  }
}

export function saveCurrentWorkflowId(id: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_CURRENT_WORKFLOW_KEY, id);
  } catch {
    // ignore storage failures
  }
}

export function upsertWorkflow(
  workflows: SavedWorkflow[],
  workflow: SavedWorkflow,
): SavedWorkflow[] {
  const next = [
    workflow,
    ...workflows.filter((item) => item.id !== workflow.id),
  ];
  return next.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}
