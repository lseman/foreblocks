
import type {
  Connection,
  NodeData,
  NodeTypeMap,
  WorkflowPreflightResult,
} from "../types/types";

export function topoSort(nodes: NodeData[], connections: Connection[]): NodeData[] {
  const id2node: Record<string, NodeData> = Object.fromEntries(nodes.map(n => [n.id, n]));
  const indeg: Record<string, number> = Object.fromEntries(nodes.map(n => [n.id, 0]));
  const adj: Record<string, string[]> = Object.fromEntries(nodes.map(n => [n.id, [] as string[]]));
  for (const c of connections) {
    if (id2node[c.from] && id2node[c.to]) {
      indeg[c.to] = (indeg[c.to] ?? 0) + 1;
      adj[c.from].push(c.to);
    }
  }
  const q: string[] = [];
  for (const n of nodes) if (indeg[n.id] === 0) q.push(n.id);
  const order: NodeData[] = [];
  while (q.length) {
    const u = q.shift()!;
    order.push(id2node[u]);
    for (const v of adj[u]) {
      indeg[v] -= 1;
      if (indeg[v] === 0) q.push(v);
    }
  }
  const seen = new Set(order.map(n => n.id));
  for (const n of nodes) if (!seen.has(n.id)) order.push(n);
  return order;
}

export function validateWorkflow(
  nodes: NodeData[],
  connections: Connection[],
  nodeTypes: NodeTypeMap,
): WorkflowPreflightResult {
  const errors: WorkflowIssue[] = [];
  const warnings: WorkflowIssue[] = [];

  if (nodes.length === 0) {
    errors.push({
      code: "EMPTY_GRAPH",
      message: "Canvas is empty. Add at least one node before running.",
    });
    return { errors, warnings };
  }

  const nodeById = new Map<string, NodeData>(nodes.map((n) => [n.id, n]));
  const incoming = new Map<string, Connection[]>();
  const outgoing = new Map<string, Connection[]>();
  nodes.forEach((n) => {
    incoming.set(n.id, []);
    outgoing.set(n.id, []);
  });

  nodes.forEach((node) => {
    if (!nodeTypes[node.type]) {
      errors.push({
        code: "UNKNOWN_NODE_TYPE",
        nodeId: node.id,
        message: `Node "${node.id}" has unknown type "${node.type}".`,
      });
    }
  });

  const targetPortSeen = new Map<string, string>();
  connections.forEach((conn) => {
    const fromNode = nodeById.get(conn.from);
    const toNode = nodeById.get(conn.to);
    if (!fromNode || !toNode) {
      errors.push({
        code: "BROKEN_CONNECTION",
        connectionId: conn.id,
        message: `Connection "${conn.id}" references a missing node.`,
      });
      return;
    }

    incoming.get(toNode.id)!.push(conn);
    outgoing.get(fromNode.id)!.push(conn);

    const fromDef = nodeTypes[fromNode.type];
    const toDef = nodeTypes[toNode.type];
    if (fromDef && !(fromDef.outputs || []).includes(conn.fromPort)) {
      errors.push({
        code: "INVALID_SOURCE_PORT",
        nodeId: fromNode.id,
        connectionId: conn.id,
        message: `Connection "${conn.id}" uses unknown output "${conn.fromPort}" on node "${fromNode.id}".`,
      });
    }
    if (toDef && !(toDef.inputs || []).includes(conn.toPort)) {
      errors.push({
        code: "INVALID_TARGET_PORT",
        nodeId: toNode.id,
        connectionId: conn.id,
        message: `Connection "${conn.id}" uses unknown input "${conn.toPort}" on node "${toNode.id}".`,
      });
    }

    if (conn.from === conn.to) {
      errors.push({
        code: "SELF_CYCLE",
        nodeId: conn.from,
        connectionId: conn.id,
        message: `Connection "${conn.id}" creates a self-cycle on node "${conn.from}".`,
      });
    }

    const targetKey = `${conn.to}:${conn.toPort}`;
    if (targetPortSeen.has(targetKey)) {
      errors.push({
        code: "DUPLICATE_TARGET_PORT",
        nodeId: conn.to,
        connectionId: conn.id,
        message: `Input "${conn.toPort}" on node "${conn.to}" has multiple incoming connections.`,
      });
    } else {
      targetPortSeen.set(targetKey, conn.id);
    }
  });

  const indeg = new Map<string, number>(nodes.map((n) => [n.id, 0]));
  const adjacency = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  connections.forEach((conn) => {
    if (!nodeById.has(conn.from) || !nodeById.has(conn.to)) return;
    indeg.set(conn.to, (indeg.get(conn.to) || 0) + 1);
    adjacency.get(conn.from)!.push(conn.to);
  });

  const q: string[] = [];
  indeg.forEach((d, nodeId) => {
    if (d === 0) q.push(nodeId);
  });
  let visited = 0;
  while (q.length > 0) {
    const u = q.shift()!;
    visited += 1;
    for (const v of adjacency.get(u) || []) {
      const next = (indeg.get(v) || 0) - 1;
      indeg.set(v, next);
      if (next === 0) q.push(v);
    }
  }
  if (visited < nodes.length) {
    errors.push({
      code: "CYCLE_DETECTED",
      message: "Workflow contains a cycle. Remove looped connections before running.",
    });
  }

  const roots = nodes.filter((n) => (incoming.get(n.id)?.length || 0) === 0);
  if (roots.length === 0) {
    errors.push({
      code: "NO_ENTRY_NODE",
      message: "No entry node found (every node has an input).",
    });
  }

  const hasOutputNode = nodes.some((n) => n.type === "output");
  if (!hasOutputNode) {
    warnings.push({
      code: "NO_OUTPUT_NODE",
      message: 'No node with type "output" found. You may not see final artifacts.',
    });
  }

  nodes.forEach((node) => {
    const def = nodeTypes[node.type];
    if (def) {
      const definedInputs = def.inputs || [];
      if (definedInputs.length > 0) {
        const incomingPorts = new Set((incoming.get(node.id) || []).map((c) => c.toPort));
        const missing = definedInputs.filter((port) => !incomingPorts.has(port));
        if (missing.length > 0) {
          warnings.push({
            code: "MISSING_INPUTS",
            nodeId: node.id,
            message: `Node "${node.id}" is missing inputs: ${missing.join(", ")}.`,
          });
        }
      }
    }

    const inCount = incoming.get(node.id)?.length || 0;
    const outCount = outgoing.get(node.id)?.length || 0;
    if (inCount === 0 && outCount === 0) {
      warnings.push({
        code: "ISOLATED_NODE",
        nodeId: node.id,
        message: `Node "${node.id}" is isolated (no incoming or outgoing connections).`,
      });
    }
  });

  return { errors, warnings };
}
