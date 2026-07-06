import type { Connection, NodeData, NodeTypeMap } from "../types/types";
import { sanitizeIdent } from "./ids";

/** Render a JS value (or stringified boolean/null) as a Python literal. */
function pyLiteral(v: any): string {
  // Normalize string booleans/null (regardless of case)
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "true") return "True";
    if (s === "false") return "False";
    if (s === "null" || s === "none") return "None";
    // otherwise treat as a normal string literal
    return JSON.stringify(v);
  }

  // null / undefined
  if (v === null || v === undefined) return "None";

  const t = typeof v;

  // booleans
  if (t === "boolean") return v ? "True" : "False";

  // numbers
  if (t === "number") {
    if (Number.isFinite(v)) return String(v);
    if (Number.isNaN(v)) return "float('nan')";
    return v > 0 ? "float('inf')" : "float('-inf')";
  }

  // arrays
  if (Array.isArray(v)) return `[${v.map(pyLiteral).join(", ")}]`;

  // plain objects -> Python dict { "k": v, ... }
  if (t === "object") {
    const entries = Object.entries(v).map(([k, val]) => `${JSON.stringify(k)}: ${pyLiteral(val)}`);
    return `{${entries.join(", ")}}`;
  }

  // fallback
  return "None";
}

export function pickBindings(
  bindSpec: any = {},
  ctx: {
    node: NodeData;
    nodeTypes: NodeTypeMap;
    connections: Connection[];
    id2node: Record<string, NodeData>;
    outVarByPort: Record<string, string>;
  }
): { args: string[]; kwargs: string[] } {
  const { node, connections, outVarByPort } = ctx;

  const resolveToken = (tok: any): string => {
    // Non-string literal provided directly in bind spec
    if (typeof tok !== "string") return pyLiteral(tok);

    // Map raw string tokens "true"/"false"/"null" to Python literals
    {
      const s = tok.trim().toLowerCase();
      if (s === "true") return "True";
      if (s === "false") return "False";
      if (s === "null" || s === "none") return "None";
    }

    // Wire from another node's output into this node's input
    if (tok.startsWith("@input:")) {
      const port = tok.slice("@input:".length);
      const src = connections.find(c => c.to === node.id && c.toPort === port);
      if (!src) return "None";
      return (
        outVarByPort[`${src.from}:${src.fromPort}`] ||
        outVarByPort[`${src.from}:*`] ||
        outVarByPort[src.from] ||
        "None"
      );
    }

    // Pull from this node's config (must become a Python literal)
    if (tok.startsWith("@config:")) {
      const key = tok.slice("@config:".length);
      const val = node.config?.[key];
      return pyLiteral(val);
    }

    // Reference an attribute of this node's Python variable
    if (tok.startsWith("@attr:")) {
      return `${sanitizeIdent(node.id)}.${sanitizeIdent(tok.slice("@attr:".length))}`;
    }

    // Reference this node's Python variable
    if (tok === "@self") return sanitizeIdent(node.id);

    // Otherwise treat as raw Python snippet or symbol name (advanced usage)
    return String(tok);
  };

  const args = (bindSpec.args || []).map(resolveToken);
  const kwargs = Object.entries(bindSpec.kwargs || {}).map(([k, v]) => `${k}=${resolveToken(v)}`);

  return { args, kwargs };
}
