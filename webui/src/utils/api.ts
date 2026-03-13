
import type { CategoryMap, NodeTypeMap } from "../types/types";

export const getApiBase = (): string => {
  const fromVite = (import.meta as any)?.env?.VITE_API_BASE?.trim?.() || "";
  const fromNext = typeof process !== "undefined" ? (process.env?.NEXT_PUBLIC_API_BASE || "").trim() : "";
  const fromWindow = typeof window !== "undefined" ? ((window as any).__API_BASE__ || "").trim() : "";
  const pick = fromVite || fromNext || fromWindow;
  if (pick) return pick.replace(/\/+$/, "");
  if (typeof window !== "undefined") {
    const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
    return isLocal ? "http://localhost:8000" : window.location.origin;
  }
  return "http://localhost:8000";
};

export const getExecuteCandidates = (): string[] => {
  const base = getApiBase();
  const sameOrigin = typeof window !== "undefined" && base === window.location.origin;
  return sameOrigin ? [`${base}/api/execute`, `${base}/execute`] : [`${base}/execute`, `${base}/api/execute`];
};

export async function fetchNodeDefs(): Promise<{ nodes: NodeTypeMap; categories: CategoryMap }> {
  const candidates = getExecuteCandidates();
  const urls = [
    candidates[0].replace(/\/execute$/, "") + "/nodes",
    candidates[1].replace(/\/execute$/, "") + "/nodes",
  ];
  let lastErr: any = null;
  for (const url of urls) {
    try {
      const res = await fetch(url, { credentials: "omit" });
      if (!res.ok) { lastErr = new Error(`HTTP ${res.status} @ ${url}`); continue; }
      const data = await res.json();
      return { nodes: data.nodes || {}, categories: data.categories || {} };
    } catch (e) { lastErr = e; }
  }
  console.warn("Using default node definitions (could not fetch /nodes):", lastErr);
  return { nodes: {}, categories: {} };
}
