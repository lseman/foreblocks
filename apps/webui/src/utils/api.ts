
import type { CategoryMap, NodeTypeMap } from "../types/types";

export const getApiBase = (): string => {
  const fromVite = import.meta.env.VITE_API_BASE?.trim?.() || "";
  const fromWindow = typeof window !== "undefined" ? ((window as any).__API_BASE__ || "").trim() : "";
  const pick = fromVite || fromWindow;
  if (pick) return pick.replace(/\/+$/, "");
  if (typeof window !== "undefined") {
    // Vite proxies backend routes during development. Using the current origin
    // also works when the UI is opened through a LAN hostname or dev container.
    return window.location.origin;
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
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.toLowerCase().includes("application/json")) {
        lastErr = new Error(`Expected JSON but received ${contentType || "an unknown content type"} @ ${url}`);
        continue;
      }
      const data = await res.json();
      return { nodes: data.nodes || {}, categories: data.categories || {} };
    } catch (e) { lastErr = e; }
  }
  console.warn("Using default node definitions (could not fetch /nodes):", lastErr);
  return { nodes: {}, categories: {} };
}
