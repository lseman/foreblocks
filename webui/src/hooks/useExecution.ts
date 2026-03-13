import { useCallback, useRef } from "react";
import { useStore } from "../store/store";
import { generateCode } from "../utils/codegen";
import { getExecuteCandidates } from "../utils/api";
import { validateWorkflow } from "../utils/graph";

function httpToWs(url: string) {
    if (url.startsWith("https://")) return url.replace(/^https:\/\//, "wss://");
    if (url.startsWith("http://")) return url.replace(/^http:\/\//, "ws://");
    return url;
}

function extractBaseApi(fromExecuteUrl: string) {
    // Accept both ".../api/execute" and ".../execute".
    return fromExecuteUrl.replace(/\/(?:api\/)?execute\/?$/, "");
}

export function useExecution() {
    const wsRef = useRef<{ ws: WebSocket; baseApi: string } | null>(null);
    const pollRef = useRef<number | null>(null);
    const {
        nodes,
        connections,
        nodeTypes,
        setIsExecuting,
        setProgressMsg,
        setExecutionState,
        updateExecutionState,
        setExecutionResults,
        updateExecutionResults,
        setExecutionLogs,
        appendExecutionLogs,
        setExecutionArtifacts,
        appendExecutionArtifact,
        clearExecutionTrace,
        setTaskId,
        setShowResultsPanel,
        setPreflightIssues,
        setShowPreflightPanel,
        isExecuting,
    } = useStore();

    const closeWs = useCallback(() => {
        if (wsRef.current) {
            try {
                wsRef.current.ws.close();
            } catch { }
            wsRef.current = null;
        }
        if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, []);

    const openStatusPolling = useCallback(
        (baseApi: string, tid: string) => {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = window.setInterval(async () => {
                try {
                    const urls = [`${baseApi}/api/status/${tid}`, `${baseApi}/status/${tid}`];
                    let s: any = null;
                    for (const url of urls) {
                        try {
                            const res = await fetch(url);
                            if (!res.ok) continue;
                            s = await res.json();
                            break;
                        } catch {
                            // try next candidate
                        }
                    }
                    if (!s) return;
                    setProgressMsg(s.message || "");
                    try {
                        const logsUrls = [`${baseApi}/api/logs/${tid}`, `${baseApi}/logs/${tid}`];
                        for (const url of logsUrls) {
                            try {
                                const logsRes = await fetch(url);
                                if (!logsRes.ok) continue;
                                const logsData = await logsRes.json();
                                if (Array.isArray(logsData.logs)) {
                                    setExecutionLogs(logsData.logs.map((line: unknown) => String(line)));
                                }
                                break;
                            } catch {
                                // try next candidate
                            }
                        }
                    } catch {
                        // ignore log polling failures
                    }
                    if (s.results) {
                        updateExecutionResults((prev) => ({ ...prev, ...s.results }));
                        Object.keys(s.results).forEach((nodeId) =>
                            updateExecutionState((prev) => ({ ...prev, [nodeId]: "success" }))
                        );
                        setShowResultsPanel(true);
                    }
                    if (["success", "error", "cancelled"].includes(s.state)) {
                        updateExecutionState((prev) => {
                            const next = { ...prev };
                            nodes.forEach((n) => {
                                if (next[n.id] === "running" || next[n.id] == null) {
                                    next[n.id] = s.state === "success" ? "success" : "error";
                                }
                            });
                            return next;
                        });
                        setIsExecuting(false);
                        setTaskId(null);
                        closeWs();
                    }
                } catch { }
            }, 1000);
        },
        [nodes, setProgressMsg, updateExecutionResults, updateExecutionState, setExecutionLogs, setShowResultsPanel, setIsExecuting, setTaskId, closeWs]
    );

    const openWebSocket = useCallback(
        (baseApi: string, tid: string) => {
            const root = extractBaseApi(baseApi);
            const wsUrl = httpToWs(`${root}/api/ws/${tid}`);
            try {
                const ws = new WebSocket(wsUrl);
                wsRef.current = { ws, baseApi };
                // Always-on safety polling: keeps UI progress/results in sync even
                // when websocket events are delayed/dropped by proxies.
                openStatusPolling(root, tid);
                let settled = false; // terminal message received or fallback already started

                const startFallbackPolling = () => {
                    // Polling is already active; mark settled to avoid duplicate fallbacks.
                    if (settled) return;
                    settled = true;
                    openStatusPolling(root, tid);
                };

                ws.onerror = startFallbackPolling;
                ws.onclose = startFallbackPolling;

                ws.onmessage = (ev) => {
                    const msg = JSON.parse(ev.data);
                    if (msg.event === "ping") {
                        try {
                            ws.send(JSON.stringify({ event: "pong" }));
                        } catch { }
                        return;
                    }
                        switch (msg.event) {
                        case "log":
                            if (msg.line) appendExecutionLogs([String(msg.line)]);
                            break;
                        case "logs":
                            if (Array.isArray(msg.lines)) {
                                appendExecutionLogs(msg.lines.map((line: unknown) => String(line)));
                            }
                            break;
                        case "artifact":
                            if (msg.name) appendExecutionArtifact(String(msg.name));
                            break;
                        case "snapshot":
                        case "state":
                            setProgressMsg(msg.message || "");
                            if (Array.isArray(msg.logs)) {
                                setExecutionLogs(msg.logs.map((line: unknown) => String(line)));
                            }
                            if (["success", "error", "cancelled"].includes(msg.state)) {
                                settled = true;
                                if (msg.results) {
                                    updateExecutionResults((prev) => ({ ...prev, ...msg.results }));
                                }
                                updateExecutionState((prev) => {
                                    const next = { ...prev };
                                    if (msg.results) {
                                        Object.keys(msg.results).forEach((nodeId) => {
                                            next[nodeId] = msg.state === "success" ? "success" : "error";
                                        });
                                    }
                                    nodes.forEach((n) => {
                                        if (next[n.id] === "running" || next[n.id] == null) {
                                            next[n.id] = msg.state === "success" ? "success" : "error";
                                        }
                                    });
                                    return next;
                                });
                                setIsExecuting(false);
                                setTaskId(null);
                                closeWs();
                            }
                            break;
                        case "node_result":
                            updateExecutionResults((prev) => ({ ...prev, [msg.node_id]: msg.result }));
                            updateExecutionState((prev) => ({ ...prev, [msg.node_id]: "success" }));
                            setShowResultsPanel(true);
                            break;
                        case "done":
                            settled = true;
                            if (msg.success && msg.results) {
                                updateExecutionResults((prev) => ({ ...prev, ...msg.results }));
                                updateExecutionState((prev) => {
                                    const next = { ...prev };
                                    Object.keys(msg.results).forEach((nodeId) => (next[nodeId] = "success"));
                                    nodes.forEach((n) => {
                                        if (next[n.id] === "running" || next[n.id] == null) next[n.id] = "success";
                                    });
                                    return next;
                                });
                            } else if (!msg.success) {
                                updateExecutionState((prev) => {
                                    const next = { ...prev };
                                    nodes.forEach((n) => {
                                        if (next[n.id] === "running" || next[n.id] == null) next[n.id] = "error";
                                    });
                                    return next;
                                });
                            }
                            setIsExecuting(false);
                            setTaskId(null);
                            closeWs();
                            break;
                    }
                };
            } catch {
                openStatusPolling(root, tid);
            }
        },
        [nodes, setProgressMsg, updateExecutionResults, updateExecutionState, setExecutionLogs, appendExecutionLogs, appendExecutionArtifact, setShowResultsPanel, setIsExecuting, setTaskId, closeWs, openStatusPolling]
    );

    const executeWorkflow = useCallback(async () => {
        closeWs();

        const preflight = validateWorkflow(nodes, connections, nodeTypes);
        setPreflightIssues(preflight);
        if (preflight.errors.length > 0) {
            const firstError = preflight.errors[0];

            setExecutionState({});
            setExecutionResults({});
            updateExecutionState(() => {
                const next: Record<string, "error"> = {};
                preflight.errors.forEach((issue) => {
                    if (issue.nodeId) next[issue.nodeId] = "error";
                });
                return next;
            });
            setTaskId(null);
            setIsExecuting(false);
            setProgressMsg(`Preflight failed: ${firstError.message}`);
            setShowPreflightPanel(true);
            return;
        }

        if (preflight.warnings.length > 0) {
            console.warn("Workflow preflight warnings:", preflight.warnings);
            setShowPreflightPanel(true);
        } else {
            setShowPreflightPanel(false);
        }

        setIsExecuting(true);
        setProgressMsg("Submitting...");
        clearExecutionTrace();
        setExecutionState({});
        setExecutionResults({});
        setExecutionLogs([]);
        setExecutionArtifacts([]);
        setTaskId(null);

        const code = generateCode(nodes, connections, nodeTypes);
        const payload = {
            code,
            workflow: {
                nodes: nodes.map((n) => ({ id: n.id, type: n.type, config: n.config })),
                connections,
            },
            sync: false,
            timeout_sec: 0,
        };

        const candidates = getExecuteCandidates();
        const tryFetch = async (url: string) => {
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), 15000);
            try {
                const sameOrigin = typeof window !== "undefined" && url.startsWith(window.location.origin);
                const res = await fetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    credentials: sameOrigin ? "same-origin" : "omit",
                    signal: controller.signal,
                });
                clearTimeout(t);
                return res;
            } catch (e) {
                clearTimeout(t);
                throw e;
            }
        };

        try {
            let response: Response | null = null;
            let chosenExecuteUrl: string | null = null;
            let lastErr: any = null;
            for (const url of candidates) {
                try {
                    response = await tryFetch(url);
                    if (response && response.ok) {
                        chosenExecuteUrl = url;
                        break;
                    } else if (response) {
                        lastErr = new Error(`HTTP ${response.status} on ${url}`);
                    }
                } catch (e) {
                    lastErr = e;
                }
            }
            if (!response || !chosenExecuteUrl) {
                if (lastErr) throw lastErr;
                throw new Error("No reachable API endpoint");
            }
            const result = await response.json();
            if ((result as any).success) {
                const tid: string = (result as any).task_id;
                setTaskId(tid);
                setProgressMsg("Running...");
                const baseApi = extractBaseApi(chosenExecuteUrl);
                openWebSocket(baseApi, tid);
                setShowResultsPanel(true);
            } else {
                throw new Error((result as any).error || "Execution failed to start");
            }
        } catch (error) {
            console.error("Execution error:", error);
            const message =
                error instanceof Error ? error.message : "Unknown execution error";
            setProgressMsg(`Execution failed: ${message}`);
            updateExecutionState(() => {
                const next: Record<string, "error"> = {};
                nodes.forEach((node) => {
                    next[node.id] = "error";
                });
                return next;
            });
            setTaskId(null);
            setIsExecuting(false);
        }
    }, [nodes, connections, nodeTypes, closeWs, setIsExecuting, setProgressMsg, setExecutionState, setExecutionResults, setExecutionLogs, setExecutionArtifacts, setTaskId, openWebSocket, setShowResultsPanel, updateExecutionState, updateExecutionResults, setPreflightIssues, setShowPreflightPanel, clearExecutionTrace]);

    return { executeWorkflow, isExecuting, closeWs };
}
