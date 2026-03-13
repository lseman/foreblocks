import React, { useMemo } from "react";
import { AlertTriangle, CircleAlert, LocateFixed, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../store/store";
import type { WorkflowIssue } from "../types/types";

type IssueRow = {
    level: "error" | "warning";
    issue: WorkflowIssue;
};

export const PreflightPanel: React.FC = () => {
    const {
        preflightIssues,
        showPreflightPanel,
        setShowPreflightPanel,
        setSelectedNodeId,
        setSelectedNodes,
        setShowConfigPanel,
        requestFocusNode,
    } = useStore();

    const rows = useMemo<IssueRow[]>(
        () => [
            ...preflightIssues.errors.map((issue) => ({ level: "error" as const, issue })),
            ...preflightIssues.warnings.map((issue) => ({ level: "warning" as const, issue })),
        ],
        [preflightIssues],
    );

    const focusIssueNode = (nodeId: string) => {
        setSelectedNodeId(nodeId);
        setSelectedNodes(new Set([nodeId]));
        setShowConfigPanel(true);
        requestFocusNode(nodeId);
    };

    return (
        <AnimatePresence>
            {showPreflightPanel && rows.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, x: 28 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 28 }}
                    className="fixed top-5 right-5 z-[90] w-[420px] max-w-[calc(100vw-2rem)]"
                >
                    <div className="bg-neutral-900/90 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-[0_20px_60px_rgba(0,0,0,0.55)] overflow-hidden">
                        <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <AlertTriangle size={14} className="text-amber-400" />
                                <span className="text-xs font-bold uppercase tracking-widest text-neutral-200">
                                    Preflight Issues
                                </span>
                            </div>
                            <div className="flex items-center gap-3">
                                <span className="text-[10px] text-rose-300">
                                    {preflightIssues.errors.length} error{preflightIssues.errors.length === 1 ? "" : "s"}
                                </span>
                                <span className="text-[10px] text-amber-300">
                                    {preflightIssues.warnings.length} warning{preflightIssues.warnings.length === 1 ? "" : "s"}
                                </span>
                                <button
                                    type="button"
                                    onClick={() => setShowPreflightPanel(false)}
                                    className="p-1 rounded hover:bg-white/10 text-neutral-400 hover:text-neutral-100 transition-colors"
                                    aria-label="Close preflight panel"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        </div>

                        <div className="max-h-[60vh] overflow-y-auto p-3 space-y-2">
                            {rows.map((row, index) => (
                                <div
                                    key={`${row.level}-${row.issue.code}-${row.issue.nodeId || "none"}-${row.issue.connectionId || "none"}-${index}`}
                                    className={`
                                        p-3 rounded-xl border text-xs
                                        ${row.level === "error"
                                            ? "bg-rose-500/10 border-rose-500/30 text-rose-100"
                                            : "bg-amber-500/10 border-amber-500/30 text-amber-100"}
                                    `}
                                >
                                    <div className="flex items-start gap-2">
                                        <CircleAlert
                                            size={14}
                                            className={row.level === "error" ? "text-rose-300 mt-0.5" : "text-amber-300 mt-0.5"}
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="text-[10px] uppercase tracking-wider opacity-80 mb-1">
                                                {row.level} · {row.issue.code}
                                            </div>
                                            <div className="leading-relaxed break-words">{row.issue.message}</div>
                                            {row.issue.nodeId && (
                                                <button
                                                    type="button"
                                                    onClick={() => focusIssueNode(row.issue.nodeId!)}
                                                    className="mt-2 inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide bg-black/30 hover:bg-black/45 border border-white/15 transition-colors"
                                                >
                                                    <LocateFixed size={12} />
                                                    Focus {row.issue.nodeId}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
