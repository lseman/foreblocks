import React, { useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  ChevronDown,
  ChevronUp,
  FileText,
  FolderKanban,
  Image as ImageIcon,
  TerminalSquare,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../store/store";
import type { ExecutionResult } from "../types/types";
import { getApiBase } from "../utils/api";

interface ResultsPanelProps {
  onShowResultModal: (nodeId: string, result: ExecutionResult) => void;
}

function formatResultBadge(type: string) {
  switch (type) {
    case "plot":
      return { label: "Plot", color: "bg-fuchsia-500/10 text-fuchsia-300" };
    case "metrics":
      return { label: "Metrics", color: "bg-amber-500/10 text-amber-300" };
    case "image":
      return { label: "Image", color: "bg-cyan-500/10 text-cyan-300" };
    case "table":
      return { label: "Table", color: "bg-slate-500/10 text-slate-200" };
    case "json":
      return { label: "JSON", color: "bg-slate-500/10 text-slate-200" };
    default:
      return { label: type.toUpperCase(), color: "bg-white/10 text-slate-200" };
  }
}

function renderMetricPreview(data: Record<string, any>) {
  const entries = Object.entries(data || {}).slice(0, 4);
  const numericValues = entries
    .map(([, value]) => (typeof value === "number" ? Math.abs(value) : 0))
    .filter((value) => value > 0);
  const maxValue = Math.max(...numericValues, 1);

  return (
    <div className="space-y-3">
      {entries.map(([key, value]) => {
        const normalized = typeof value === "number" ? Math.min(100, (Math.abs(value) / maxValue) * 100) : 0;
        return (
          <div key={key} className="space-y-2 rounded-2xl bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
              <span>{key}</span>
              <span>{typeof value === "number" ? value.toFixed(6) : String(value)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-amber-400 transition-all"
                style={{ width: `${normalized}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

type TraceTab = "logs" | "results" | "artifacts";

export const ResultsPanel: React.FC<ResultsPanelProps> = ({ onShowResultModal }) => {
  const [activeTab, setActiveTab] = useState<TraceTab>("logs");
  const {
    nodes,
    nodeTypes,
    executionResults,
    executionLogs,
    executionArtifacts,
    isExecuting,
    progressMsg,
    taskId,
    showResultsPanel,
    setShowResultsPanel,
  } = useStore();

  const resultEntries = useMemo(
    () =>
      Object.entries(executionResults).filter((entry): entry is [string, ExecutionResult] => Boolean(entry[1])),
    [executionResults],
  );

  const resultCount = resultEntries.length;
  const resultTypeCounts = useMemo(
    () =>
      resultEntries.reduce<Record<string, number>>((acc, [, result]) => {
        acc[result.type] = (acc[result.type] || 0) + 1;
        return acc;
      }, {}),
    [resultEntries],
  );

  const hasContent = isExecuting || executionLogs.length > 0 || resultCount > 0 || executionArtifacts.length > 0;

  if (!hasContent) return null;

  const tabs: Array<{ id: TraceTab; label: string; count: number }> = [
    { id: "logs", label: "Logs", count: executionLogs.length },
    { id: "results", label: "Results", count: resultCount },
    { id: "artifacts", label: "Artifacts", count: executionArtifacts.length },
  ];

  return (
    <AnimatePresence>
      <motion.section
        initial={{ y: 24, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 24, opacity: 0 }}
        className="bg-[#09101a]/78 shadow-[0_-18px_40px_rgba(0,0,0,0.24),inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-2xl"
      >
        <div
          className="flex cursor-pointer items-center justify-between px-5 py-3"
          onClick={() => setShowResultsPanel(!showResultsPanel)}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-300">
              {isExecuting ? <Activity size={16} className="animate-pulse" /> : <TerminalSquare size={16} />}
            </div>
            <div>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                Execution Trace
              </div>
              <div className="mt-1 text-sm font-semibold text-white">
                {isExecuting ? progressMsg || "Running workflow..." : "Latest workflow activity"}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden items-center gap-2 text-[11px] text-slate-400 md:flex">
              <span>{executionLogs.length} logs</span>
              <span>{resultCount} results</span>
              <span>{executionArtifacts.length} artifacts</span>
            </div>
            {taskId && (
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[10px] font-mono uppercase tracking-[0.14em] text-slate-400">
                {taskId.slice(0, 8)}
              </span>
            )}
            {showResultsPanel ? <ChevronDown size={18} className="text-slate-500" /> : <ChevronUp size={18} className="text-slate-500" />}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {showResultsPanel && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 280, opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="flex h-full min-h-0 flex-col">
                <div className="flex items-center gap-2 px-5 py-3">
                  {tabs.map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setActiveTab(tab.id)}
                      className={`rounded-full px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] transition ${activeTab === tab.id
                          ? "bg-cyan-500/12 text-cyan-300 ring-1 ring-cyan-400/25"
                          : "bg-white/[0.04] text-slate-400 hover:bg-white/[0.06] hover:text-slate-200"
                        }`}
                    >
                      {tab.label} <span className="ml-1 text-slate-500">{tab.count}</span>
                    </button>
                  ))}
                </div>

                <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
                  {activeTab === "logs" && (
                    <div className="rounded-3xl bg-black/20 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                      <div className="space-y-2 font-mono text-[12px] leading-6 text-slate-300">
                        {executionLogs.length > 0 ? (
                          executionLogs.slice(-250).map((line, index) => (
                            <div key={`${index}-${line}`} className="rounded-xl bg-white/[0.03] px-3 py-2">
                              {line}
                            </div>
                          ))
                        ) : (
                          <div className="text-sm text-slate-500">Logs will stream here during execution.</div>
                        )}
                      </div>
                    </div>
                  )}

                  {activeTab === "results" && (
                    <div className="space-y-4">
                      {resultEntries.length > 0 && (
                        <div className="flex flex-wrap items-center gap-2 rounded-3xl bg-white/[0.04] px-4 py-3 text-[12px] text-slate-300 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                          <span className="font-semibold text-white">Result summary</span>
                          {Object.entries(resultTypeCounts).map(([type, count]) => {
                            const badge = formatResultBadge(type);
                            return (
                              <span
                                key={type}
                                className={`rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.16em] ${badge.color}`}
                              >
                                {badge.label}: {count}
                              </span>
                            );
                          })}
                        </div>
                      )}

                      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
                        {resultEntries.length > 0 ? (
                          resultEntries.map(([nodeId, result]) => {
                            const node = nodes.find((item) => item.id === nodeId);
                            const nodeName = node ? nodeTypes[node.type]?.name || node.id : nodeId;
                            const typeIcon =
                              result.type === "plot" ? <ImageIcon size={14} className="text-fuchsia-300" /> :
                                result.type === "metrics" ? <BarChart3 size={14} className="text-amber-300" /> :
                                  <FileText size={14} className="text-cyan-300" />;

                            return (
                              <button
                                key={nodeId}
                                type="button"
                                onClick={() => onShowResultModal(nodeId, result)}
                                className="rounded-3xl bg-white/[0.03] p-4 text-left shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:bg-cyan-500/[0.06]"
                              >
                                <div className="mb-4 flex items-center justify-between gap-3">
                                  <div className="flex min-w-0 items-center gap-2">
                                    {typeIcon}
                                    <span className="truncate text-sm font-semibold text-white">{nodeName}</span>
                                  </div>
                                  <span className="rounded-full border border-white/10 bg-black/20 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-slate-500">
                                    {result.type}
                                  </span>
                                </div>
                                <div className="rounded-2xl bg-black/20 p-3 text-xs text-slate-300 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                                  {result.type === "metrics" ? (
                                    renderMetricPreview((result as any).data || {})
                                  ) : result.type === "plot" || result.type === "image" ? (
                                    <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-slate-950/40">
                                      <img
                                        src={`data:image/png;base64,${(result as any).data}`}
                                        alt="Result preview"
                                        className="h-32 w-full object-cover"
                                      />
                                      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/90 to-transparent p-3 text-[11px] text-slate-200">
                                        {result.type === "plot" ? "Plot preview" : "Image preview"}
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="line-clamp-5 font-mono leading-5 text-slate-400">
                                      {String((result as any).data)}
                                    </div>
                                  )}
                                </div>
                              </button>
                            );
                          })
                        ) : (
                          <div className="text-sm text-slate-500">Node outputs and generated plots will appear here.</div>
                        )}
                      </div>
                  )}

                      {activeTab === "artifacts" && (
                        <div className="space-y-3">
                          {executionArtifacts.length > 0 ? (
                            executionArtifacts.map((artifact) => (
                              <a
                                key={artifact}
                                href={taskId ? `${getApiBase()}/artifact/${taskId}/${artifact}` : "#"}
                                target="_blank"
                                rel="noreferrer"
                                onClick={(event) => {
                                  if (!taskId) {
                                    event.preventDefault();
                                  }
                                }}
                                className="flex items-center justify-between rounded-2xl bg-white/[0.03] px-4 py-3 text-sm text-slate-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:bg-cyan-500/[0.06]"
                              >
                                <span className="flex items-center gap-2">
                                  <FolderKanban size={15} className="text-cyan-300" />
                                  {artifact}
                                </span>
                                <span className="text-[10px] uppercase tracking-[0.14em] text-slate-500">Download</span>
                              </a>
                            ))
                          ) : (
                            <div className="text-sm text-slate-500">Saved plots and files from workflow execution will appear here.</div>
                          )}
                        </div>
                      )}
                    </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>
    </AnimatePresence>
  );
};
