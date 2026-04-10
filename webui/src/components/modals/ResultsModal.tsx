
import React from "react";
import { X, Settings } from "lucide-react";
import type { ExecutionResult } from "../../types/types";

export const ResultsModal: React.FC<{
  modal: { nodeId: string; result: ExecutionResult } | null;
  onClose: () => void;
}> = ({ modal, onClose }) => {
  if (!modal) return null;

  const rawJson = JSON.stringify(modal.result.data, null, 2);
  const downloadHref =
    modal.result.type === "plot" || modal.result.type === "image"
      ? `data:image/png;base64,${modal.result.data}`
      : undefined;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(rawJson);
    } catch {
      // ignore clipboard failures
    }
  };

  const renderMetricsContent = (data: Record<string, any>) => {
    const entries = Object.entries(data || {});
    const numericValues = entries
      .map(([, value]) => (typeof value === "number" ? Math.abs(value) : 0))
      .filter((value) => value > 0);
    const maxValue = Math.max(...numericValues, 1);

    return (
      <div className="space-y-4">
        {entries.map(([key, value]) => (
          <div key={key} className="rounded-3xl bg-slate-950/70 p-4 border border-slate-800/70">
            <div className="flex items-center justify-between gap-3 text-sm text-slate-400">
              <span className="font-medium text-slate-200">{key.replace(/_/g, " ")}</span>
              <span className="font-semibold text-white">
                {typeof value === "number" ? value.toFixed(6) : String(value)}
              </span>
            </div>
            {typeof value === "number" && (
              <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-emerald-400 transition-all"
                  style={{ width: `${Math.min(100, (Math.abs(value) / maxValue) * 100)}%` }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderResult = (res: ExecutionResult) => {
    if (!res) return <div className="text-slate-400">No data available</div>;
    switch (res.type) {
      case "plot":
      case "image":
        return (
          <div className="overflow-hidden rounded-3xl border border-slate-700/60 bg-slate-950/70 shadow-xl shadow-slate-950/20">
            <img src={`data:image/png;base64,${res.data}`} alt="Visualization" className="w-full h-auto rounded-3xl" />
          </div>
        );
      case "metrics":
        return (
          <div className="space-y-5">
            <div className="rounded-3xl bg-slate-950/70 p-6 border border-slate-800/60 shadow-inner shadow-slate-950/20">
              <div className="text-lg font-semibold text-white">Performance Metrics</div>
              <div className="mt-2 text-sm text-slate-400">A preview of the most important model evaluation numbers.</div>
            </div>
            {renderMetricsContent(res.data || {})}
          </div>
        );
      case "table":
        return (
          <div className="overflow-x-auto rounded-3xl border border-slate-700/60 bg-slate-950/70 p-4 shadow-xl shadow-slate-950/10">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  {res.data.columns?.map((col: string, i: number) => (
                    <th key={i} className="text-left p-2 text-slate-300 font-semibold">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {res.data.rows?.map((row: any[], i: number) => (
                  <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/30">
                    {row.map((cell, j) => (
                      <td key={j} className="p-2 text-slate-200">{String(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      case "json":
        return (
          <div className="overflow-x-auto rounded-3xl border border-slate-700/60 bg-slate-950/70 p-6 shadow-xl shadow-slate-950/10">
            <pre className="text-sm text-green-400 font-mono">{rawJson}</pre>
          </div>
        );
      case "array":
        return (
          <div className="space-y-4">
            <div className="rounded-3xl bg-slate-950/70 p-6 border border-slate-800/60 shadow-inner shadow-slate-950/20">
              <h3 className="text-lg font-semibold text-white">Array Data</h3>
              <p className="mt-2 text-sm text-slate-400">Showing the first 100 elements.</p>
            </div>
            <div className="grid gap-2">
              {Array.isArray(res.data) ? (
                res.data.slice(0, 100).map((item: any, i: number) => (
                  <div key={i} className="rounded-2xl bg-slate-900/80 p-3 text-sm text-slate-200">
                    <span className="text-slate-500">[{i}]</span> {JSON.stringify(item)}
                  </div>
                ))
              ) : (
                <div className="text-slate-400">Invalid array data</div>
              )}
              {Array.isArray(res.data) && res.data.length > 100 && (
                <div className="text-slate-500 text-sm">... and {res.data.length - 100} more items</div>
              )}
            </div>
          </div>
        );
      case "text":
      default:
        return (
          <div className="overflow-x-auto rounded-3xl border border-slate-700/60 bg-slate-950/70 p-6 shadow-xl shadow-slate-950/10">
            <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">{String(res.data)}</pre>
          </div>
        );
    }
  };
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900/95 backdrop-blur-xl rounded-2xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col border border-slate-700/50">
        <div className="flex flex-col gap-4 p-6 border-b border-slate-700/50 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
              Execution Results
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Node: <span className="font-semibold text-white">{modal.nodeId}</span> • Type: <span className="font-semibold text-cyan-300">{modal.result.type}</span>
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              className="rounded-2xl bg-white/5 px-3 py-2 text-sm text-slate-100 transition hover:bg-white/[0.08]"
            >
              Copy data
            </button>
            {downloadHref && (
              <a
                href={downloadHref}
                download={`${modal.nodeId}-${modal.result.type}.png`}
                className="rounded-2xl bg-cyan-500/10 px-3 py-2 text-sm font-semibold text-cyan-200 transition hover:bg-cyan-500/15"
              >
                Download image
              </a>
            )}
            <button onClick={onClose} className="rounded-2xl bg-white/5 px-3 py-2 text-sm text-slate-100 transition hover:bg-white/[0.08]">
              Close
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-6">
          {renderResult(modal.result)}
        </div>
      </div>
    </div>
  );
};
