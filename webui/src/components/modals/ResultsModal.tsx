
import React from "react";
import { X, Settings } from "lucide-react";
import type { ExecutionResult } from "../../types/types";

export const ResultsModal: React.FC<{
  modal: { nodeId: string; result: ExecutionResult } | null;
  onClose: () => void;
}> = ({ modal, onClose }) => {
  if (!modal) return null;
  const renderResult = (res: ExecutionResult) => {
    if (!res) return <div className="text-slate-400">No data available</div>;
    switch (res.type) {
      case "plot":
      case "image":
        return (
          <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800/50">
            <img src={`data:image/png;base64,${res.data}`} alt="Visualization" className="w-full h-auto rounded-lg" />
          </div>
        );
      case "metrics":
        return (
          <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-800/50">
            <h3 className="text-lg font-semibold mb-4 text-white">Performance Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(res.data || {}).map(([k, v]) => (
                <div key={k} className="bg-slate-800/50 p-4 rounded-lg border border-slate-700/50">
                  <div className="text-sm text-slate-400 uppercase tracking-wider mb-1">{k.replace(/_/g, " ")}</div>
                  <div className="text-2xl font-bold text-white">
                    {typeof v === "number" ? v.toFixed(6) : String(v)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      case "table":
        return (
          <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800/50 overflow-x-auto">
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
          <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-800/50">
            <pre className="text-sm text-green-400 font-mono overflow-x-auto">
              {JSON.stringify(res.data, null, 2)}
            </pre>
          </div>
        );
      case "array":
        return (
          <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-800/50">
            <h3 className="text-lg font-semibold mb-4 text-white">Array Data</h3>
            <div className="space-y-2">
              {Array.isArray(res.data) ? (
                <div className="grid grid-cols-1 gap-2">
                  {res.data.slice(0, 100).map((item: any, i: number) => (
                    <div key={i} className="bg-slate-800/50 p-2 rounded text-sm">
                      <span className="text-slate-400 mr-2">[{i}]</span>
                      <span className="text-white">{JSON.stringify(item)}</span>
                    </div>
                  ))}
                  {res.data.length > 100 && (
                    <div className="text-slate-400 text-sm">... and {res.data.length - 100} more items</div>
                  )}
                </div>
              ) : (
                <div className="text-slate-400">Invalid array data</div>
              )}
            </div>
          </div>
        );
      case "text":
      default:
        return (
          <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-800/50">
            <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">{String(res.data)}</pre>
          </div>
        );
    }
  };
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900/95 backdrop-blur-xl rounded-2xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col border border-slate-700/50">
        <div className="flex items-center justify-between p-6 border-b border-slate-700/50">
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
              Execution Results
            </h2>
          </div>
          <button onClick={onClose} className="hover:bg-slate-700/50 p-2 rounded-lg transition">
            <X size={24} />
          </button>
        </div>
        <div className="flex-1 overflow-auto p-6">
          {renderResult(modal.result)}
        </div>
      </div>
    </div>
  );
};
