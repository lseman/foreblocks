import React from "react";
import {
  Code,
  Download,
  Layout,
  Play,
  Save,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useStore } from "../store/store";
import { useExecution } from "../hooks/useExecution";

interface EditorToolbarProps {
  workflowName: string;
  onWorkflowNameChange: (value: string) => void;
  onViewCode: () => void;
  loadTemplate: (name: string) => void;
  autoLayout: (dir: "LR" | "TB") => void;
  onSaveWorkflow: () => void;
  onCreateWorkflow: () => void;
}

export const EditorToolbar: React.FC<EditorToolbarProps> = ({
  workflowName,
  onWorkflowNameChange,
  onViewCode,
  loadTemplate,
  autoLayout,
  onSaveWorkflow,
  onCreateWorkflow,
}) => {
  const {
    nodes,
    connections,
    isExecuting,
    setSelectedNodeId,
    setGraph,
    executionLogs,
  } = useStore();

  const { executeWorkflow } = useExecution();

  const handleExport = () => {
    const config = {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.type,
        subtype: n.subtype,
        position: n.position,
        config: n.config,
      })),
      connections,
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${workflowName || "workflow"}.json`;
    a.click();
  };

  return (
    <header className="flex h-16 items-center justify-between bg-[#0a1018]/84 px-5 shadow-[0_18px_50px_rgba(0,0,0,0.26),inset_0_-1px_0_rgba(255,255,255,0.04)] backdrop-blur-2xl">
      <div className="flex min-w-0 items-center gap-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-indigo-600 shadow-lg shadow-blue-500/20">
          <Sparkles size={18} />
        </div>
        <div className="min-w-0">
          <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
            Workflow Studio
          </div>
          <input
            value={workflowName}
            onChange={(e) => onWorkflowNameChange(e.target.value)}
            className="mt-1 w-[320px] max-w-full rounded-xl border border-transparent bg-transparent px-0 py-0 text-base font-semibold tracking-tight text-white outline-none transition focus:border-cyan-400/30 focus:bg-white/[0.03] focus:px-3 focus:py-2"
          />
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onCreateWorkflow}
          className="rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          New
        </button>
        <button
          type="button"
          onClick={onSaveWorkflow}
          className="inline-flex items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          <Save size={14} />
          Save
        </button>
        <button
          type="button"
          onClick={() => loadTemplate("basic_transformer")}
          className="rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          Template
        </button>
        <button
          type="button"
          onClick={() => autoLayout("LR")}
          className="inline-flex items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          <Layout size={14} />
          Layout
        </button>
        <button
          type="button"
          onClick={onViewCode}
          className="inline-flex items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          <Code size={14} />
          Code
        </button>
        <button
          type="button"
          onClick={handleExport}
          className="inline-flex items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08]"
        >
          <Download size={14} />
          Export
        </button>
        <div className="mx-2 hidden h-8 w-px bg-white/8 lg:block" />
        <div className="hidden items-center gap-3 text-[11px] text-slate-400 lg:flex">
          <span>{nodes.length} nodes</span>
          <span>{connections.length} edges</span>
          <span>{executionLogs.length} logs</span>
        </div>
        <button
          type="button"
          onClick={executeWorkflow}
          disabled={nodes.length === 0 || isExecuting}
          className={`ml-2 inline-flex items-center gap-2 rounded-2xl px-4 py-2.5 text-xs font-bold uppercase tracking-[0.14em] transition ${
            isExecuting
              ? "bg-amber-500/16 text-amber-300 ring-1 ring-amber-400/25"
              : "bg-gradient-to-br from-emerald-400 to-teal-500 text-slate-950 shadow-lg shadow-emerald-500/20 hover:from-emerald-300 hover:to-teal-400"
          } disabled:cursor-not-allowed disabled:opacity-40`}
        >
          <Play size={14} className={isExecuting ? "animate-pulse" : ""} />
          {isExecuting ? "Running" : "Run"}
        </button>
        <button
          type="button"
          onClick={() => {
            if (window.confirm("Clear all nodes and connections?")) {
              setGraph([], []);
              setSelectedNodeId(null);
            }
          }}
          className="rounded-xl bg-rose-500/[0.08] p-2.5 text-rose-300 transition hover:bg-rose-500/[0.14]"
          title="Clear canvas"
        >
          <Trash2 size={15} />
        </button>
      </div>
    </header>
  );
};
