import React from "react";
import {
  Code,
  Download,
  Layout,
  MoreHorizontal,
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
    <header className="flex h-16 items-center justify-between gap-3 bg-[#0a1018]/84 px-3 shadow-[0_18px_50px_rgba(0,0,0,0.26),inset_0_-1px_0_rgba(255,255,255,0.04)] backdrop-blur-2xl sm:px-5">
      <div className="flex min-w-0 items-center gap-3 sm:gap-4">
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
            aria-label="Workflow name"
            className="mt-1 w-[150px] max-w-full rounded-xl border border-transparent bg-transparent px-0 py-0 text-sm font-semibold tracking-tight text-white outline-none transition focus:border-cyan-400/30 focus:bg-white/[0.03] focus:px-3 focus:py-2 sm:w-[240px] sm:text-base 2xl:w-[320px]"
          />
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onCreateWorkflow}
          className="hidden rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] xl:block"
        >
          New
        </button>
        <button
          type="button"
          onClick={onSaveWorkflow}
          className="hidden items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] xl:inline-flex"
        >
          <Save size={14} />
          Save
        </button>
        <select
          defaultValue=""
          onChange={(event) => {
            if (!event.target.value) return;
            loadTemplate(event.target.value);
            event.target.value = "";
          }}
          aria-label="Load workflow template"
          className="hidden rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] 2xl:block"
        >
          <option value="" disabled>
            Template
          </option>
          <option value="basic_transformer">Transformer</option>
          <option value="basic_mamba">Mamba Direct</option>
        </select>
        <button
          type="button"
          onClick={() => autoLayout("LR")}
          className="hidden items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] 2xl:inline-flex"
        >
          <Layout size={14} />
          Layout
        </button>
        <button
          type="button"
          onClick={onViewCode}
          className="hidden items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] xl:inline-flex"
        >
          <Code size={14} />
          Code
        </button>
        <button
          type="button"
          onClick={handleExport}
          className="hidden items-center gap-2 rounded-xl bg-white/[0.05] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-cyan-500/[0.08] 2xl:inline-flex"
        >
          <Download size={14} />
          Export
        </button>
        <details className="group relative xl:hidden">
          <summary
            className="flex h-10 w-10 cursor-pointer list-none items-center justify-center rounded-xl bg-white/[0.05] text-slate-200 transition hover:bg-cyan-500/[0.08] [&::-webkit-details-marker]:hidden"
            aria-label="More workflow actions"
          >
            <MoreHorizontal size={17} />
          </summary>
          <div className="absolute right-0 top-12 z-50 w-48 overflow-hidden rounded-2xl border border-white/10 bg-[#101824]/95 p-2 shadow-2xl backdrop-blur-2xl">
            {[
              ["New workflow", onCreateWorkflow],
              ["Save workflow", onSaveWorkflow],
              ["Auto layout", () => autoLayout("LR")],
              ["View code", onViewCode],
              ["Export JSON", handleExport],
            ].map(([label, action]) => (
              <button
                key={String(label)}
                type="button"
                onClick={action as () => void}
                className="block w-full rounded-xl px-3 py-2 text-left text-xs font-medium text-slate-200 transition hover:bg-white/[0.06]"
              >
                {label as string}
              </button>
            ))}
          </div>
        </details>
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
          aria-label="Clear canvas"
        >
          <Trash2 size={15} />
        </button>
      </div>
    </header>
  );
};
