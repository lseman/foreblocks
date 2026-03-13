import React from "react";
import { AlertTriangle, FileJson, Settings2, Sparkles, TerminalSquare } from "lucide-react";
import { useStore } from "../store/store";
import { ConfigPanel } from "./ConfigPanel";

type InspectorSidebarProps = {
  workflowName: string;
};

export const InspectorSidebar: React.FC<InspectorSidebarProps> = ({ workflowName }) => {
  const {
    selectedNodeId,
    nodes,
    connections,
    executionResults,
    executionLogs,
    preflightIssues,
  } = useStore();

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) || null;
  const resultCount = Object.keys(executionResults).length;

  return (
    <aside className="h-full min-h-0 bg-[#0b1119]/88 shadow-[inset_1px_0_0_rgba(255,255,255,0.04)] backdrop-blur-2xl">
      <div className="flex h-full flex-col">
        <div className="px-5 py-4">
          <div className="flex items-center gap-3 rounded-[24px] bg-black/20 px-4 py-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/[0.05] text-cyan-300">
              {selectedNode ? <Settings2 size={18} /> : <Sparkles size={18} />}
            </div>
            <div className="min-w-0">
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                Inspector
              </div>
              <div className="mt-1 truncate text-sm font-semibold text-white">
                {selectedNode ? selectedNode.id : workflowName}
              </div>
            </div>
          </div>
        </div>

        {selectedNode ? (
          <div className="min-h-0 flex-1 overflow-hidden">
            <ConfigPanel variant="dock" />
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
            <div className="space-y-4">
              <div className="rounded-3xl bg-white/[0.03] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                <div className="mb-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                  <FileJson size={12} />
                  Workflow Summary
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-2xl bg-black/20 p-3">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Nodes</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{nodes.length}</div>
                  </div>
                  <div className="rounded-2xl bg-black/20 p-3">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Connections</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{connections.length}</div>
                  </div>
                  <div className="rounded-2xl bg-black/20 p-3">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Results</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{resultCount}</div>
                  </div>
                  <div className="rounded-2xl bg-black/20 p-3">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">Log Lines</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{executionLogs.length}</div>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl bg-white/[0.03] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                <div className="mb-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                  <AlertTriangle size={12} />
                  Preflight
                </div>
                <div className="space-y-2 text-sm text-slate-300">
                  <div className="flex items-center justify-between rounded-2xl bg-black/20 px-3 py-2">
                    <span>Errors</span>
                    <span className="font-semibold text-rose-300">{preflightIssues.errors.length}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl bg-black/20 px-3 py-2">
                    <span>Warnings</span>
                    <span className="font-semibold text-amber-300">{preflightIssues.warnings.length}</span>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl bg-white/[0.03] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                <div className="mb-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                  <TerminalSquare size={12} />
                  Hint
                </div>
                <p className="text-sm leading-6 text-slate-400">
                  Select a node to open its configuration. Execution logs and artifacts appear in the bottom trace panel while runs are active.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
