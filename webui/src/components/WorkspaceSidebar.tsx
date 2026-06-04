import React, { useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Clock3,
  FolderKanban,
  Info,
  Layers3,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  Sparkles,
  Workflow,
} from "lucide-react";
import type { CategoryMap, NodeTypeMap } from "../types/types";
import type { SavedWorkflow } from "../utils/workflows";

type SidebarSection = "workflows" | "recent" | "library";

type WorkspaceSidebarProps = {
  nodeSearch: string;
  setNodeSearch: (value: string) => void;
  groups: CategoryMap;
  nodeTypes: NodeTypeMap;
  onAddNode: (type: string) => void;
  recentNodeTypes: string[];
  workflows: SavedWorkflow[];
  currentWorkflowId: string | null;
  onSelectWorkflow: (id: string) => void;
  onCreateWorkflow: () => void;
  onDeleteWorkflow: (id: string) => void;
  collapsed: boolean;
  activeSection: SidebarSection;
  onToggleCollapsed: () => void;
  onSectionChange: (section: SidebarSection) => void;
};

function getRequiredInputCount(_type: string, info: NodeTypeMap[string]): number {
  const optionalInputs = new Set(info.optional_inputs || []);
  return (info.inputs || []).filter((input) => !optionalInputs.has(input)).length;
}

function getNodeRole(type: string, info: NodeTypeMap[string]): string {
  const category = (info.category || "").toLowerCase();
  const role = info.py?.role || "";
  if (role) return role;
  if (type.includes("trainer") || category.includes("train")) return "trainer";
  if (type.includes("data") || type.includes("csv") || category.includes("data")) return "data";
  if (type.includes("model") || category.includes("model")) return "model";
  if (type.includes("head") || category.includes("head")) return "head";
  if (category.includes("encoder")) return "encoder";
  if (category.includes("decoder")) return "decoder";
  return category || "block";
}

function roleBadgeClass(role: string): string {
  switch (role) {
    case "trainer":
      return "bg-amber-500/10 text-amber-200 ring-1 ring-amber-400/20";
    case "data":
      return "bg-blue-500/10 text-blue-200 ring-1 ring-blue-400/20";
    case "model":
      return "bg-cyan-500/10 text-cyan-200 ring-1 ring-cyan-400/20";
    case "head":
      return "bg-fuchsia-500/10 text-fuchsia-200 ring-1 ring-fuchsia-400/20";
    case "encoder":
      return "bg-emerald-500/10 text-emerald-200 ring-1 ring-emerald-400/20";
    case "decoder":
      return "bg-violet-500/10 text-violet-200 ring-1 ring-violet-400/20";
    default:
      return "bg-white/[0.05] text-slate-300 ring-1 ring-white/10";
  }
}

export const WorkspaceSidebar: React.FC<WorkspaceSidebarProps> = ({
  nodeSearch,
  setNodeSearch,
  groups,
  nodeTypes,
  onAddNode,
  recentNodeTypes,
  workflows,
  currentWorkflowId,
  onSelectWorkflow,
  onCreateWorkflow,
  onDeleteWorkflow,
  collapsed,
  activeSection,
  onToggleCollapsed,
  onSectionChange,
}) => {
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(() => new Set());
  const [selectedLibraryType, setSelectedLibraryType] = useState<string | null>(null);

  const recentNodes = useMemo(
    () =>
      recentNodeTypes
        .map((type) => ({ type, def: nodeTypes[type] }))
        .filter((item) => item.def),
    [nodeTypes, recentNodeTypes],
  );

  const selectedLibraryInfo = selectedLibraryType ? nodeTypes[selectedLibraryType] : null;

  const sections: Array<{
    id: SidebarSection;
    label: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
    helper: string;
  }> = [
    { id: "workflows", label: "Workflows", icon: FolderKanban, helper: "Projects and autosave" },
    { id: "recent", label: "Recent", icon: Clock3, helper: "Quick-add blocks" },
    { id: "library", label: "Library", icon: Layers3, helper: "Browse all nodes" },
  ];

  const renderSectionContent = () => {
    if (activeSection === "workflows") {
      return (
        <div className="flex min-h-0 flex-1 flex-col px-4 pb-4">
          <div className="mb-3 flex items-center justify-between rounded-2xl bg-black/20 px-3 py-3">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                Workflows
              </div>
              <div className="mt-1 text-xs text-slate-400">
                Saved locally with autosave
              </div>
            </div>
            <button
              type="button"
              onClick={onCreateWorkflow}
              className="inline-flex h-9 items-center gap-2 rounded-xl bg-cyan-500/10 px-3 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-500/16"
            >
              <Plus size={14} />
              New
            </button>
          </div>

          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
            {workflows.map((workflow) => (
              <button
                key={workflow.id}
                type="button"
                onClick={() => onSelectWorkflow(workflow.id)}
                className={`group w-full rounded-3xl px-3 py-3 text-left transition ${
                  workflow.id === currentWorkflowId
                    ? "bg-cyan-500/12 shadow-[inset_0_0_0_1px_rgba(56,189,248,0.25)]"
                    : "bg-white/[0.03] hover:bg-white/[0.05]"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-white">
                      {workflow.name}
                    </div>
                    <div className="mt-1 line-clamp-2 text-[11px] text-slate-400">
                      {workflow.description || `${workflow.nodes.length} nodes · ${workflow.connections.length} connections`}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteWorkflow(workflow.id);
                    }}
                    className="rounded-lg px-2 py-1 text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500 opacity-0 transition group-hover:opacity-100 hover:bg-rose-500/10 hover:text-rose-300"
                  >
                    Delete
                  </button>
                </div>
                <div className="mt-3 flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-slate-500">
                  <span>{workflow.nodes.length} nodes</span>
                  <span>{new Date(workflow.updatedAt).toLocaleDateString()}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      );
    }

    if (activeSection === "recent") {
      return (
        <div className="flex min-h-0 flex-1 flex-col px-4 pb-4">
          <div className="mb-3 rounded-2xl bg-black/20 px-3 py-3">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
              <Clock3 size={12} />
              Recent Nodes
            </div>
            <div className="mt-1 text-xs text-slate-400">
              Reuse the blocks you touched most recently.
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <div className="flex flex-wrap gap-2">
              {recentNodes.length > 0 ? (
                recentNodes.map(({ type, def }) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => onAddNode(type)}
                    className="rounded-full bg-white/[0.05] px-3 py-1.5 text-[11px] font-medium text-slate-200 transition hover:bg-cyan-500/10 hover:text-white"
                  >
                    {def?.name || type}
                  </button>
                ))
              ) : (
                <div className="text-xs text-slate-500">Your recently used blocks will show up here.</div>
              )}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex min-h-0 flex-1 flex-col px-4 pb-4">
        <div className="mb-3 rounded-2xl bg-black/20 px-3 py-3">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
            <Layers3 size={12} />
            Node Library
          </div>
          <div className="mt-1 text-xs text-slate-400">
            Search blocks, categories, and roles.
          </div>
        </div>
        <div className="relative mb-4">
          <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={15} />
          <input
            value={nodeSearch}
            onChange={(e) => setNodeSearch(e.target.value)}
            placeholder="Search blocks, categories, roles..."
            className="w-full rounded-2xl bg-white/[0.04] py-3 pl-10 pr-4 text-sm text-white outline-none transition focus:bg-white/[0.07]"
          />
        </div>

        {selectedLibraryInfo && selectedLibraryType && (
          <div className="mb-4 rounded-2xl bg-white/[0.035] px-3 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500">
                  <Info size={12} />
                  Node
                </div>
                <div className="mt-1 truncate text-sm font-semibold text-white">
                  {selectedLibraryInfo.name || selectedLibraryType}
                </div>
              </div>
              <span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${roleBadgeClass(getNodeRole(selectedLibraryType, selectedLibraryInfo))}`}>
                {getNodeRole(selectedLibraryType, selectedLibraryInfo)}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[10px] uppercase tracking-[0.12em] text-slate-500">
              <div className="rounded-xl bg-black/20 px-2 py-2">
                <div className="text-sm font-semibold text-slate-100">{getRequiredInputCount(selectedLibraryType, selectedLibraryInfo)}</div>
                <div>Required</div>
              </div>
              <div className="rounded-xl bg-black/20 px-2 py-2">
                <div className="text-sm font-semibold text-slate-100">{selectedLibraryInfo.optional_inputs?.length || 0}</div>
                <div>Optional</div>
              </div>
              <div className="rounded-xl bg-black/20 px-2 py-2">
                <div className="text-sm font-semibold text-slate-100">{selectedLibraryInfo.outputs?.length || 0}</div>
                <div>Outputs</div>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {(selectedLibraryInfo.outputs || []).slice(0, 5).map((port) => (
                <span key={port} className="rounded-full bg-cyan-500/10 px-2 py-1 text-[10px] text-cyan-200">
                  {port}
                </span>
              ))}
              {(selectedLibraryInfo.py?.ctor || selectedLibraryInfo.py?.role) && (
                <span className="rounded-full bg-white/[0.05] px-2 py-1 text-[10px] text-slate-300">
                  {selectedLibraryInfo.py?.ctor || selectedLibraryInfo.py?.role}
                </span>
              )}
            </div>
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto pr-1">
          {Object.entries(groups).map(([category, nodeTypeIds]) => {
            const isCollapsed = collapsedCategories.has(category);
            return (
              <section key={category} className="mb-4">
                <button
                  type="button"
                  onClick={() =>
                    setCollapsedCategories((prev) => {
                      const next = new Set(prev);
                      if (next.has(category)) next.delete(category);
                      else next.add(category);
                      return next;
                    })
                  }
                  className="mb-2 flex w-full items-center justify-between rounded-xl px-2 py-1.5 text-left text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 transition hover:bg-white/[0.04] hover:text-slate-300"
                >
                  <span className="flex items-center gap-2">
                    <Sparkles size={11} />
                    {category}
                  </span>
                  <span className="flex items-center gap-2">
                    {nodeTypeIds.length}
                    {isCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                  </span>
                </button>

                {!isCollapsed && (
                  <div className="space-y-2">
                    {nodeTypeIds.map((type) => {
                      const info = nodeTypes[type] || {};
                      const optionalInputs = new Set(info.optional_inputs || []);
                      const requiredInputCount = getRequiredInputCount(type, info);
                      const optionalInputCount = optionalInputs.size;
                      const role = getNodeRole(type, info);
                      const selected = selectedLibraryType === type;
                      return (
                        <button
                          key={type}
                          type="button"
                          draggable
                          onClick={() => onAddNode(type)}
                          onFocus={() => setSelectedLibraryType(type)}
                          onMouseEnter={() => setSelectedLibraryType(type)}
                          onDragStart={(event) => {
                            setSelectedLibraryType(type);
                            event.dataTransfer.setData("application/x-foreblocks-node-type", type);
                            event.dataTransfer.effectAllowed = "copy";
                          }}
                          className={`w-full rounded-2xl px-3 py-3 text-left transition ${
                            selected
                              ? "bg-cyan-500/[0.08] shadow-[inset_0_0_0_1px_rgba(56,189,248,0.18)]"
                              : "bg-white/[0.03] hover:bg-cyan-500/[0.07]"
                          }`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex min-w-0 items-center gap-2">
                                <span className="truncate text-sm font-semibold text-white">
                                  {info.name || type}
                                </span>
                                <span className={`shrink-0 rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] ${roleBadgeClass(role)}`}>
                                  {role}
                                </span>
                              </div>
                              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-400">
                                <span>{requiredInputCount} required</span>
                                {optionalInputCount > 0 && <span>{optionalInputCount} optional</span>}
                                <span>{info.outputs?.length || 0} out</span>
                              </div>
                            </div>
                            <span
                              className="mt-1 h-3 w-3 shrink-0 rounded-full shadow-[0_0_14px_rgba(56,189,248,0.25)]"
                              style={{
                                background:
                                  typeof info.color === "string" && info.color.startsWith("#")
                                    ? info.color
                                    : "#38bdf8",
                              }}
                            />
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      </div>
    );
  };

  if (collapsed) {
    return (
      <aside className="h-full min-h-0 bg-[#0b1119]/88 text-white backdrop-blur-2xl">
        <div className="flex h-full w-[78px] flex-col items-center gap-3 px-3 py-4">
          <button
            type="button"
            onClick={onToggleCollapsed}
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-indigo-600 shadow-lg shadow-blue-500/20"
            title="Expand workspace"
          >
            <PanelLeftOpen size={18} />
          </button>

          <div className="mt-1 flex flex-col items-center gap-2 rounded-[24px] bg-black/20 px-2 py-2">
            {sections.map((section) => {
              const Icon = section.icon;
              const active = section.id === activeSection;
              return (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => {
                    onSectionChange(section.id);
                    onToggleCollapsed();
                  }}
                  className={`flex h-11 w-11 items-center justify-center rounded-2xl transition ${
                    active
                      ? "bg-cyan-500/16 text-cyan-200 shadow-[inset_0_0_0_1px_rgba(56,189,248,0.22)]"
                      : "text-slate-400 hover:bg-white/[0.05] hover:text-slate-100"
                  }`}
                  title={section.label}
                >
                  <Icon size={17} />
                </button>
              );
            })}
          </div>

          <button
            type="button"
            onClick={onCreateWorkflow}
            className="mt-auto flex h-11 w-11 items-center justify-center rounded-2xl bg-white/[0.05] text-slate-200 transition hover:bg-cyan-500/10 hover:text-white"
            title="New workflow"
          >
            <Plus size={16} />
          </button>
        </div>
      </aside>
    );
  }

  return (
    <aside className="h-full min-h-0 bg-[#0b1119]/92 text-white backdrop-blur-2xl">
      <div className="flex h-full flex-col">
        <div className="px-4 py-4">
          <div className="flex items-center justify-between gap-3 rounded-[28px] bg-black/20 px-4 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-indigo-600 shadow-lg shadow-blue-500/20">
                <Workflow size={20} />
              </div>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-300/80">
                  ForeBlocks Studio
                </div>
                <h1 className="mt-1 text-sm font-semibold tracking-tight text-white">
                  Workflow Workspace
                </h1>
              </div>
            </div>
            <button
              type="button"
              onClick={onToggleCollapsed}
              className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/[0.05] text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
              title="Collapse workspace"
            >
              <PanelLeftClose size={16} />
            </button>
          </div>
        </div>

        <div className="px-4 pb-4">
          <div className="grid grid-cols-3 gap-2 rounded-[24px] bg-black/20 p-2">
            {sections.map((section) => {
              const Icon = section.icon;
              const active = section.id === activeSection;
              return (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => onSectionChange(section.id)}
                  className={`rounded-2xl px-3 py-3 text-left transition ${
                    active
                      ? "bg-white/[0.07] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                      : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Icon size={15} className={active ? "text-cyan-300" : ""} />
                    <span className="text-xs font-semibold text-white">{section.label}</span>
                  </div>
                  <div className="mt-1 text-[10px] text-slate-500">
                    {section.helper}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {renderSectionContent()}
      </div>
    </aside>
  );
};
