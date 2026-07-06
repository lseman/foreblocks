
import React from "react";
import { X } from "lucide-react";
import type { CategoryMap, NodeTypeMap } from "../types/types";

export const NodeMenu: React.FC<{
  visible: boolean;
  onClose: () => void;
  onAdd: (type: string) => void;
  searchQuery: string;
  setSearchQuery: (v: string) => void;
  groups: CategoryMap;
  nodeTypes: NodeTypeMap;
}> = ({ visible, onClose, onAdd, searchQuery, setSearchQuery, groups, nodeTypes }) => {
  if (!visible) return null;
  return (
    <div className="w-80 bg-[#0f172a]/70 backdrop-blur-2xl border-r border-slate-600/35 p-4 overflow-y-auto shadow-[0_20px_45px_rgba(0,0,0,0.45)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-semibold text-lg tracking-tight">Node Library</h2>
          <p className="text-[11px] text-slate-400 mt-0.5">Build flows with reusable blocks</p>
        </div>
        <button onClick={onClose} className="hover:bg-slate-700/50 p-2 rounded-lg transition border border-transparent hover:border-slate-500/50">
          <X size={18} />
        </button>
      </div>
      <input
        type="text" placeholder="Search nodes..." value={searchQuery}
        onChange={(e) => setSearchQuery((e.target as HTMLInputElement).value)}
        className="w-full px-3 py-2.5 bg-slate-900/70 border border-slate-600/50 rounded-xl mb-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
      />
      {Object.entries(groups).map(([category, nodeTypeIds]) => (
        <div key={category} className="mb-4">
          <h3 className="text-[10px] font-semibold text-slate-400 mb-2 uppercase tracking-[0.18em]">{category}</h3>
          {nodeTypeIds.map(type => {
            const info = nodeTypes[type] || {};
            return (
              <button
                key={type}
                onClick={() => onAdd(type)}
                className="w-full text-left p-3 mb-2 rounded-xl bg-slate-800/60 border border-slate-600/35 hover:border-slate-400/40 transition-all shadow-md hover:shadow-lg hover:translate-y-[-1px]"
                style={{
                  boxShadow: "0 8px 24px rgba(0,0,0,0.22)",
                }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="inline-flex h-2.5 w-2.5 rounded-full"
                    style={{ background: typeof info.color === "string" && info.color.startsWith("#") ? info.color : "#60a5fa" }}
                  />
                  <div className="font-medium text-sm">{info.name || type}</div>
                </div>
                {info.subtypes && info.subtypes.length > 0 && (
                  <div className="text-xs mt-1 opacity-75 text-slate-300">
                    {info.subtypes.slice(0, 2).join(', ')}
                    {info.subtypes.length > 2 && '...'}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
};
