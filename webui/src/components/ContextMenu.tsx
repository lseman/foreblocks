
import React from "react";
import type { CategoryMap, NodeTypeMap } from "../types/types";

export const ContextMenu: React.FC<{
  ctx: { x: number; y: number; screenX: number; screenY: number } | null;
  groups: CategoryMap;
  onAdd: (type: string, pos?: { x: number; y: number }) => void;
  searchQuery: string;
  setSearchQuery: (v: string) => void;
  onClose: () => void;
  nodeTypes: NodeTypeMap;
}> = ({ ctx, groups, onAdd, searchQuery, setSearchQuery, onClose, nodeTypes }) => {
  if (!ctx) return null;
  return (
    <div
      className="absolute bg-slate-800/95 backdrop-blur-xl rounded-xl shadow-2xl border border-slate-700/50 overflow-hidden"
      style={{
        left: Math.min(ctx.screenX, window.innerWidth - 300),
        top: Math.min(ctx.screenY, window.innerHeight - 400),
        width: 280, maxHeight: 500, zIndex: 1000
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="p-3 border-b border-slate-700/50">
        <input
          type="text" placeholder="Search nodes..." value={searchQuery}
          onChange={(e) => setSearchQuery((e.target as HTMLInputElement).value)} autoFocus
          className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600/50 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
        />
      </div>
      <div className="overflow-y-auto max-h-96 p-2">
        {Object.entries(groups).map(([category, nodeTypeIds]) => (
          <div key={category} className="mb-3">
            <h3 className="text-xs font-semibold text-slate-400 mb-1.5 px-2 uppercase tracking-wider">{category}</h3>
            {nodeTypeIds.map(type => {
              const info = nodeTypes[type] || {};
              return (
                <button
                  key={type}
                  onClick={() => { onAdd(type, { x: ctx.x - 130, y: ctx.y - 50 }); onClose(); }}
                  className={`w-full text-left p-2.5 mb-1 rounded-lg ${info.color || 'bg-slate-700/60'} hover:opacity-90 transition-all text-sm`}
                >
                  <div className="font-medium">{info.name || type}</div>
                  {info.subtypes && info.subtypes.length > 0 && (
                    <div className="text-xs mt-0.5 opacity-75">
                      {info.subtypes[0]}{info.subtypes.length > 1 && ` +${info.subtypes.length - 1}`}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
