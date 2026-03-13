
import React from "react";
import { Download, Play, Plus, Settings, Trash2 } from "lucide-react";

export const Toolbar: React.FC<{
  onLoadTemplate: (name: string) => void;
  onToggleMenu: () => void;
  onRun: () => void;
  onViewCode: () => void;
  onExport: () => void;
  onAutoAlign: () => void;
  onClear: () => void;
  isExecuting: boolean;
  nodesCount: number;
  canRun?: boolean;
}> = ({ onLoadTemplate, onToggleMenu, onRun, onViewCode, onExport, onAutoAlign, onClear, isExecuting, nodesCount, canRun = true }) => (
  <div className="bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 p-3 flex items-center justify-between shadow-lg">
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
        <h1 className="text-lg font-bold">Time Series Model Builder</h1>
      </div>
    </div>
    <div className="flex gap-2">
      <div className="relative">
        <select
          onChange={(e) => { if (e.target.value) { onLoadTemplate(e.target.value); (e.target as HTMLSelectElement).value = ""; } }}
          className="px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-600/50 rounded-lg transition-all cursor-pointer"
          defaultValue=""
        >
          <option value="" disabled>📋 Templates</option>
          <option value="basic_transformer">Basic Transformer</option>
          <option value="with_heads">With HeadComposer</option>
        </select>
      </div>
      <button onClick={onToggleMenu} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-lg transition-all shadow-lg hover:shadow-blue-500/50">
        <Plus size={18} /> Add Node
      </button>
      <button onClick={onRun} disabled={isExecuting || !canRun} className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all shadow-lg font-medium ${isExecuting ? "bg-slate-600 cursor-not-allowed" : "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 hover:shadow-green-500/50"}`}>
        {isExecuting ? (<><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Executing...</>) : (<><Play size={18} /> Run Workflow</>)}
      </button>
      <button onClick={onViewCode} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-lg transition-all shadow-lg hover:shadow-blue-500/50">
        <Settings size={18} /> View Code
      </button>
      <button onClick={onExport} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-orange-600 to-amber-600 hover:from-orange-500 hover:to-amber-500 rounded-lg transition-all shadow-lg hover:shadow-orange-500/50" title="Export workflow as JSON">
        <Download size={18} /> Export
      </button>
      <button onClick={onAutoAlign} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-lg transition-all shadow-lg hover:shadow-purple-500/50" title="Auto-align nodes">
        <Settings size={18} style={{ transform: "rotate(0deg)" }} /> Auto Align
      </button>
      <button onClick={onClear} className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 rounded-lg transition-all shadow-lg hover:shadow-red-500/50" title="Clear canvas">
        <Trash2 size={18} /> Clear
      </button>
    </div>
  </div>
);
