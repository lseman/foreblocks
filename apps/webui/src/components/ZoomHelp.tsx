
import React from "react";
import { Plus, Settings } from "lucide-react";
import { Panel, Kbd } from "./Panel";

export const ZoomControls: React.FC<{
  scale: number;
  onIn: () => void;
  onReset: () => void;
  onOut: () => void;
}> = ({ scale, onIn, onReset, onOut }) => (
  <Panel className="absolute bottom-6 right-6 p-2 shadow-xl">
    <button onClick={onIn} className="p-2 hover:bg-slate-700/50 rounded-lg transition" title="Zoom In">
      <Plus size={20} />
    </button>
    <button onClick={onReset} className="p-2 hover:bg-slate-700/50 rounded-lg transition text-xs font-bold" title="Reset Zoom">
      {Math.round(scale * 100)}%
    </button>
    <button onClick={onOut} className="p-2 hover:bg-slate-700/50 rounded-lg transition" title="Zoom Out">
      <Settings size={20} style={{ transform: "rotate(90deg)" }} />
    </button>
  </Panel>
);

export const ShortcutsHelp: React.FC = () => (
  <Panel className="absolute top-6 right-6 p-3 shadow-2xl text-xs max-w-xs">
    <div className="font-semibold mb-2 text-slate-300">⌨️ Keyboard Shortcuts</div>
    <div className="space-y-1 text-slate-400">
      <div><Kbd>Ctrl+Z</Kbd> Undo</div>
      <div><Kbd>Ctrl+Y</Kbd>/<Kbd>Shift+Ctrl+Z</Kbd> Redo</div>
      <div><Kbd>Ctrl+C</Kbd>/<Kbd>Ctrl+V</Kbd> Copy/Paste</div>
      <div><Kbd>Ctrl+D</Kbd> Duplicate</div>
      <div><Kbd>Del</Kbd> Delete</div>
      <div><Kbd>Ctrl</Kbd>+Drag Box Select</div>
      <div>Drag Canvas Pan</div>
      <div>Scroll Zoom</div>
      <div><Kbd>Ctrl+E</Kbd> Execute</div>
    </div>
  </Panel>
);
