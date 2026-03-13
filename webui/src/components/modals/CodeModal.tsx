
import React from "react";
import { X } from "lucide-react";

export const CodeModal: React.FC<{
  open: boolean;
  code: string;
  onClose: () => void;
}> = ({ open, code, onClose }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900/95 backdrop-blur-xl rounded-2xl shadow-2xl w-full max-w-5xl h-3/4 flex flex-col border border-slate-700/50">
        <div className="flex items-center justify-between p-6 border-b border-slate-700/50">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Generated Python Code</h2>
          <button onClick={onClose} className="hover:bg-slate-700/50 p-2 rounded-lg transition"><X size={24} /></button>
        </div>
        <div className="flex-1 overflow-auto p-6">
          <pre className="bg-slate-950/50 p-6 rounded-xl text-sm font-mono overflow-x-auto border border-slate-800/50">
            <code className="text-green-400">{code}</code>
          </pre>
        </div>
        <div className="p-6 border-t border-slate-700/50 flex gap-3 justify-end">
          <button
            onClick={() => { navigator.clipboard.writeText(code); alert('Code copied to clipboard!'); }}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-lg transition-all font-medium shadow-lg hover:shadow-blue-500/50"
          >
            📋 Copy to Clipboard
          </button>
          <button
            onClick={() => {
              const blob = new Blob([code], { type: 'text/plain' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url; a.download = 'model_config.py'; a.click();
            }}
            className="px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 rounded-lg transition-all font-medium shadow-lg hover:shadow-green-500/50"
          >
            💾 Download .py File
          </button>
        </div>
      </div>
    </div>
  );
};
