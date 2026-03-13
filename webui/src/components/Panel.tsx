
import React from "react";

export const Panel: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className = "", children }) => (
  <div className={`bg-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl ${className}`}>{children}</div>
);

export const Kbd: React.FC<React.PropsWithChildren<{}>> = ({ children }) => (
  <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700/60">{children}</kbd>
);
