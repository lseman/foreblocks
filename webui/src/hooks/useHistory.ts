
import { useCallback, useReducer } from "react";

type HistState<T> = { stack: T[]; index: number };
type Action<T> = { type: "PUSH"; payload: T } | { type: "UNDO" } | { type: "REDO" };

function historyReducer<T>(state: HistState<T>, action: Action<T>): HistState<T> {
  switch (action.type) {
    case "PUSH": {
      const next = state.stack.slice(0, state.index + 1);
      next.push(action.payload);
      return { stack: next, index: next.length - 1 };
    }
    case "UNDO": {
      if (state.index <= 0) return state;
      return { ...state, index: state.index - 1 };
    }
    case "REDO": {
      if (state.index >= state.stack.length - 1) return state;
      return { ...state, index: state.index + 1 };
    }
    default:
      return state;
  }
}

export function useHistory<T>(initial: T) {
  const [hist, dispatch] = useReducer(historyReducer<T>, { stack: [initial], index: 0 });
  const value = hist.stack[hist.index];
  const push = useCallback((v: T) => dispatch({ type: "PUSH", payload: v }), []);
  const undo = useCallback(() => dispatch({ type: "UNDO" }), []);
  const redo = useCallback(() => dispatch({ type: "REDO" }), []);
  return { value, push, undo, redo, canUndo: hist.index > 0, canRedo: hist.index < hist.stack.length - 1 };
}
