
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const uid = (p = "id") => `${p}_${Math.random().toString(36).slice(2, 9)}`;
export const sanitizeIdent = (s: string) => s.replace(/[^a-zA-Z0-9_]/g, "_");
export const varNameFor = (type: string, idx: number | string, varPrefix?: string) =>
  `${(varPrefix || type).replace(/[^a-zA-Z0-9_]/g, "_")}${idx ? idx : ""}`;
