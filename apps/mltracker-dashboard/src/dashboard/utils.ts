import type { Run } from '../types';
import {
  DEFAULT_MODULE_LAYOUT,
  type DashboardModuleId,
  type DashboardModuleLayoutItem,
  type DashboardModuleSpan,
  type GroupMode,
  type ParsedUrlState,
  type RunGroup,
  type SavedView,
  STATUS_ORDER,
  type SortMode,
  type SweepAxisMode,
  type ViewStateSnapshot,
} from './types';

const STORAGE_VIEW_PREFS_PREFIX = 'mltracker_view_prefs_';
const STORAGE_SAVED_VIEWS_PREFIX = 'mltracker_saved_views_';
const STORAGE_MODULE_LAYOUT_PREFIX = 'mltracker_module_layout_';

export function isDashboardModuleId(value: string): value is DashboardModuleId {
  return ['leaderboard', 'sweep', 'parallel', 'importance', 'compare'].includes(value);
}

export function normalizeModuleSpan(value: unknown): DashboardModuleSpan {
  return value === 2 || value === 3 ? value : 1;
}

export function normalizeModuleLayout(value: unknown): DashboardModuleLayoutItem[] {
  const items = Array.isArray(value) ? value : [];
  const next: DashboardModuleLayoutItem[] = [];
  const seen = new Set<DashboardModuleId>();

  for (const item of items) {
    if (!item || typeof item !== 'object') continue;
    const id = (item as { id?: unknown }).id;
    if (typeof id !== 'string' || !isDashboardModuleId(id) || seen.has(id)) continue;
    next.push({
      id,
      span: normalizeModuleSpan((item as { span?: unknown }).span),
    });
    seen.add(id);
  }

  for (const item of DEFAULT_MODULE_LAYOUT) {
    if (seen.has(item.id)) continue;
    next.push(item);
  }

  return next;
}

export function parseTime(iso?: string): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  return Number.isFinite(t) ? t : 0;
}

export function formatDuration(seconds?: number): string {
  if (!seconds || seconds <= 0) return '-';
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) return `${mins}m ${secs}s`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ${mins % 60}m`;
}

export function formatMetricValue(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  }
  return String(value ?? '-');
}

export function normalizeText(value: unknown): string {
  return String(value ?? '').trim().toLowerCase();
}

export function runMatchesTagFilter(run: Run, tagFilter: string): boolean {
  const query = normalizeText(tagFilter);
  if (!query) return true;
  const tags = run.tags ?? {};
  const entries = Object.entries(tags);

  if (query.includes(':')) {
    const [rawKey, rawVal = ''] = query.split(':', 2);
    const keyQuery = normalizeText(rawKey);
    const valQuery = normalizeText(rawVal);
    if (!keyQuery) return true;

    for (const [k, v] of entries) {
      const keyNorm = normalizeText(k);
      if (!keyNorm.includes(keyQuery)) continue;
      if (!valQuery || normalizeText(v).includes(valQuery)) return true;
    }
    return false;
  }

  return entries.some(([k, v]) => normalizeText(k).includes(query) || normalizeText(v).includes(query));
}

export function sortRuns(runs: Run[], mode: SortMode): Run[] {
  const data = [...runs];
  const duration = (r: Run) => (Number.isFinite(r.duration) ? Number(r.duration) : -1);
  switch (mode) {
    case 'start_time':
      return data.sort((a, b) => parseTime(a.start_time) - parseTime(b.start_time));
    case '-duration':
      return data.sort((a, b) => duration(b) - duration(a));
    case 'duration':
      return data.sort((a, b) => duration(a) - duration(b));
    case '-start_time':
    default:
      return data.sort((a, b) => parseTime(b.start_time) - parseTime(a.start_time));
  }
}

export function groupRuns(runs: Run[], mode: GroupMode): RunGroup[] {
  if (mode === 'none') return [{ label: null, runs }];
  if (mode === 'status') {
    const buckets = new Map<string, Run[]>();
    for (const run of runs) {
      const key = run.status ?? 'UNKNOWN';
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key)!.push(run);
    }
    const known = STATUS_ORDER.filter((x) => buckets.has(x));
    const extra = [...buckets.keys()].filter((x) => !STATUS_ORDER.includes(x)).sort();
    return [...known, ...extra].map((label) => ({ label, runs: buckets.get(label) ?? [] }));
  }

  const buckets = new Map<string, Run[]>();
  for (const run of runs) {
    const key = run.start_time ? String(run.start_time).slice(0, 10) : 'Unknown day';
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(run);
  }
  return [...buckets.keys()]
    .sort((a, b) => b.localeCompare(a))
    .map((label) => ({ label, runs: buckets.get(label) ?? [] }));
}

export function extractNumericMetricKeys(runs: Run[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const [key, value] of Object.entries(run.metrics ?? {})) {
      if (typeof value === 'number' && Number.isFinite(value)) keys.add(key);
    }
  }
  return [...keys].sort();
}

export function extractNumericParamKeys(runs: Run[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const [key, value] of Object.entries(run.params ?? {})) {
      const num = Number(value);
      if (Number.isFinite(num)) keys.add(key);
    }
  }
  return [...keys].sort();
}

export function pickPrimaryMetricKey(runs: Run[]): string | null {
  const counts = new Map<string, number>();
  for (const run of runs) {
    for (const [key, value] of Object.entries(run.metrics ?? {})) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
  }

  if (!counts.size) return null;

  const preference = [
    'val_loss',
    'valid_loss',
    'loss',
    'val_rmse',
    'rmse',
    'val_mae',
    'mae',
    'mse',
    'accuracy',
    'acc',
    'f1',
    'auc',
  ];

  const rank = (key: string) => {
    const idx = preference.indexOf(key.toLowerCase());
    return idx === -1 ? 999 : idx;
  };

  return [...counts.entries()]
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return rank(a[0]) - rank(b[0]);
    })[0][0];
}

export function metricLowerIsBetter(metricKey: string): boolean {
  return /(loss|error|rmse|mae|mse|mape|nll|perplexity|wer|cer)/i.test(metricKey);
}

export function formatCompactDate(iso?: string): string {
  if (!iso) return '-';
  try {
    return new Date(iso).toLocaleString([], {
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export function metricValueForRun(run: Run, metricKey: string): number | null {
  if (!metricKey) return null;
  const value = run.metrics?.[metricKey];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function collectParallelDimensions(runs: Run[]): string[] {
  const dims = extractNumericMetricKeys(runs);
  const hasDuration = runs.some((run) => Number.isFinite(run.duration));
  return hasDuration ? ['__duration__', ...dims] : dims;
}

export function parallelDimensionValue(run: Run, key: string): number | null {
  if (key === '__duration__') {
    return Number.isFinite(run.duration) ? Number(run.duration) : null;
  }
  return metricValueForRun(run, key);
}

export function dimensionLabel(key: string): string {
  return key === '__duration__' ? 'duration' : key;
}

export function defaultParallelDimensions(availableDims: string[], primaryMetric: string | null): string[] {
  if (!availableDims.length) return [];
  const chosen: string[] = [];
  const push = (value: string | null | undefined) => {
    if (!value || !availableDims.includes(value) || chosen.includes(value)) return;
    chosen.push(value);
  };

  push('__duration__');
  push(primaryMetric);
  for (const dim of availableDims) {
    if (chosen.length >= 4) break;
    push(dim);
  }
  return chosen.slice(0, 4);
}

export function pearsonCorrelation(xs: number[], ys: number[]): number {
  if (xs.length < 2 || ys.length < 2 || xs.length !== ys.length) return 0;
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  if (denX <= 0 || denY <= 0) return 0;
  return num / Math.sqrt(denX * denY);
}

export function getSweepValue(run: Run, axis: SweepAxisMode): number | null {
  if (!axis) return null;
  if (axis.startsWith('metric:')) {
    const key = axis.slice('metric:'.length);
    const value = run.metrics?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  const key = axis.slice('param:'.length);
  const value = Number(run.params?.[key]);
  return Number.isFinite(value) ? value : null;
}

export function colorForIndex(index: number): string {
  const palette = ['#3bc8ff', '#47d8a6', '#f7b541', '#ff8d5f', '#b7ff6a', '#ff79b4', '#8ea4ff'];
  return palette[index % palette.length];
}

export function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function sweepAxisLabel(axis: SweepAxisMode): string {
  if (!axis) return '-';
  if (axis.startsWith('metric:')) return axis.slice('metric:'.length);
  if (axis.startsWith('param:')) return axis.slice('param:'.length);
  return axis;
}

export function snapshotToUrl(exp: string, state: ViewStateSnapshot): void {
  const url = new URL(window.location.href);
  url.hash = `#exp/${encodeURIComponent(exp)}`;
  const p = url.searchParams;
  const set = (k: string, v: string | boolean | undefined) => {
    if (!v || v === 'ALL') p.delete(k);
    else p.set(k, String(v));
  };
  set('exp', exp);
  set('q', state.runQuery);
  set('sort', state.sortMode);
  set('group', state.groupMode);
  set('status', state.activeStatus);
  set('tag', state.tagFilter);
  set('mkey', state.metricFilterKey);
  set('mmin', state.metricMin);
  set('mmax', state.metricMax);
  set('only_selected', state.onlySelected ? '1' : '');
  set('selected', state.selectedRunIds.slice(0, 80).join(','));
  window.history.replaceState({}, '', `${url.pathname}${p.toString() ? `?${p.toString()}` : ''}${url.hash}`);
}

function viewPrefsKey(experimentName: string): string {
  return `${STORAGE_VIEW_PREFS_PREFIX}${experimentName}`;
}

function savedViewsKey(experimentName: string): string {
  return `${STORAGE_SAVED_VIEWS_PREFIX}${experimentName}`;
}

function moduleLayoutKey(experimentName: string): string {
  return `${STORAGE_MODULE_LAYOUT_PREFIX}${experimentName}`;
}

export function loadViewPrefs(experimentName: string): Partial<ViewStateSnapshot> {
  try {
    const raw = localStorage.getItem(viewPrefsKey(experimentName));
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function saveViewPrefs(experimentName: string, state: ViewStateSnapshot): void {
  try {
    localStorage.setItem(viewPrefsKey(experimentName), JSON.stringify(state));
  } catch {
    // no-op
  }
}

export function loadSavedViews(experimentName: string): SavedView[] {
  try {
    const raw = localStorage.getItem(savedViewsKey(experimentName));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x) => x && typeof x.name === 'string' && x.state).slice(0, 25);
  } catch {
    return [];
  }
}

export function saveSavedViews(experimentName: string, views: SavedView[]): void {
  try {
    localStorage.setItem(savedViewsKey(experimentName), JSON.stringify(views.slice(0, 25)));
  } catch {
    // no-op
  }
}

export function loadModuleLayout(experimentName: string): DashboardModuleLayoutItem[] {
  try {
    const raw = localStorage.getItem(moduleLayoutKey(experimentName));
    if (!raw) return DEFAULT_MODULE_LAYOUT;
    return normalizeModuleLayout(JSON.parse(raw));
  } catch {
    return DEFAULT_MODULE_LAYOUT;
  }
}

export function saveModuleLayout(experimentName: string, layout: DashboardModuleLayoutItem[]): void {
  try {
    localStorage.setItem(moduleLayoutKey(experimentName), JSON.stringify(normalizeModuleLayout(layout)));
  } catch {
    // no-op
  }
}

export function parseUrlState(): ParsedUrlState | null {
  if (typeof window === 'undefined') return null;

  const url = new URL(window.location.href);
  const hashMatch = url.hash.match(/^#exp\/(.+)$/);
  const experimentName =
    url.searchParams.get('exp') || (hashMatch ? decodeURIComponent(hashMatch[1]) : '');
  if (!experimentName) return null;

  const selected = url.searchParams
    .get('selected')
    ?.split(',')
    .map((x) => x.trim())
    .filter(Boolean) ?? [];

  return {
    experimentName,
    state: {
      runQuery: url.searchParams.get('q') ?? '',
      sortMode: (url.searchParams.get('sort') as SortMode) ?? '-start_time',
      groupMode: (url.searchParams.get('group') as GroupMode) ?? 'none',
      activeStatus: url.searchParams.get('status') ?? 'ALL',
      tagFilter: url.searchParams.get('tag') ?? '',
      metricFilterKey: url.searchParams.get('mkey') ?? '',
      metricMin: url.searchParams.get('mmin') ?? '',
      metricMax: url.searchParams.get('mmax') ?? '',
      onlySelected: url.searchParams.get('only_selected') === '1',
      selectedRunIds: selected.slice(0, 80),
    },
  };
}
