import { Fragment, useEffect, useMemo, useState } from 'react';
import {
  artifactDownloadUrl,
  compareRuns,
  deleteRun,
  getArtifacts,
  getExperiments,
  getHealth,
  getMetricHistory,
  getRunDetail,
  getRuns,
  searchRuns,
} from './api/client';
import { KpiCard } from './components/KpiCard';
import type {
  Artifact,
  CompareResponse,
  Experiment,
  MetricHistoryPoint,
  MetricHistoryResponse,
  Run,
} from './types';

const STORAGE_THEME_KEY = 'mltracker_theme';
const STORAGE_VIEW_PREFS_PREFIX = 'mltracker_view_prefs_';
const STORAGE_SAVED_VIEWS_PREFIX = 'mltracker_saved_views_';
const STORAGE_MODULE_LAYOUT_PREFIX = 'mltracker_module_layout_';
const STATUS_ORDER = ['RUNNING', 'FINISHED', 'FAILED', 'CANCELED'];

type SortMode = '-start_time' | 'start_time' | '-duration' | 'duration';
type GroupMode = 'none' | 'status' | 'day';
type SweepAxisMode = '' | `metric:${string}` | `param:${string}`;
type DashboardModuleId = 'leaderboard' | 'sweep' | 'parallel' | 'importance' | 'compare';
type DashboardModuleSpan = 1 | 2 | 3;

type ViewStateSnapshot = {
  runQuery: string;
  sortMode: SortMode;
  groupMode: GroupMode;
  activeStatus: string;
  tagFilter: string;
  metricFilterKey: string;
  metricMin: string;
  metricMax: string;
  onlySelected: boolean;
  selectedRunIds: string[];
};

type SavedView = {
  name: string;
  createdAt: string;
  state: ViewStateSnapshot;
};

type RunGroup = {
  label: string | null;
  runs: Run[];
};

type PaletteAction = {
  title: string;
  meta?: string;
  run: () => void | Promise<void>;
};

type ParsedUrlState = {
  experimentName: string;
  state: Partial<ViewStateSnapshot>;
};

type DashboardModuleLayoutItem = {
  id: DashboardModuleId;
  span: DashboardModuleSpan;
};

type DashboardModuleDefinition = {
  id: DashboardModuleId;
  title: string;
  controls?: JSX.Element | null;
  body: JSX.Element;
  visible: boolean;
};

type HoveredParallelRun = {
  run: Run;
  x: number;
  y: number;
  values: Array<{ dim: string; value: number }>;
};

type HoveredSweepPoint = {
  run: Run;
  x: number;
  y: number;
  plotX: number;
  plotY: number;
  valueX: number;
  valueY: number;
};

const DEFAULT_MODULE_LAYOUT: DashboardModuleLayoutItem[] = [
  { id: 'leaderboard', span: 1 },
  { id: 'sweep', span: 2 },
  { id: 'parallel', span: 3 },
  { id: 'importance', span: 3 },
  { id: 'compare', span: 3 },
];

function isDashboardModuleId(value: string): value is DashboardModuleId {
  return ['leaderboard', 'sweep', 'parallel', 'importance', 'compare'].includes(value);
}

function normalizeModuleSpan(value: unknown): DashboardModuleSpan {
  return value === 2 || value === 3 ? value : 1;
}

function normalizeModuleLayout(value: unknown): DashboardModuleLayoutItem[] {
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

function parseTime(iso?: string): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  return Number.isFinite(t) ? t : 0;
}

function formatDuration(seconds?: number): string {
  if (!seconds || seconds <= 0) return '-';
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) return `${mins}m ${secs}s`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ${mins % 60}m`;
}

function formatMetricValue(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
      return value.toExponential(2);
    }
    return value.toFixed(4);
  }
  return String(value ?? '-');
}

function normalizeText(value: unknown): string {
  return String(value ?? '').trim().toLowerCase();
}

function runMatchesTagFilter(run: Run, tagFilter: string): boolean {
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

function sortRuns(runs: Run[], mode: SortMode): Run[] {
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

function groupRuns(runs: Run[], mode: GroupMode): RunGroup[] {
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

function extractNumericMetricKeys(runs: Run[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const [key, value] of Object.entries(run.metrics ?? {})) {
      if (typeof value === 'number' && Number.isFinite(value)) keys.add(key);
    }
  }
  return [...keys].sort();
}

function extractNumericParamKeys(runs: Run[]): string[] {
  const keys = new Set<string>();
  for (const run of runs) {
    for (const [key, value] of Object.entries(run.params ?? {})) {
      const num = Number(value);
      if (Number.isFinite(num)) keys.add(key);
    }
  }
  return [...keys].sort();
}

function pickPrimaryMetricKey(runs: Run[]): string | null {
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

function metricLowerIsBetter(metricKey: string): boolean {
  return /(loss|error|rmse|mae|mse|mape|nll|perplexity|wer|cer)/i.test(metricKey);
}

function formatCompactDate(iso?: string): string {
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

function metricValueForRun(run: Run, metricKey: string): number | null {
  if (!metricKey) return null;
  const value = run.metrics?.[metricKey];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function collectParallelDimensions(runs: Run[]): string[] {
  const dims = extractNumericMetricKeys(runs);
  const hasDuration = runs.some((run) => Number.isFinite(run.duration));
  return hasDuration ? ['__duration__', ...dims] : dims;
}

function parallelDimensionValue(run: Run, key: string): number | null {
  if (key === '__duration__') {
    return Number.isFinite(run.duration) ? Number(run.duration) : null;
  }
  return metricValueForRun(run, key);
}

function dimensionLabel(key: string): string {
  return key === '__duration__' ? 'duration' : key;
}

function defaultParallelDimensions(availableDims: string[], primaryMetric: string | null): string[] {
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

function pearsonCorrelation(xs: number[], ys: number[]): number {
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

function getSweepValue(run: Run, axis: SweepAxisMode): number | null {
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

function colorForIndex(index: number): string {
  const palette = ['#3bc8ff', '#47d8a6', '#f7b541', '#ff8d5f', '#b7ff6a', '#ff79b4', '#8ea4ff'];
  return palette[index % palette.length];
}

function sweepAxisLabel(axis: SweepAxisMode): string {
  if (!axis) return '-';
  if (axis.startsWith('metric:')) return axis.slice('metric:'.length);
  if (axis.startsWith('param:')) return axis.slice('param:'.length);
  return axis;
}

function snapshotToUrl(exp: string, state: ViewStateSnapshot): void {
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

function loadViewPrefs(experimentName: string): Partial<ViewStateSnapshot> {
  try {
    const raw = localStorage.getItem(viewPrefsKey(experimentName));
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function saveViewPrefs(experimentName: string, state: ViewStateSnapshot): void {
  try {
    localStorage.setItem(viewPrefsKey(experimentName), JSON.stringify(state));
  } catch {
    // no-op
  }
}

function loadSavedViews(experimentName: string): SavedView[] {
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

function saveSavedViews(experimentName: string, views: SavedView[]): void {
  try {
    localStorage.setItem(savedViewsKey(experimentName), JSON.stringify(views.slice(0, 25)));
  } catch {
    // no-op
  }
}

function loadModuleLayout(experimentName: string): DashboardModuleLayoutItem[] {
  try {
    const raw = localStorage.getItem(moduleLayoutKey(experimentName));
    if (!raw) return DEFAULT_MODULE_LAYOUT;
    return normalizeModuleLayout(JSON.parse(raw));
  } catch {
    return DEFAULT_MODULE_LAYOUT;
  }
}

function saveModuleLayout(experimentName: string, layout: DashboardModuleLayoutItem[]): void {
  try {
    localStorage.setItem(moduleLayoutKey(experimentName), JSON.stringify(normalizeModuleLayout(layout)));
  } catch {
    // no-op
  }
}

function parseUrlState(): ParsedUrlState | null {
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

export function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem(STORAGE_THEME_KEY);
    return saved === 'light' ? 'light' : 'dark';
  });
  const [health, setHealth] = useState<string>('checking');
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [experimentFilter, setExperimentFilter] = useState('');
  const [activeExperiment, setActiveExperiment] = useState<string>('');
  const [runsRaw, setRunsRaw] = useState<Run[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const [runQuery, setRunQuery] = useState('');
  const [sortMode, setSortMode] = useState<SortMode>('-start_time');
  const [groupMode, setGroupMode] = useState<GroupMode>('none');
  const [activeStatus, setActiveStatus] = useState('ALL');
  const [tagFilter, setTagFilter] = useState('');
  const [metricFilterKey, setMetricFilterKey] = useState('');
  const [metricMin, setMetricMin] = useState('');
  const [metricMax, setMetricMax] = useState('');
  const [onlySelected, setOnlySelected] = useState(false);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);

  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [compareMetric, setCompareMetric] = useState('');
  const [compareHistory, setCompareHistory] = useState<Record<string, MetricHistoryPoint[]>>({});
  const [compareHistoryLoading, setCompareHistoryLoading] = useState(false);

  const [sweepXAxis, setSweepXAxis] = useState<SweepAxisMode>('');
  const [sweepYAxis, setSweepYAxis] = useState<SweepAxisMode>('');
  const [leaderboardMetric, setLeaderboardMetric] = useState('');
  const [leaderboardObjective, setLeaderboardObjective] = useState<'auto' | 'min' | 'max'>('auto');
  const [parallelDims, setParallelDims] = useState<string[]>([]);
  const [importanceTarget, setImportanceTarget] = useState('');
  const [moduleLayout, setModuleLayout] = useState<DashboardModuleLayoutItem[]>(DEFAULT_MODULE_LAYOUT);
  const [draggedModuleId, setDraggedModuleId] = useState<DashboardModuleId | null>(null);
  const [dropTargetModuleId, setDropTargetModuleId] = useState<DashboardModuleId | null>(null);
  const [hoveredParallelRun, setHoveredParallelRun] = useState<HoveredParallelRun | null>(null);
  const [hoveredSweepPoint, setHoveredSweepPoint] = useState<HoveredSweepPoint | null>(null);

  const [runDetailOpen, setRunDetailOpen] = useState(false);
  const [runDetail, setRunDetail] = useState<Run | null>(null);
  const [runDetailMetrics, setRunDetailMetrics] = useState<MetricHistoryResponse | null>(null);
  const [runDetailArtifacts, setRunDetailArtifacts] = useState<Artifact[]>([]);
  const [runDetailMetricKey, setRunDetailMetricKey] = useState('');

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState('');

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem(STORAGE_THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    void (async () => {
      try {
        const [healthData, experimentsData] = await Promise.all([
          getHealth(),
          getExperiments(),
        ]);
        setHealth(healthData.status ?? 'unknown');
        setExperiments(experimentsData);
        if (experimentsData.length > 0) {
          const parsed = parseUrlState();
          const requested = parsed?.experimentName ?? '';
          const match = experimentsData.find((exp) => exp.name === requested);
          setActiveExperiment(match?.name ?? experimentsData[0].name);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load API data');
        setHealth('offline');
      }
    })();
  }, [refreshTick]);

  useEffect(() => {
    if (!activeExperiment) return;
    const prefs = loadViewPrefs(activeExperiment);
    const parsed = parseUrlState();
    const urlState = parsed?.experimentName === activeExperiment ? parsed.state : {};
    const source = { ...prefs, ...urlState };
    setSortMode((source.sortMode as SortMode) ?? '-start_time');
    setGroupMode((source.groupMode as GroupMode) ?? 'none');
    setActiveStatus((source.activeStatus as string) ?? 'ALL');
    setTagFilter((source.tagFilter as string) ?? '');
    setMetricFilterKey((source.metricFilterKey as string) ?? '');
    setMetricMin((source.metricMin as string) ?? '');
    setMetricMax((source.metricMax as string) ?? '');
    setOnlySelected(Boolean(source.onlySelected));
    setRunQuery((source.runQuery as string) ?? '');
    setSelectedRunIds(
      Array.isArray(source.selectedRunIds) ? source.selectedRunIds.slice(0, 80) : []
    );
    setCompareData(null);
    setCompareHistory({});
    setSweepXAxis('');
    setSweepYAxis('');
    setLeaderboardMetric('');
    setLeaderboardObjective('auto');
    setParallelDims([]);
    setImportanceTarget('');
    setModuleLayout(loadModuleLayout(activeExperiment));
    setDraggedModuleId(null);
    setDropTargetModuleId(null);
  }, [activeExperiment]);

  useEffect(() => {
    if (!activeExperiment) return;
    setLoadingRuns(true);
    setError(null);
    void (async () => {
      try {
        const data = runQuery.trim()
          ? await searchRuns(activeExperiment, runQuery.trim())
          : await getRuns(activeExperiment);
        setRunsRaw(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load runs');
      } finally {
        setLoadingRuns(false);
      }
    })();
  }, [activeExperiment, runQuery, refreshTick]);

  const sortedRuns = useMemo(() => sortRuns(runsRaw, sortMode), [runsRaw, sortMode]);

  const statusCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const run of sortedRuns) {
      const status = run.status ?? 'UNKNOWN';
      counts.set(status, (counts.get(status) ?? 0) + 1);
    }
    return counts;
  }, [sortedRuns]);

  const statusOptions = useMemo(() => {
    const known = STATUS_ORDER.filter((x) => statusCounts.has(x));
    const extra = [...statusCounts.keys()].filter((x) => !STATUS_ORDER.includes(x)).sort();
    return ['ALL', ...known, ...extra];
  }, [statusCounts]);

  const metricKeys = useMemo(() => extractNumericMetricKeys(sortedRuns), [sortedRuns]);
  const paramKeys = useMemo(() => extractNumericParamKeys(sortedRuns), [sortedRuns]);

  const sweepAxisOptions = useMemo(
    () => [
      ...metricKeys.map((k) => ({ value: `metric:${k}` as SweepAxisMode, label: `metric: ${k}` })),
      ...paramKeys.map((k) => ({ value: `param:${k}` as SweepAxisMode, label: `param: ${k}` })),
    ],
    [metricKeys, paramKeys]
  );

  const selectedSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);

  const filteredRuns = useMemo(() => {
    const min = metricMin.trim() ? Number(metricMin) : null;
    const max = metricMax.trim() ? Number(metricMax) : null;
    const useMetric = Boolean(metricFilterKey) && (Number.isFinite(min) || Number.isFinite(max));

    return sortedRuns.filter((run) => {
      if (activeStatus !== 'ALL' && (run.status ?? 'UNKNOWN') !== activeStatus) return false;
      if (onlySelected && !selectedSet.has(run.run_id)) return false;
      if (!runMatchesTagFilter(run, tagFilter)) return false;
      if (!useMetric) return true;

      const value = run.metrics?.[metricFilterKey];
      if (typeof value !== 'number' || !Number.isFinite(value)) return false;
      if (Number.isFinite(min) && value < Number(min)) return false;
      if (Number.isFinite(max) && value > Number(max)) return false;
      return true;
    });
  }, [sortedRuns, activeStatus, onlySelected, selectedSet, tagFilter, metricFilterKey, metricMin, metricMax]);

  const groupedRuns = useMemo(() => groupRuns(filteredRuns, groupMode), [filteredRuns, groupMode]);

  const visibleExperiments = useMemo(() => {
    const q = experimentFilter.trim().toLowerCase();
    if (!q) return experiments;
    return experiments.filter((x) => String(x.name ?? '').toLowerCase().includes(q));
  }, [experiments, experimentFilter]);

  const finishedRuns = useMemo(
    () => filteredRuns.filter((r) => (r.status ?? '').toUpperCase() === 'FINISHED').length,
    [filteredRuns]
  );

  const runningRuns = useMemo(
    () => filteredRuns.filter((r) => (r.status ?? '').toUpperCase() === 'RUNNING').length,
    [filteredRuns]
  );

  const failedRuns = useMemo(
    () => filteredRuns.filter((r) => (r.status ?? '').toUpperCase() === 'FAILED').length,
    [filteredRuns]
  );

  useEffect(() => {
    if (!sweepYAxis && metricKeys.length > 0) {
      setSweepYAxis(`metric:${metricKeys[0]}`);
    }
    if (!sweepXAxis) {
      if (paramKeys.length > 0) {
        setSweepXAxis(`param:${paramKeys[0]}`);
      } else if (metricKeys.length > 1) {
        setSweepXAxis(`metric:${metricKeys[1]}`);
      }
    }
  }, [metricKeys, paramKeys, sweepXAxis, sweepYAxis]);

  const primaryMetric = useMemo(() => pickPrimaryMetricKey(filteredRuns), [filteredRuns]);
  const avgDuration = useMemo(() => {
    const durations = filteredRuns
      .map((run) => (Number.isFinite(run.duration) ? Number(run.duration) : 0))
      .filter((value) => value > 0);
    if (!durations.length) return 0;
    return Math.round(durations.reduce((a, b) => a + b, 0) / durations.length);
  }, [filteredRuns]);

  const latestRun = useMemo(
    () => [...filteredRuns].sort((a, b) => parseTime(b.start_time) - parseTime(a.start_time))[0] ?? null,
    [filteredRuns]
  );

  const last24hRuns = useMemo(
    () =>
      filteredRuns.filter((run) => {
        const t = parseTime(run.start_time);
        return t > 0 && t >= Date.now() - 24 * 60 * 60 * 1000;
      }).length,
    [filteredRuns]
  );

  const successRate = useMemo(() => {
    if (!filteredRuns.length) return 0;
    return (finishedRuns / filteredRuns.length) * 100;
  }, [filteredRuns.length, finishedRuns]);

  useEffect(() => {
    if (!leaderboardMetric && primaryMetric) {
      setLeaderboardMetric(primaryMetric);
    }
  }, [leaderboardMetric, primaryMetric]);

  const leaderboardRows = useMemo(() => {
    if (!leaderboardMetric) return [] as Array<{ run: Run; value: number }>;
    const objective =
      leaderboardObjective === 'auto'
        ? metricLowerIsBetter(leaderboardMetric)
          ? 'min'
          : 'max'
        : leaderboardObjective;

    return filteredRuns
      .map((run) => ({ run, value: metricValueForRun(run, leaderboardMetric) }))
      .filter((entry): entry is { run: Run; value: number } => entry.value !== null)
      .sort((a, b) => (objective === 'min' ? a.value - b.value : b.value - a.value))
      .slice(0, 8);
  }, [filteredRuns, leaderboardMetric, leaderboardObjective]);

  const parallelAvailableDims = useMemo(() => collectParallelDimensions(filteredRuns), [filteredRuns]);

  useEffect(() => {
    if (!parallelAvailableDims.length) {
      if (parallelDims.length) setParallelDims([]);
      return;
    }
    const safe = parallelDims.filter((dim) => parallelAvailableDims.includes(dim));
    if (!safe.length) {
      setParallelDims(defaultParallelDimensions(parallelAvailableDims, primaryMetric));
    } else if (safe.length !== parallelDims.length) {
      setParallelDims(safe.slice(0, 6));
    }
  }, [parallelAvailableDims, parallelDims, primaryMetric]);

  const parallelSeries = useMemo(() => {
    const dims = parallelDims.filter((dim) => parallelAvailableDims.includes(dim)).slice(0, 6);
    if (dims.length < 2) {
      return {
        dims,
        ranges: [] as Array<{ dim: string; min: number; max: number }>,
        rows: [] as Array<{ run: Run; values: number[]; points: string; pointEntries: Array<{ x: number; y: number; dim: string; value: number }> }>,
      };
    }

    const ranges = dims.map((dim) => {
      const values = filteredRuns
        .map((run) => parallelDimensionValue(run, dim))
        .filter((value): value is number => value !== null);
      const min = values.length ? Math.min(...values) : 0;
      const max = values.length ? Math.max(...values) : 1;
      return {
        dim,
        min,
        max: min === max ? max + 1 : max,
      };
    });

    const rows = filteredRuns
      .slice(0, 80)
      .map((run) => {
        const values = dims.map((dim) => parallelDimensionValue(run, dim));
        if (values.some((value) => value === null)) return null;
        const numericValues = values as number[];
        const pointEntries = numericValues.map((value, idx) => {
          const range = ranges[idx];
          const x = 70 + (idx / Math.max(1, ranges.length - 1)) * 620;
          const y = 252 - ((value - range.min) / (range.max - range.min)) * 216;
          return {
            x,
            y,
            dim: range.dim,
            value,
          };
        });
        return {
          run,
          values: numericValues,
          pointEntries,
          points: pointEntries.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' '),
        };
      })
      .filter((entry): entry is { run: Run; values: number[]; points: string; pointEntries: Array<{ x: number; y: number; dim: string; value: number }> } => entry !== null);

    return { dims, ranges, rows };
  }, [filteredRuns, parallelDims, parallelAvailableDims]);

  useEffect(() => {
    if (!hoveredParallelRun) return;
    const stillPresent = parallelSeries.rows.some((entry) => entry.run.run_id === hoveredParallelRun.run.run_id);
    if (!stillPresent) {
      setHoveredParallelRun(null);
    }
  }, [hoveredParallelRun, parallelSeries.rows]);

  useEffect(() => {
    if (!importanceTarget && primaryMetric) {
      setImportanceTarget(primaryMetric);
    }
  }, [importanceTarget, primaryMetric]);

  const importanceRows = useMemo(() => {
    if (!importanceTarget) return [] as Array<{ key: string; score: number; direction: number; samples: number }>;

    const rows: Array<{ key: string; score: number; direction: number; samples: number }> = [];
    for (const key of paramKeys) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const run of filteredRuns) {
        const x = Number(run.params?.[key]);
        const y = metricValueForRun(run, importanceTarget);
        if (!Number.isFinite(x) || y === null) continue;
        xs.push(x);
        ys.push(y);
      }
      if (xs.length < 3) continue;
      const corr = pearsonCorrelation(xs, ys);
      rows.push({
        key,
        score: Math.abs(corr),
        direction: corr,
        samples: xs.length,
      });
    }
    return rows.sort((a, b) => b.score - a.score).slice(0, 8);
  }, [filteredRuns, paramKeys, importanceTarget]);

  useEffect(() => {
    if (!activeExperiment) return;
    const snapshot: ViewStateSnapshot = {
      runQuery,
      sortMode,
      groupMode,
      activeStatus,
      tagFilter,
      metricFilterKey,
      metricMin,
      metricMax,
      onlySelected,
      selectedRunIds,
    };
    saveViewPrefs(activeExperiment, snapshot);
    snapshotToUrl(activeExperiment, snapshot);
  }, [
    activeExperiment,
    runQuery,
    sortMode,
    groupMode,
    activeStatus,
    tagFilter,
    metricFilterKey,
    metricMin,
    metricMax,
    onlySelected,
    selectedRunIds,
  ]);

  useEffect(() => {
    if (!activeExperiment) return;
    saveModuleLayout(activeExperiment, moduleLayout);
  }, [activeExperiment, moduleLayout]);

  useEffect(() => {
    if (!compareData || !compareMetric) {
      setCompareHistory({});
      return;
    }

    const runs = compareData.runs.slice(0, 8);
    setCompareHistoryLoading(true);
    void (async () => {
      try {
        const entries = await Promise.all(
          runs.map(async (run) => {
            const history = await getMetricHistory(run.run_id, compareMetric);
            const points = history.metrics?.[compareMetric] ?? [];
            return [run.run_id, points] as const;
          })
        );
        setCompareHistory(Object.fromEntries(entries));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load compare metric history');
        setCompareHistory({});
      } finally {
        setCompareHistoryLoading(false);
      }
    })();
  }, [compareData, compareMetric]);

  const compareHistoryStats = useMemo(() => {
    const lines = (compareData?.runs ?? [])
      .slice(0, 8)
      .map((run) => ({ run, points: compareHistory[run.run_id] ?? [] }))
      .filter((x) => x.points.length > 0);

    if (!lines.length) {
      return {
        lines,
        minStep: 0,
        maxStep: 1,
        minValue: 0,
        maxValue: 1,
      };
    }

    let minStep = Number.POSITIVE_INFINITY;
    let maxStep = Number.NEGATIVE_INFINITY;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;

    for (const entry of lines) {
      for (const p of entry.points) {
        minStep = Math.min(minStep, p.step);
        maxStep = Math.max(maxStep, p.step);
        minValue = Math.min(minValue, p.value);
        maxValue = Math.max(maxValue, p.value);
      }
    }

    if (minStep === maxStep) maxStep += 1;
    if (minValue === maxValue) maxValue += Math.max(Math.abs(minValue) * 0.1, 1);

    return { lines, minStep, maxStep, minValue, maxValue };
  }, [compareData, compareHistory]);

  const sweepPoints = useMemo(() => {
    if (!sweepXAxis || !sweepYAxis) return [] as Array<{ run: Run; x: number; y: number }>;
    const points: Array<{ run: Run; x: number; y: number }> = [];
    for (const run of filteredRuns.slice(0, 600)) {
      const x = getSweepValue(run, sweepXAxis);
      const y = getSweepValue(run, sweepYAxis);
      if (x == null || y == null) continue;
      points.push({ run, x, y });
    }
    return points;
  }, [filteredRuns, sweepXAxis, sweepYAxis]);

  const sweepBounds = useMemo(() => {
    if (!sweepPoints.length) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const p of sweepPoints) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    if (minX === maxX) maxX += 1;
    if (minY === maxY) maxY += 1;
    return { minX, maxX, minY, maxY };
  }, [sweepPoints]);

  const sweepTopRuns = useMemo(() => {
    if (!sweepPoints.length) return [] as Array<{ run: Run; x: number; y: number }>;
    const objective =
      sweepYAxis.startsWith('metric:') && metricLowerIsBetter(sweepYAxis.slice('metric:'.length))
        ? 'min'
        : 'max';
    return [...sweepPoints]
      .sort((a, b) => (objective === 'min' ? a.y - b.y : b.y - a.y))
      .slice(0, 5);
  }, [sweepPoints, sweepYAxis]);

  useEffect(() => {
    if (!hoveredSweepPoint) return;
    const stillPresent = sweepPoints.some((entry) => entry.run.run_id === hoveredSweepPoint.run.run_id);
    if (!stillPresent) {
      setHoveredSweepPoint(null);
    }
  }, [hoveredSweepPoint, sweepPoints]);

  function updateModuleSpan(id: DashboardModuleId, span: DashboardModuleSpan): void {
    setModuleLayout((prev) =>
      normalizeModuleLayout(
        prev.map((item) => (item.id === id ? { ...item, span } : item))
      )
    );
  }

  function moveModule(sourceId: DashboardModuleId, targetId: DashboardModuleId): void {
    if (sourceId === targetId) return;
    setModuleLayout((prev) => {
      const next = normalizeModuleLayout(prev);
      const sourceIndex = next.findIndex((item) => item.id === sourceId);
      const targetIndex = next.findIndex((item) => item.id === targetId);
      if (sourceIndex === -1 || targetIndex === -1) return next;
      const reordered = [...next];
      const [moved] = reordered.splice(sourceIndex, 1);
      reordered.splice(targetIndex, 0, moved);
      return reordered;
    });
  }

  const runDetailMetricKeys = useMemo(() => {
    const historyKeys = Object.keys(runDetailMetrics?.metrics ?? {});
    if (historyKeys.length) return historyKeys.sort();
    return Object.keys(runDetail?.metrics ?? {}).sort();
  }, [runDetail, runDetailMetrics]);

  useEffect(() => {
    if (!runDetailMetricKeys.length) {
      if (runDetailMetricKey) setRunDetailMetricKey('');
      return;
    }

    if (runDetailMetricKey && runDetailMetricKeys.includes(runDetailMetricKey)) return;

    const preferred =
      (primaryMetric && runDetailMetricKeys.includes(primaryMetric) ? primaryMetric : '') ||
      runDetailMetricKeys[0];
    setRunDetailMetricKey(preferred);
  }, [primaryMetric, runDetailMetricKey, runDetailMetricKeys]);

  const runDetailHeroMetrics = useMemo(
    () => Object.entries(runDetail?.metrics ?? {}).slice(0, 5),
    [runDetail]
  );

  const runDetailChart = useMemo(() => {
    if (!runDetailMetricKey) return null;
    const points = runDetailMetrics?.metrics?.[runDetailMetricKey] ?? [];
    if (!points.length) return null;

    let minStep = Number.POSITIVE_INFINITY;
    let maxStep = Number.NEGATIVE_INFINITY;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;

    for (const point of points) {
      minStep = Math.min(minStep, point.step);
      maxStep = Math.max(maxStep, point.step);
      minValue = Math.min(minValue, point.value);
      maxValue = Math.max(maxValue, point.value);
    }

    if (minStep === maxStep) maxStep += 1;
    if (minValue === maxValue) maxValue += Math.max(Math.abs(minValue) * 0.1, 1);

    const path = points
      .map((point, idx) => {
        const x = 56 + ((point.step - minStep) / (maxStep - minStep)) * 688;
        const y = 214 - ((point.value - minValue) / (maxValue - minValue)) * 172;
        return `${idx === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');

    return { points, minStep, maxStep, minValue, maxValue, path };
  }, [runDetailMetricKey, runDetailMetrics]);

  async function openRun(runId: string): Promise<void> {
    try {
      const [run, metrics, artifacts] = await Promise.all([
        getRunDetail(runId),
        getMetricHistory(runId),
        getArtifacts(runId),
      ]);
      setRunDetail(run);
      setRunDetailMetrics(metrics);
      setRunDetailArtifacts(artifacts);
      setRunDetailMetricKey('');
      setRunDetailOpen(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load run detail');
    }
  }

  const savedViews = useMemo(
    () => (activeExperiment ? loadSavedViews(activeExperiment) : []),
    [activeExperiment, runQuery, sortMode, groupMode, activeStatus, tagFilter, metricFilterKey, metricMin, metricMax, onlySelected, selectedRunIds]
  );

  const paletteActions = useMemo<PaletteAction[]>(() => {
    const actions: PaletteAction[] = [
      {
        title: 'Refresh Dashboard',
        meta: 'Reload experiments and runs',
        run: () => setRefreshTick((x) => x + 1),
      },
      {
        title: 'Toggle Theme',
        meta: 'Switch dark/light mode',
        run: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
      },
    ];

    if (activeExperiment) {
      actions.push({
        title: 'Copy Shareable View URL',
        meta: 'Copy current filtered view URL',
        run: async () => {
          const snapshot: ViewStateSnapshot = {
            runQuery,
            sortMode,
            groupMode,
            activeStatus,
            tagFilter,
            metricFilterKey,
            metricMin,
            metricMax,
            onlySelected,
            selectedRunIds,
          };
          snapshotToUrl(activeExperiment, snapshot);
          const url = window.location.href;
          try {
            await navigator.clipboard.writeText(url);
          } catch {
            window.prompt('Copy this URL:', url);
          }
        },
      });

      actions.push({
        title: 'Save Current View',
        meta: 'Store this view preset locally',
        run: () => {
          const name = window.prompt('Saved view name:', `view-${new Date().toISOString().slice(0, 16)}`);
          if (!name || !name.trim()) return;
          const snapshot: ViewStateSnapshot = {
            runQuery,
            sortMode,
            groupMode,
            activeStatus,
            tagFilter,
            metricFilterKey,
            metricMin,
            metricMax,
            onlySelected,
            selectedRunIds,
          };
          const next: SavedView[] = [
            { name: name.trim(), createdAt: new Date().toISOString(), state: snapshot },
            ...savedViews.filter((x) => x.name !== name.trim()),
          ].slice(0, 25);
          saveSavedViews(activeExperiment, next);
          setRefreshTick((x) => x + 1);
        },
      });

      for (const entry of savedViews) {
        actions.push({
          title: `Load View: ${entry.name}`,
          meta: `Saved ${new Date(entry.createdAt).toLocaleString()}`,
          run: () => {
            const s = entry.state;
            setRunQuery(s.runQuery ?? '');
            setSortMode((s.sortMode as SortMode) ?? '-start_time');
            setGroupMode((s.groupMode as GroupMode) ?? 'none');
            setActiveStatus(s.activeStatus ?? 'ALL');
            setTagFilter(s.tagFilter ?? '');
            setMetricFilterKey(s.metricFilterKey ?? '');
            setMetricMin(s.metricMin ?? '');
            setMetricMax(s.metricMax ?? '');
            setOnlySelected(Boolean(s.onlySelected));
            setSelectedRunIds(Array.isArray(s.selectedRunIds) ? s.selectedRunIds.slice(0, 80) : []);
          },
        });
      }
    }

    const q = paletteQuery.trim().toLowerCase();
    if (!q) return actions;
    return actions.filter((x) => `${x.title} ${x.meta ?? ''}`.toLowerCase().includes(q));
  }, [
    paletteQuery,
    activeExperiment,
    runQuery,
    sortMode,
    groupMode,
    activeStatus,
    tagFilter,
    metricFilterKey,
    metricMin,
    metricMax,
    onlySelected,
    selectedRunIds,
    savedViews,
  ]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent): void {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen(true);
      }
      if (e.key === 'Escape') {
        setPaletteOpen(false);
        setRunDetailOpen(false);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  const layoutById = new Map(moduleLayout.map((item) => [item.id, item] as const));

  const dashboardModules: DashboardModuleDefinition[] = [
    {
      id: 'leaderboard',
      title: 'Leaderboard',
      visible: true,
      controls: (
        <div className="leaderboard-controls">
          <select className="sort-select" value={leaderboardMetric} onChange={(e) => setLeaderboardMetric(e.target.value)}>
            {metricKeys.map((key) => (
              <option key={`lb-${key}`} value={key}>{key}</option>
            ))}
          </select>
          <select className="sort-select" value={leaderboardObjective} onChange={(e) => setLeaderboardObjective(e.target.value as 'auto' | 'min' | 'max')}>
            <option value="auto">Objective: Auto</option>
            <option value="min">Objective: Min</option>
            <option value="max">Objective: Max</option>
          </select>
        </div>
      ),
      body: !leaderboardRows.length ? (
        <div className="module-empty">Need numeric metrics to build a leaderboard.</div>
      ) : (
        <ol className="leaderboard-list">
          {(() => {
            const values = leaderboardRows.map((entry) => entry.value);
            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const spread = Math.max(1e-12, maxVal - minVal);
            const objective =
              leaderboardObjective === 'auto'
                ? metricLowerIsBetter(leaderboardMetric)
                  ? 'min'
                  : 'max'
                : leaderboardObjective;

            return leaderboardRows.map((entry, idx) => {
              const ratio =
                objective === 'min'
                  ? (maxVal - entry.value) / spread
                  : (entry.value - minVal) / spread;
              const width = maxVal === minVal ? 92 : Math.max(18, Math.round(20 + ratio * 72));
              return (
                <li key={`leader-${entry.run.run_id}`} className="leaderboard-item">
                  <div className="leaderboard-rank">#{idx + 1}</div>
                  <button type="button" className="leaderboard-run-btn" onClick={() => void openRun(entry.run.run_id)}>
                    <span>{entry.run.name ?? entry.run.run_id}</span>
                    <small>{entry.run.run_id}</small>
                  </button>
                  <div className="leaderboard-bar-wrap">
                    <div className="leaderboard-bar" style={{ width: `${width}%` }} />
                  </div>
                  <div className="leaderboard-value">{formatMetricValue(entry.value)}</div>
                </li>
              );
            });
          })()}
        </ol>
      ),
    },
    {
      id: 'sweep',
      title: 'Sweep Explorer',
      visible: true,
      controls: (
        <div className="sweep-controls">
          <select className="sort-select" value={sweepXAxis} onChange={(e) => setSweepXAxis(e.target.value as SweepAxisMode)}>
            <option value="">X Axis</option>
            {sweepAxisOptions.map((axis) => (
              <option key={`sx-${axis.value}`} value={axis.value}>{axis.label}</option>
            ))}
          </select>
          <select className="sort-select" value={sweepYAxis} onChange={(e) => setSweepYAxis(e.target.value as SweepAxisMode)}>
            <option value="">Y Axis</option>
            {sweepAxisOptions.map((axis) => (
              <option key={`sy-${axis.value}`} value={axis.value}>{axis.label}</option>
            ))}
          </select>
        </div>
      ),
      body: (
        <div className="module-grid">
          <div className="chart-container compact">
            <svg className="viz-svg" viewBox="0 0 760 240" role="img" aria-label="Sweep scatter chart">
              <rect x="0" y="0" width="760" height="240" rx="8" fill="rgba(6,16,28,0.45)" />
              <line x1="54" y1="18" x2="54" y2="205" className="viz-axis" />
              <line x1="54" y1="205" x2="744" y2="205" className="viz-axis" />
              {sweepPoints.map((p, idx) => {
                const plotX = 54 + ((p.x - sweepBounds.minX) / (sweepBounds.maxX - sweepBounds.minX)) * 690;
                const plotY = 205 - ((p.y - sweepBounds.minY) / (sweepBounds.maxY - sweepBounds.minY)) * 187;
                const isHovered = hoveredSweepPoint?.run.run_id === p.run.run_id;
                return (
                  <g key={`sweep-${p.run.run_id}-${idx}`}>
                    <circle
                      cx={plotX}
                      cy={plotY}
                      r={isHovered ? '5.1' : '3.2'}
                      fill={(p.run.status ?? '').toUpperCase() === 'FAILED' ? '#f87171' : '#4f8ef7'}
                      opacity={isHovered ? '0.98' : '0.78'}
                    />
                    <circle
                      cx={plotX}
                      cy={plotY}
                      r="10"
                      fill="transparent"
                      className="parallel-hit-line"
                      onMouseEnter={(e) => {
                        const svg = e.currentTarget.ownerSVGElement;
                        if (!svg) return;
                        const rect = svg.getBoundingClientRect();
                        setHoveredSweepPoint({
                          run: p.run,
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top,
                          plotX,
                          plotY,
                          valueX: p.x,
                          valueY: p.y,
                        });
                      }}
                      onMouseMove={(e) => {
                        const svg = e.currentTarget.ownerSVGElement;
                        if (!svg) return;
                        const rect = svg.getBoundingClientRect();
                        setHoveredSweepPoint({
                          run: p.run,
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top,
                          plotX,
                          plotY,
                          valueX: p.x,
                          valueY: p.y,
                        });
                      }}
                      onMouseLeave={() => {
                        setHoveredSweepPoint((current) =>
                          current?.run.run_id === p.run.run_id ? null : current
                        );
                      }}
                    />
                  </g>
                );
              })}
            </svg>
            {hoveredSweepPoint && (
              <div
                className="viz-tooltip"
                style={{
                  left: `${Math.min(610, Math.max(12, hoveredSweepPoint.x + 12))}px`,
                  top: `${Math.min(208, Math.max(12, hoveredSweepPoint.y - 12))}px`,
                }}
              >
                <div className="viz-tooltip-title">
                  {hoveredSweepPoint.run.name ?? hoveredSweepPoint.run.run_id}
                </div>
                <div className="viz-tooltip-subtitle">{hoveredSweepPoint.run.run_id}</div>
                <div className="viz-tooltip-status">
                  status: {hoveredSweepPoint.run.status ?? 'UNKNOWN'}
                </div>
                <div className="viz-tooltip-values">
                  <span>
                    {sweepAxisLabel(sweepXAxis)} {formatMetricValue(hoveredSweepPoint.valueX)}
                  </span>
                  <span>
                    {sweepAxisLabel(sweepYAxis)} {formatMetricValue(hoveredSweepPoint.valueY)}
                  </span>
                </div>
              </div>
            )}
          </div>
          <div>
            <h4 className="module-subtitle">Top Runs</h4>
            <ul className="compact-list">
              {sweepTopRuns.map((entry) => (
                <li key={`top-${entry.run.run_id}`}>
                  <button type="button" className="link-btn" onClick={() => void openRun(entry.run.run_id)}>
                    {entry.run.name ?? entry.run.run_id}
                  </button>
                  <span>{formatMetricValue(entry.y)}</span>
                </li>
              ))}
              {!sweepTopRuns.length && <li>Select numeric axes to rank sweep points.</li>}
            </ul>
          </div>
        </div>
      ),
    },
    {
      id: 'parallel',
      title: 'Parallel Coordinates',
      visible: true,
      controls: (
        <div className="parallel-controls">
          <button className="action-btn subtle" onClick={() => setParallelDims(defaultParallelDimensions(parallelAvailableDims, primaryMetric))}>
            Reset Axes
          </button>
        </div>
      ),
      body: (
        <>
          <div className="chip-selector">
            {parallelAvailableDims.map((dim) => (
              <button
                key={`dim-${dim}`}
                type="button"
                className={`axis-chip ${parallelDims.includes(dim) ? 'active' : ''}`}
                onClick={() => {
                  setParallelDims((prev) => {
                    if (prev.includes(dim)) return prev.filter((x) => x !== dim);
                    return [...prev, dim].slice(0, 6);
                  });
                }}
              >
                {dimensionLabel(dim)}
              </button>
            ))}
          </div>
          <div className="chart-container compact">
            <svg className="viz-svg" viewBox="0 0 760 300" role="img" aria-label="Parallel coordinates chart">
              <rect x="0" y="0" width="760" height="300" rx="8" fill="rgba(6,16,28,0.45)" />
              {parallelSeries.ranges.map((range, idx) => {
                const x = 70 + (idx / Math.max(1, parallelSeries.ranges.length - 1)) * 620;
                return (
                  <g key={`axis-${range.dim}`}>
                    <line x1={x} y1="36" x2={x} y2="252" className="viz-axis" />
                    <text x={x} y="274" textAnchor="middle" className="viz-label">{dimensionLabel(range.dim)}</text>
                    <text x={x} y="30" textAnchor="middle" className="viz-subtle">{formatMetricValue(range.max)}</text>
                    <text x={x} y="252" textAnchor="middle" className="viz-subtle">{formatMetricValue(range.min)}</text>
                  </g>
                );
              })}
              {parallelSeries.rows.map((entry, rowIdx) => {
                const isHovered = hoveredParallelRun?.run.run_id === entry.run.run_id;
                return (
                  <g key={`parallel-${entry.run.run_id}`}>
                    <polyline
                      points={entry.points}
                      fill="none"
                      stroke={selectedSet.has(entry.run.run_id) ? '#f6a623' : colorForIndex(rowIdx)}
                      strokeWidth={isHovered ? 2.8 : selectedSet.has(entry.run.run_id) ? 2.2 : 1.2}
                      opacity={
                        isHovered
                          ? 0.96
                          : selectedSet.size && !selectedSet.has(entry.run.run_id)
                            ? 0.15
                            : 0.45
                      }
                    />
                    <polyline
                      points={entry.points}
                      fill="none"
                      stroke="transparent"
                      strokeWidth="12"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="parallel-hit-line"
                      onMouseEnter={(e) => {
                        const svg = e.currentTarget.ownerSVGElement;
                        if (!svg) return;
                        const rect = svg.getBoundingClientRect();
                        setHoveredParallelRun({
                          run: entry.run,
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top,
                          values: entry.pointEntries.map((point) => ({ dim: point.dim, value: point.value })),
                        });
                      }}
                      onMouseMove={(e) => {
                        const svg = e.currentTarget.ownerSVGElement;
                        if (!svg) return;
                        const rect = svg.getBoundingClientRect();
                        setHoveredParallelRun({
                          run: entry.run,
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top,
                          values: entry.pointEntries.map((point) => ({ dim: point.dim, value: point.value })),
                        });
                      }}
                      onMouseLeave={() => {
                        setHoveredParallelRun((current) =>
                          current?.run.run_id === entry.run.run_id ? null : current
                        );
                      }}
                    />
                  </g>
                );
              })}
            </svg>
            {hoveredParallelRun && (
              <div
                className="viz-tooltip"
                style={{
                  left: `${Math.min(610, Math.max(12, hoveredParallelRun.x + 12))}px`,
                  top: `${Math.min(208, Math.max(12, hoveredParallelRun.y - 12))}px`,
                }}
              >
                <div className="viz-tooltip-title">
                  {hoveredParallelRun.run.name ?? hoveredParallelRun.run.run_id}
                </div>
                <div className="viz-tooltip-subtitle">{hoveredParallelRun.run.run_id}</div>
                <div className="viz-tooltip-status">
                  status: {hoveredParallelRun.run.status ?? 'UNKNOWN'}
                </div>
                <div className="viz-tooltip-values">
                  {hoveredParallelRun.values.slice(0, 4).map((entry) => (
                    <span key={`parallel-hover-${hoveredParallelRun.run.run_id}-${entry.dim}`}>
                      {dimensionLabel(entry.dim)} {formatMetricValue(entry.value)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      ),
    },
    {
      id: 'importance',
      title: 'Feature Importance (Proxy)',
      visible: true,
      controls: (
        <div className="importance-controls">
          <select className="sort-select" value={importanceTarget} onChange={(e) => setImportanceTarget(e.target.value)}>
            {metricKeys.map((key) => (
              <option key={`importance-${key}`} value={key}>{key}</option>
            ))}
          </select>
        </div>
      ),
      body: (
        <div className="module-grid">
          <div className="chart-container compact">
            <svg className="viz-svg" viewBox="0 0 760 240" role="img" aria-label="Importance chart">
              <rect x="0" y="0" width="760" height="240" rx="8" fill="rgba(6,16,28,0.45)" />
              {importanceRows.map((entry, idx) => {
                const x = 54;
                const y = 28 + idx * 24;
                const width = Math.max(10, entry.score * 620);
                return (
                  <g key={`importance-bar-${entry.key}`}>
                    <text x="54" y={y - 6} className="viz-label">{entry.key}</text>
                    <rect x={x} y={y} width="620" height="12" rx="6" fill="rgba(255,255,255,0.08)" />
                    <rect x={x} y={y} width={width} height="12" rx="6" fill={entry.direction >= 0 ? '#34d399' : '#f87171'} />
                  </g>
                );
              })}
            </svg>
          </div>
          <div>
            <h4 className="module-subtitle">Top Drivers</h4>
            <ul className="importance-list">
              {importanceRows.map((entry) => (
                <li key={`importance-row-${entry.key}`} className="importance-item">
                  <div className="importance-main">
                    <span>{entry.key}</span>
                    <small>{entry.samples} runs · {entry.direction >= 0 ? 'positive' : 'negative'} signal</small>
                  </div>
                  <div className="importance-score">{formatMetricValue(entry.score)}</div>
                </li>
              ))}
              {!importanceRows.length && <li className="module-empty">Need numeric params and a numeric target metric.</li>}
            </ul>
          </div>
        </div>
      ),
    },
    {
      id: 'compare',
      title: 'Run Comparison',
      visible: Boolean(compareData),
      controls: compareData ? (
        <div className="compare-controls">
          <select className="sort-select" value={compareMetric} onChange={(e) => setCompareMetric(e.target.value)}>
            {compareData.metric_keys.map((k) => (
              <option key={`compare-${k}`} value={k}>{k}</option>
            ))}
          </select>
          <button className="action-btn subtle" onClick={() => setCompareData(null)}>Close</button>
        </div>
      ) : null,
      body: compareData ? (
        <div className="compare-grid">
          <div className="chart-container compact">
            <svg className="viz-svg" viewBox="0 0 760 240" role="img" aria-label="Compare metric history chart">
              <rect x="0" y="0" width="760" height="240" rx="8" fill="rgba(6,16,28,0.45)" />
              <line x1="54" y1="18" x2="54" y2="205" className="viz-axis" />
              <line x1="54" y1="205" x2="744" y2="205" className="viz-axis" />
              {compareHistoryStats.lines.map((entry, idx) => {
                const points = entry.points
                  .map((p) => {
                    const x =
                      54 +
                      ((p.step - compareHistoryStats.minStep) /
                        (compareHistoryStats.maxStep - compareHistoryStats.minStep)) *
                        690;
                    const y =
                      205 -
                      ((p.value - compareHistoryStats.minValue) /
                        (compareHistoryStats.maxValue - compareHistoryStats.minValue)) *
                        187;
                    return `${x.toFixed(2)},${y.toFixed(2)}`;
                  })
                  .join(' ');
                if (!points) return null;
                return (
                  <polyline
                    key={`line-${entry.run.run_id}`}
                    points={points}
                    fill="none"
                    stroke={colorForIndex(idx)}
                    strokeWidth="2"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                );
              })}
            </svg>
            <div className="viz-legend">
              {compareHistoryLoading && <span className="module-empty">Loading metric history…</span>}
              {compareHistoryStats.lines.map((entry, idx) => (
                <span key={`legend-${entry.run.run_id}`} className="viz-tag">
                  <i style={{ backgroundColor: colorForIndex(idx) }} />
                  {entry.run.name ?? entry.run.run_id}
                </span>
              ))}
            </div>
          </div>
          <div className="table-container">
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Duration</th>
                  <th>{compareMetric || 'Value'}</th>
                </tr>
              </thead>
              <tbody>
                {compareData.runs.map((run) => (
                  <tr key={`cmp-${run.run_id}`}>
                    <td>{run.name ?? run.run_id}</td>
                    <td>{run.status ?? '-'}</td>
                    <td>{formatDuration(run.duration)}</td>
                    <td>{compareMetric ? formatMetricValue(run.metrics?.[compareMetric]) : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="module-empty">Choose at least two runs to compare them here.</div>
      ),
    },
  ];

  const orderedDashboardModules = (() => {
    const defsById = new Map(dashboardModules.map((module) => [module.id, module] as const));
    const seen = new Set<DashboardModuleId>();
    const ordered: DashboardModuleDefinition[] = [];

    for (const item of moduleLayout) {
      const module = defsById.get(item.id);
      if (!module || !module.visible || seen.has(item.id)) continue;
      ordered.push(module);
      seen.add(item.id);
    }

    for (const module of dashboardModules) {
      if (!module.visible || seen.has(module.id)) continue;
      ordered.push(module);
    }

    return ordered;
  })();

  function renderDashboardModule(module: DashboardModuleDefinition): JSX.Element {
    const span = layoutById.get(module.id)?.span ?? DEFAULT_MODULE_LAYOUT.find((item) => item.id === module.id)?.span ?? 1;

    return (
      <section
        key={module.id}
        className={`module-card dashboard-module board-span-${span}${draggedModuleId === module.id ? ' is-dragging' : ''}${dropTargetModuleId === module.id ? ' is-drop-target' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (draggedModuleId && draggedModuleId !== module.id) {
            setDropTargetModuleId(module.id);
          }
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            setDropTargetModuleId((current) => (current === module.id ? null : current));
          }
        }}
        onDrop={(e) => {
          e.preventDefault();
          const draggedId = e.dataTransfer.getData('text/plain');
          const sourceId =
            typeof draggedId === 'string' && isDashboardModuleId(draggedId)
              ? draggedId
              : draggedModuleId;
          if (sourceId) {
            moveModule(sourceId, module.id);
          }
          setDraggedModuleId(null);
          setDropTargetModuleId(null);
        }}
      >
        <div className="module-header">
          <h3>{module.title}</h3>
          <div className="module-header-actions">
            {module.controls}
            <div className="module-layout-tools">
              {[1, 2, 3].map((option) => (
                <button
                  key={`${module.id}-span-${option}`}
                  type="button"
                  className={`module-size-btn ${span === option ? 'active' : ''}`}
                  onClick={() => updateModuleSpan(module.id, option as DashboardModuleSpan)}
                  title={`Set module width to ${option} column${option > 1 ? 's' : ''}`}
                >
                  {option}x1
                </button>
              ))}
              <button
                type="button"
                className="module-drag-handle"
                draggable
                title="Drag to reorder"
                onDragStart={(e) => {
                  setDraggedModuleId(module.id);
                  e.dataTransfer.effectAllowed = 'move';
                  e.dataTransfer.setData('text/plain', module.id);
                }}
                onDragEnd={() => {
                  setDraggedModuleId(null);
                  setDropTargetModuleId(null);
                }}
              >
                Drag
              </button>
            </div>
          </div>
        </div>
        {module.body}
      </section>
    );
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo-icon" aria-hidden="true" />
          <div>
            <h1>MLTracker</h1>
            <p>Experiment Control Center</p>
          </div>
        </div>

        <div className="nav-section">
          <div className="sidebar-search-wrap">
            <input
              className="search-input"
              placeholder="Search experiments..."
              value={experimentFilter}
              onChange={(e) => setExperimentFilter(e.target.value)}
            />
          </div>
          <h3>Experiments</h3>
          <ul className="nav-list">
            {visibleExperiments.map((exp) => (
              <li key={exp.name}>
                <button
                  type="button"
                  className={`nav-item ${activeExperiment === exp.name ? 'active' : ''}`}
                  onClick={() => {
                    setActiveExperiment(exp.name);
                    window.location.hash = `#exp/${encodeURIComponent(exp.name)}`;
                  }}
                >
                  <span className="exp-name-text">{exp.name}</span>
                  <span className="exp-run-count">{exp.run_count ?? '-'}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="sidebar-footer">v2 workspace</div>
      </aside>

      <main className="main-content">
        <header className="top-bar">
          <div>
            <div className="breadcrumbs">
              <span className="crumb root">Dashboard</span>
              {activeExperiment && (
                <>
                  <span className="sep">/</span>
                  <span className="crumb active">{activeExperiment}</span>
                </>
              )}
            </div>
            <div className="topbar-meta">Experiment Intelligence</div>
          </div>
          <div className="actions">
            <span className={`connection-pill ${health === 'healthy' ? 'online' : health === 'offline' ? 'offline' : 'pending'}`}>
              API {health}
            </span>
            <button className="action-btn subtle" onClick={() => setPaletteOpen(true)}>Command</button>
            <button className="icon-btn" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>
              {theme === 'dark' ? '☀' : '☾'}
            </button>
            <button className="icon-btn" onClick={() => setRefreshTick((x) => x + 1)}>↻</button>
          </div>
        </header>

        <div className={`content-shell ${activeExperiment ? 'with-run-sidebar' : ''}`}>
          <aside className={`run-sidebar ${activeExperiment ? '' : 'hidden'}`}>
            <div className="run-sidebar-header">
              <div>
                <h3>Runs</h3>
                <p>{filteredRuns.length} runs · {selectedRunIds.length} selected</p>
              </div>
              <button className="icon-btn" onClick={() => setSelectedRunIds([])}>✕</button>
            </div>
            <div className="run-sidebar-list">
              {filteredRuns.slice(0, 120).map((run) => (
                <article
                  key={`rail-${run.run_id}`}
                  className={`run-rail-item ${selectedSet.has(run.run_id) ? 'is-selected' : ''}`}
                  onClick={() => void openRun(run.run_id)}
                >
                  <input
                    className="run-rail-check"
                    type="checkbox"
                    checked={selectedSet.has(run.run_id)}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedRunIds((prev) => [...new Set([...prev, run.run_id])]);
                      } else {
                        setSelectedRunIds((prev) => prev.filter((x) => x !== run.run_id));
                      }
                    }}
                  />
                  <div className="run-rail-main">
                    <div className="run-rail-head">
                      <span className="run-rail-name">{run.name ?? run.run_id}</span>
                      <span className="run-rail-status">{run.status ?? 'UNKNOWN'}</span>
                    </div>
                    <div className="run-rail-meta">{run.run_id} · {formatDuration(run.duration)}</div>
                  </div>
                </article>
              ))}
              {!filteredRuns.length && !loadingRuns && (
                <div className="run-rail-empty">No runs match current filters.</div>
              )}
            </div>
          </aside>

          <section className="view-container">
            {error && <div className="error-banner">{error}</div>}
            {!activeExperiment ? (
              <div className="empty-state">
                <h2>Select a project</h2>
                <p>Pick an experiment from the sidebar to explore runs, metrics, charts, and artifacts.</p>
              </div>
            ) : (
              <>
                <div className="view-header">
                  <div>
                    <div className="exp-kicker">Experiment Workspace</div>
                    <h2>{activeExperiment}</h2>
                    <p className="view-subtitle">
                      {loadingRuns ? 'Loading runs...' : `${filteredRuns.length} / ${sortedRuns.length} runs visible`}
                    </p>
                  </div>
                  <div className="controls">
                    <input
                      className="search-input"
                      placeholder="Search run name or id"
                      value={runQuery}
                      onChange={(e) => setRunQuery(e.target.value)}
                    />
                    <select className="sort-select" value={sortMode} onChange={(e) => setSortMode(e.target.value as SortMode)}>
                      <option value="-start_time">Newest First</option>
                      <option value="start_time">Oldest First</option>
                      <option value="-duration">Longest Duration</option>
                      <option value="duration">Shortest Duration</option>
                    </select>
                    <select className="sort-select" value={groupMode} onChange={(e) => setGroupMode(e.target.value as GroupMode)}>
                      <option value="none">No Grouping</option>
                      <option value="status">Group by Status</option>
                      <option value="day">Group by Day</option>
                    </select>
                    <button
                      className="action-btn"
                      disabled={selectedRunIds.length < 2}
                      onClick={async () => {
                        try {
                          const data = await compareRuns(selectedRunIds);
                          setCompareData(data);
                          setCompareMetric(data.metric_keys[0] ?? '');
                        } catch (err) {
                          setError(err instanceof Error ? err.message : 'Compare failed');
                        }
                      }}
                    >
                      Compare Selected ({selectedRunIds.length})
                    </button>
                  </div>
                </div>

                <div className="summary-grid">
                  <KpiCard label="Total Runs" value={filteredRuns.length} tone="gold" />
                  <KpiCard label="Running" value={runningRuns} tone="cyan" />
                  <KpiCard label={failedRuns ? 'Failed' : 'Finished'} value={failedRuns || finishedRuns} tone="mint" />
                  <KpiCard label="Avg Duration" value={formatDuration(avgDuration)} tone="gold" />
                </div>

                <div className="insights-strip">
                  <article className="insight-card">
                    <div className="insight-label">Primary Signal</div>
                    <div className="insight-value">
                      {primaryMetric && leaderboardRows[0]
                        ? `${primaryMetric}: ${formatMetricValue(leaderboardRows[0].value)}`
                        : 'No numeric metric'}
                    </div>
                    <div className="insight-note">
                      {leaderboardRows[0] ? `Best run: ${leaderboardRows[0].run.name ?? leaderboardRows[0].run.run_id}` : 'Log metrics to unlock ranking'}
                    </div>
                  </article>
                  <article className="insight-card">
                    <div className="insight-label">Execution Quality</div>
                    <div className="insight-value">{successRate.toFixed(1)}%</div>
                    <div className="insight-note">{finishedRuns}/{filteredRuns.length} runs finished</div>
                  </article>
                  <article className="insight-card">
                    <div className="insight-label">Recent Activity</div>
                    <div className="insight-value">{last24hRuns} runs / 24h</div>
                    <div className="insight-note">
                      {latestRun ? `Latest: ${latestRun.name ?? latestRun.run_id} · ${formatCompactDate(latestRun.start_time)}` : 'No recent run activity'}
                    </div>
                  </article>
                </div>

                <div className="dashboard-board">
                  {orderedDashboardModules.map((module) => renderDashboardModule(module))}
                </div>

                <div className="advanced-controls">
                  <input
                    className="search-input small"
                    placeholder="Tag filter (key:value or value)"
                    value={tagFilter}
                    onChange={(e) => setTagFilter(e.target.value)}
                  />
                  <select className="sort-select" value={metricFilterKey} onChange={(e) => setMetricFilterKey(e.target.value)}>
                    <option value="">Metric filter: Any</option>
                    {metricKeys.map((k) => (
                      <option key={`filter-${k}`} value={k}>{k}</option>
                    ))}
                  </select>
                  <input className="search-input xs" placeholder="Metric min" value={metricMin} onChange={(e) => setMetricMin(e.target.value)} />
                  <input className="search-input xs" placeholder="Metric max" value={metricMax} onChange={(e) => setMetricMax(e.target.value)} />
                  <label className="toggle-chip">
                    <input type="checkbox" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
                    <span>Only selected</span>
                  </label>
                  <button
                    className="action-btn subtle"
                    onClick={() => {
                      const merged = new Set(selectedRunIds);
                      for (const run of filteredRuns) merged.add(run.run_id);
                      setSelectedRunIds([...merged]);
                    }}
                  >
                    Select Visible
                  </button>
                  <button className="action-btn subtle" onClick={() => setSelectedRunIds([])}>
                    Clear Selected
                  </button>
                </div>

                <div className="status-filters">
                  {statusOptions.map((status) => (
                    <button
                      key={status}
                      type="button"
                      className={`status-filter-btn ${status === activeStatus ? 'active' : ''}`}
                      onClick={() => setActiveStatus(status)}
                    >
                      <span>{status}</span>
                      <b>{status === 'ALL' ? sortedRuns.length : statusCounts.get(status) ?? 0}</b>
                    </button>
                  ))}
                </div>

                <div className="table-container">
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Pick</th>
                        <th>Run</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Duration</th>
                        <th>Latest Metrics</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {groupedRuns.map((group, groupIdx) => (
                        <Fragment key={`grp-${group.label ?? 'none'}-${groupIdx}`}>
                          {group.label && (
                            <tr className="group-row">
                              <td colSpan={7}><span>{group.label}</span></td>
                            </tr>
                          )}
                          {group.runs.slice(0, 250).map((run) => (
                            <tr key={run.run_id} className={`run-row ${selectedSet.has(run.run_id) ? 'is-selected' : ''}`}>
                              <td className="checkbox-cell">
                                <input
                                  className="run-picker"
                                  type="checkbox"
                                  checked={selectedSet.has(run.run_id)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setSelectedRunIds((prev) => [...new Set([...prev, run.run_id])]);
                                    } else {
                                      setSelectedRunIds((prev) => prev.filter((x) => x !== run.run_id));
                                    }
                                  }}
                                />
                              </td>
                              <td>
                                <button type="button" className="leaderboard-run-btn" onClick={() => void openRun(run.run_id)}>
                                  <span className="run-name">{run.name ?? run.run_id}</span>
                                  <small className="run-id">{run.run_id}</small>
                                </button>
                              </td>
                              <td>
                                <span className={`status-badge status-${(run.status ?? 'UNKNOWN').toUpperCase()}`}>
                                  {run.status ?? 'UNKNOWN'}
                                </span>
                              </td>
                              <td>{run.start_time ? new Date(run.start_time).toLocaleString() : '-'}</td>
                              <td>{formatDuration(run.duration)}</td>
                              <td>
                                <div className="metric-preview">
                                  {Object.entries(run.metrics ?? {}).slice(0, 3).map(([k, v]) => (
                                    <span key={`${run.run_id}-${k}`} className="metric-chip">
                                      {k} <b>{formatMetricValue(v)}</b>
                                    </span>
                                  ))}
                                  {!Object.keys(run.metrics ?? {}).length && <span className="metric-chip">-</span>}
                                </div>
                              </td>
                              <td>
                                <button
                                  className="action-btn subtle"
                                  onClick={async () => {
                                    if (!window.confirm(`Delete run "${run.run_id}"?`)) return;
                                    try {
                                      await deleteRun(run.run_id);
                                      setRefreshTick((x) => x + 1);
                                    } catch (err) {
                                      setError(err instanceof Error ? err.message : 'Delete failed');
                                    }
                                  }}
                                >
                                  Delete
                                </button>
                              </td>
                            </tr>
                          ))}
                        </Fragment>
                      ))}
                    </tbody>
                  </table>
                  {!loadingRuns && filteredRuns.length === 0 && <p className="empty-note">No runs found for this experiment.</p>}
                </div>
              </>
            )}
          </section>
        </div>

        {runDetailOpen && runDetail && (
          <div className="drawer-overlay" onClick={() => setRunDetailOpen(false)}>
            <aside className="drawer" onClick={(e) => e.stopPropagation()}>
              <div className="drawer-head">
                <h3>{runDetail.name ?? runDetail.run_id}</h3>
                <button className="icon-btn" onClick={() => setRunDetailOpen(false)}>✕</button>
              </div>

              <section className="run-hero">
                {runDetailHeroMetrics.length ? (
                  runDetailHeroMetrics.map(([key, value]) => (
                    <div key={`hero-${key}`} className="hero-metric-item">
                      <span className="hero-metric-label">{key}</span>
                      <span className="hero-metric-value">{formatMetricValue(value)}</span>
                    </div>
                  ))
                ) : (
                  <div className="module-empty">No latest metrics logged for this run yet.</div>
                )}
              </section>

              <div className="drawer-section">
                <div className="module-header">
                  <h4>Metric History</h4>
                  {runDetailMetricKeys.length > 0 && (
                    <select
                      className="sort-select"
                      value={runDetailMetricKey}
                      onChange={(e) => setRunDetailMetricKey(e.target.value)}
                    >
                      {runDetailMetricKeys.map((key) => (
                        <option key={`detail-metric-${key}`} value={key}>
                          {key}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                {runDetailChart ? (
                  <>
                    <div className="chart-container compact">
                      <svg className="viz-svg" viewBox="0 0 800 260" role="img" aria-label="Run metric history chart">
                        <rect x="0" y="0" width="800" height="260" rx="8" fill="rgba(6,16,28,0.45)" />
                        <line x1="56" y1="24" x2="56" y2="214" className="viz-axis" />
                        <line x1="56" y1="214" x2="744" y2="214" className="viz-axis" />
                        <path
                          d={runDetailChart.path}
                          fill="none"
                          stroke="#f6a623"
                          strokeWidth="2.4"
                          strokeLinejoin="round"
                          strokeLinecap="round"
                        />
                        {runDetailChart.points.map((point, idx) => {
                          const x =
                            56 +
                            ((point.step - runDetailChart.minStep) /
                              (runDetailChart.maxStep - runDetailChart.minStep)) *
                              688;
                          const y =
                            214 -
                            ((point.value - runDetailChart.minValue) /
                              (runDetailChart.maxValue - runDetailChart.minValue)) *
                              172;
                          return (
                            <circle
                              key={`detail-point-${idx}`}
                              cx={x}
                              cy={y}
                              r="2.8"
                              fill="#4f8ef7"
                              opacity="0.9"
                            />
                          );
                        })}
                        <text x="56" y="18" className="viz-subtle">
                          {formatMetricValue(runDetailChart.maxValue)}
                        </text>
                        <text x="56" y="232" className="viz-subtle">
                          {formatMetricValue(runDetailChart.minValue)}
                        </text>
                        <text x="56" y="248" className="viz-subtle">
                          step {runDetailChart.minStep}
                        </text>
                        <text x="690" y="248" className="viz-subtle">
                          step {runDetailChart.maxStep}
                        </text>
                      </svg>
                    </div>
                    <div className="viz-legend">
                      <span className="viz-tag">
                        <i style={{ backgroundColor: '#f6a623' }} />
                        {runDetailMetricKey}
                      </span>
                      <span className="viz-tag">{runDetailChart.points.length} points</span>
                    </div>
                  </>
                ) : (
                  <div className="module-empty">
                    No metric history is available for this run yet.
                  </div>
                )}
              </div>

              <div className="drawer-section">
                <h4>Run Info</h4>
                <div className="kv-grid">
                  <span className="kv-k">run_id</span>
                  <span className="kv-v">{runDetail.run_id}</span>
                  <span className="kv-k">status</span>
                  <span className="kv-v">{runDetail.status ?? '-'}</span>
                  <span className="kv-k">duration</span>
                  <span className="kv-v">{formatDuration(runDetail.duration)}</span>
                  <span className="kv-k">started</span>
                  <span className="kv-v">{runDetail.start_time ? new Date(runDetail.start_time).toLocaleString() : '-'}</span>
                </div>
              </div>

              <div className="drawer-section">
                <h4>Params</h4>
                <div className="kv-grid">
                  {Object.entries(runDetail.params ?? {}).map(([k, v]) => (
                    <Fragment key={`pk-${k}`}>
                      <span className="kv-k">{k}</span>
                      <span className="kv-v">{String(v)}</span>
                    </Fragment>
                  ))}
                </div>
              </div>

              <div className="drawer-section">
                <h4>Tags</h4>
                <div className="tags-cloud">
                  {Object.entries(runDetail.tags ?? {}).map(([k, v]) => (
                    <span key={`tag-${k}`} className="tag-chip">{k}:{String(v)}</span>
                  ))}
                </div>
              </div>

              <div className="drawer-section">
                <h4>Metrics (Latest)</h4>
                <div className="kv-grid">
                  {Object.entries(runDetail.metrics ?? {}).map(([k, v]) => (
                    <Fragment key={`mk-${k}`}>
                      <span className="kv-k">{k}</span>
                      <span className="kv-v">{formatMetricValue(v)}</span>
                    </Fragment>
                  ))}
                </div>
              </div>

              <div className="drawer-section">
                <h4>Artifacts</h4>
                <ul className="artifact-list">
                  {runDetailArtifacts.map((a) => (
                    <li key={a.path}>
                      <span>{a.path}</span>
                      <a className="artifact-link" href={artifactDownloadUrl(runDetail.run_id, a.path)} target="_blank" rel="noreferrer">
                        Download
                      </a>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="drawer-section">
                <h4>Metric History Keys</h4>
                <p className="mono-copy">
                  {runDetailMetricKeys.join(', ') || 'No history'}
                </p>
              </div>
            </aside>
          </div>
        )}

        {paletteOpen && (
          <div className="drawer-overlay" onClick={() => setPaletteOpen(false)}>
            <section className="palette" onClick={(e) => e.stopPropagation()}>
              <div className="palette-head">
                <input
                  autoFocus
                  className="palette-input"
                  placeholder="Type a command..."
                  value={paletteQuery}
                  onChange={(e) => setPaletteQuery(e.target.value)}
                />
                <button className="icon-btn" onClick={() => setPaletteOpen(false)}>✕</button>
              </div>
              <div className="palette-list">
                {paletteActions.map((action) => (
                  <button
                    key={action.title}
                    className="palette-item"
                    onClick={async () => {
                      await action.run();
                      setPaletteOpen(false);
                      setPaletteQuery('');
                    }}
                  >
                    <strong>{action.title}</strong>
                    <small>{action.meta}</small>
                  </button>
                ))}
                {!paletteActions.length && <p className="empty-note">No commands match your search.</p>}
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
