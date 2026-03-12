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
const STATUS_ORDER = ['RUNNING', 'FINISHED', 'FAILED', 'CANCELED'];

type SortMode = '-start_time' | 'start_time' | '-duration' | 'duration';
type GroupMode = 'none' | 'status' | 'day';
type SweepAxisMode = '' | `metric:${string}` | `param:${string}`;

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

  const [runDetailOpen, setRunDetailOpen] = useState(false);
  const [runDetail, setRunDetail] = useState<Run | null>(null);
  const [runDetailMetrics, setRunDetailMetrics] = useState<MetricHistoryResponse | null>(null);
  const [runDetailArtifacts, setRunDetailArtifacts] = useState<Artifact[]>([]);

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
          setActiveExperiment(experimentsData[0].name);
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
    setSortMode((prefs.sortMode as SortMode) ?? '-start_time');
    setGroupMode((prefs.groupMode as GroupMode) ?? 'none');
    setActiveStatus((prefs.activeStatus as string) ?? 'ALL');
    setTagFilter((prefs.tagFilter as string) ?? '');
    setMetricFilterKey((prefs.metricFilterKey as string) ?? '');
    setMetricMin((prefs.metricMin as string) ?? '');
    setMetricMax((prefs.metricMax as string) ?? '');
    setOnlySelected(Boolean(prefs.onlySelected));
    setRunQuery((prefs.runQuery as string) ?? '');
    setSelectedRunIds(Array.isArray(prefs.selectedRunIds) ? prefs.selectedRunIds.slice(0, 80) : []);
    setCompareData(null);
    setCompareHistory({});
    setSweepXAxis('');
    setSweepYAxis('');
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

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">MLTracker / Dashboard V2</p>
          <h1>Experiment Command Center</h1>
        </div>
        <div className="toolbar-actions">
          <span className={`health-pill health-${health === 'healthy' ? 'ok' : 'warn'}`}>API: {health}</span>
          <button className="toolbar-btn" onClick={() => setPaletteOpen(true)}>Command</button>
          <button className="toolbar-btn" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
          <button className="toolbar-btn" onClick={() => setRefreshTick((x) => x + 1)}>Refresh</button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="kpi-grid">
        <KpiCard label="Experiments" value={visibleExperiments.length} tone="gold" />
        <KpiCard label="Runs in View" value={filteredRuns.length} tone="cyan" />
        <KpiCard label="Running" value={runningRuns} tone="mint" />
        <KpiCard label="Finished" value={finishedRuns} tone="gold" />
      </section>

      <section className="panel-grid">
        <aside className="panel">
          <div className="panel-head">
            <h2>Experiments</h2>
            <span>{visibleExperiments.length}</span>
          </div>
          <div className="panel-controls compact">
            <input
              className="inline-input"
              placeholder="Search experiments"
              value={experimentFilter}
              onChange={(e) => setExperimentFilter(e.target.value)}
            />
          </div>
          <div className="list-scroll">
            {visibleExperiments.map((exp) => (
              <button
                key={exp.name}
                type="button"
                className={`exp-item ${activeExperiment === exp.name ? 'active' : ''}`}
                onClick={() => {
                  setActiveExperiment(exp.name);
                  window.location.hash = `#exp/${encodeURIComponent(exp.name)}`;
                }}
              >
                <span>{exp.name}</span>
                <small>{exp.run_count ?? '-'} runs</small>
              </button>
            ))}
          </div>
        </aside>

        <main className="panel wide">
          <div className="panel-head">
            <h2>{activeExperiment || 'No experiment selected'}</h2>
            <span>{loadingRuns ? 'Loading...' : `${filteredRuns.length} / ${sortedRuns.length} runs`}</span>
          </div>
          <div className="panel-controls">
            <input
              className="inline-input"
              placeholder="Search run name/id"
              value={runQuery}
              onChange={(e) => setRunQuery(e.target.value)}
            />
            <select value={sortMode} onChange={(e) => setSortMode(e.target.value as SortMode)}>
              <option value="-start_time">Newest First</option>
              <option value="start_time">Oldest First</option>
              <option value="-duration">Longest Duration</option>
              <option value="duration">Shortest Duration</option>
            </select>
            <select value={groupMode} onChange={(e) => setGroupMode(e.target.value as GroupMode)}>
              <option value="none">No Grouping</option>
              <option value="status">Group by Status</option>
              <option value="day">Group by Day</option>
            </select>
            <button
              className="toolbar-btn"
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
              Compare ({selectedRunIds.length})
            </button>
          </div>

          <div className="panel-controls wrap">
            {statusOptions.map((status) => (
              <button
                key={status}
                className={`chip-btn ${status === activeStatus ? 'active' : ''}`}
                onClick={() => setActiveStatus(status)}
              >
                {status} {status === 'ALL' ? `(${sortedRuns.length})` : `(${statusCounts.get(status) ?? 0})`}
              </button>
            ))}
          </div>

          <div className="panel-controls wrap">
            <input
              className="inline-input"
              placeholder="Tag filter (key:value or value)"
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
            />
            <select value={metricFilterKey} onChange={(e) => setMetricFilterKey(e.target.value)}>
              <option value="">Metric filter: Any</option>
              {metricKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
            <input
              className="inline-input small"
              placeholder="Metric min"
              value={metricMin}
              onChange={(e) => setMetricMin(e.target.value)}
            />
            <input
              className="inline-input small"
              placeholder="Metric max"
              value={metricMax}
              onChange={(e) => setMetricMax(e.target.value)}
            />
            <label className="check-inline">
              <input
                type="checkbox"
                checked={onlySelected}
                onChange={(e) => setOnlySelected(e.target.checked)}
              />
              Only selected
            </label>
            <button
              className="toolbar-btn"
              onClick={() => {
                const merged = new Set(selectedRunIds);
                for (const run of filteredRuns) merged.add(run.run_id);
                setSelectedRunIds([...merged]);
              }}
            >
              Select Visible
            </button>
            <button className="toolbar-btn" onClick={() => setSelectedRunIds([])}>Clear Selected</button>
          </div>

          {compareData && (
            <section className="compare-box">
              <div className="compare-head">
                <h3>Run Comparison</h3>
                <div className="compare-actions">
                  <select value={compareMetric} onChange={(e) => setCompareMetric(e.target.value)}>
                    {compareData.metric_keys.map((k) => (
                      <option key={k} value={k}>{k}</option>
                    ))}
                  </select>
                  <button className="toolbar-btn" onClick={() => setCompareData(null)}>Close</button>
                </div>
              </div>

              <div className="viz-wrap">
                <div className="viz-head">
                  <h4>History: {compareMetric || 'metric'}</h4>
                  <span>{compareHistoryLoading ? 'Loading...' : `${compareHistoryStats.lines.length} runs`}</span>
                </div>
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
                  {compareHistoryStats.lines.map((entry, idx) => (
                    <span key={`legend-${entry.run.run_id}`} className="viz-tag">
                      <i style={{ backgroundColor: colorForIndex(idx) }} />
                      {entry.run.name ?? entry.run.run_id}
                    </span>
                  ))}
                </div>
              </div>

              <div className="table-wrap short">
                <table>
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
            </section>
          )}

          <section className="compare-box">
            <div className="compare-head">
              <h3>Sweep Scatter</h3>
              <div className="compare-actions">
                <select value={sweepXAxis} onChange={(e) => setSweepXAxis(e.target.value as SweepAxisMode)}>
                  <option value="">X Axis</option>
                  {sweepAxisOptions.map((axis) => (
                    <option key={`x-${axis.value}`} value={axis.value}>{axis.label}</option>
                  ))}
                </select>
                <select value={sweepYAxis} onChange={(e) => setSweepYAxis(e.target.value as SweepAxisMode)}>
                  <option value="">Y Axis</option>
                  {sweepAxisOptions.map((axis) => (
                    <option key={`y-${axis.value}`} value={axis.value}>{axis.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <svg className="viz-svg" viewBox="0 0 760 240" role="img" aria-label="Sweep scatter chart">
              <rect x="0" y="0" width="760" height="240" rx="8" fill="rgba(6,16,28,0.45)" />
              <line x1="54" y1="18" x2="54" y2="205" className="viz-axis" />
              <line x1="54" y1="205" x2="744" y2="205" className="viz-axis" />
              {sweepPoints.map((p, idx) => {
                const x = 54 + ((p.x - sweepBounds.minX) / (sweepBounds.maxX - sweepBounds.minX)) * 690;
                const y = 205 - ((p.y - sweepBounds.minY) / (sweepBounds.maxY - sweepBounds.minY)) * 187;
                return (
                  <circle
                    key={`scatter-${p.run.run_id}-${idx}`}
                    cx={x}
                    cy={y}
                    r="3.2"
                    fill={(p.run.status ?? '').toUpperCase() === 'FAILED' ? '#ff8d5f' : '#3bc8ff'}
                    opacity="0.75"
                  />
                );
              })}
            </svg>
            <p className="viz-note">
              {sweepPoints.length
                ? `${sweepPoints.length} points plotted from current filtered runs.`
                : 'Select numeric X and Y axes to plot run sweep points.'}
            </p>
          </section>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Pick</th>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Duration</th>
                  <th>Metrics</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {groupedRuns.map((group, groupIdx) => (
                  <Fragment key={`grp-${group.label ?? 'none'}-${groupIdx}`}>
                    {group.label && (
                      <tr className="group-row">
                        <td colSpan={7}>{group.label} ({group.runs.length})</td>
                      </tr>
                    )}
                    {group.runs.slice(0, 250).map((run) => (
                      <tr key={run.run_id}>
                        <td>
                          <input
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
                          <button className="link-run" onClick={() => void openRun(run.run_id)}>
                            {run.name ?? run.run_id}
                          </button>
                          <div className="run-subid">{run.run_id}</div>
                        </td>
                        <td>{run.status ?? '-'}</td>
                        <td>{run.start_time ? new Date(run.start_time).toLocaleString() : '-'}</td>
                        <td>{formatDuration(run.duration)}</td>
                        <td className="mono-cell">
                          {Object.entries(run.metrics ?? {})
                            .slice(0, 3)
                            .map(([k, v]) => `${k}: ${formatMetricValue(v)}`)
                            .join(' | ') || '-'}
                        </td>
                        <td>
                          <button
                            className="danger-btn"
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
            {!loadingRuns && filteredRuns.length === 0 && (
              <p className="empty-note">No runs found for this experiment.</p>
            )}
          </div>
        </main>
      </section>

      <section className="kpi-grid">
        <KpiCard label="Selected" value={selectedRunIds.length} tone="cyan" />
        <KpiCard label="Failed" value={failedRuns} tone="gold" />
        <KpiCard label="Grouped By" value={groupMode} tone="mint" />
        <KpiCard label="Theme" value={theme} tone="gold" />
      </section>

      {runDetailOpen && runDetail && (
        <div className="overlay" onClick={() => setRunDetailOpen(false)}>
          <aside className="drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <h3>{runDetail.name ?? runDetail.run_id}</h3>
              <button className="toolbar-btn" onClick={() => setRunDetailOpen(false)}>Close</button>
            </div>

            <div className="drawer-section">
              <h4>Run Info</h4>
              <p className="mono-cell">run_id: {runDetail.run_id}</p>
              <p>Status: {runDetail.status ?? '-'}</p>
              <p>Duration: {formatDuration(runDetail.duration)}</p>
              <p>Started: {runDetail.start_time ? new Date(runDetail.start_time).toLocaleString() : '-'}</p>
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
              <div className="kv-grid">
                {Object.entries(runDetail.tags ?? {}).map(([k, v]) => (
                  <Fragment key={`tk-${k}`}>
                    <span className="kv-k">{k}</span>
                    <span className="kv-v">{String(v)}</span>
                  </Fragment>
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
                    <a href={artifactDownloadUrl(runDetail.run_id, a.path)} target="_blank" rel="noreferrer">
                      Download
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            <div className="drawer-section">
              <h4>Metric History Keys</h4>
              <p className="mono-cell">{Object.keys(runDetailMetrics?.metrics ?? {}).join(', ') || 'No history'}</p>
            </div>
          </aside>
        </div>
      )}

      {paletteOpen && (
        <div className="overlay" onClick={() => setPaletteOpen(false)}>
          <section className="palette" onClick={(e) => e.stopPropagation()}>
            <div className="palette-head">
              <input
                autoFocus
                className="palette-input"
                placeholder="Type a command..."
                value={paletteQuery}
                onChange={(e) => setPaletteQuery(e.target.value)}
              />
              <button className="toolbar-btn" onClick={() => setPaletteOpen(false)}>Close</button>
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
    </div>
  );
}
