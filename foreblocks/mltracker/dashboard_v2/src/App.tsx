import { Fragment, useEffect, useMemo, useState } from 'react';
import {
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
import {
  buildCompareChartOption,
  buildImportanceChartOption,
  buildParallelChartOption,
  buildRunDetailChartOption,
  buildSweepChartOption,
  getChartTheme,
} from './dashboard/chartOptions';
import {
  CompareSection,
  ImportanceSection,
  LeaderboardSection,
  ParallelSection,
  SweepSection,
} from './dashboard/components/PanelSections';
import { RunDetailDrawer } from './dashboard/components/RunDetailDrawer';
import type {
  Artifact,
  CompareResponse,
  Experiment,
  MetricHistoryPoint,
  MetricHistoryResponse,
  Run,
} from './types';
import {
  DEFAULT_MODULE_LAYOUT,
  type DashboardModuleDefinition,
  type DashboardModuleId,
  type DashboardModuleLayoutItem,
  type DashboardModuleSpan,
  type GroupMode,
  type PaletteAction,
  type SavedView,
  type SortMode,
  type SweepAxisMode,
  type ViewStateSnapshot,
  STATUS_ORDER,
} from './dashboard/types';
import {
  collectParallelDimensions,
  defaultParallelDimensions,
  extractNumericMetricKeys,
  extractNumericParamKeys,
  formatCompactDate,
  formatDuration,
  formatMetricValue,
  getSweepValue,
  groupRuns,
  isDashboardModuleId,
  loadModuleLayout,
  loadSavedViews,
  loadViewPrefs,
  metricLowerIsBetter,
  metricValueForRun,
  normalizeModuleLayout,
  parseTime,
  parseUrlState,
  pearsonCorrelation,
  pickPrimaryMetricKey,
  runMatchesTagFilter,
  saveModuleLayout,
  saveSavedViews,
  saveViewPrefs,
  snapshotToUrl,
  sortRuns,
} from './dashboard/utils';

const STORAGE_THEME_KEY = 'mltracker_theme';

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

  const [runDetailOpen, setRunDetailOpen] = useState(false);
  const [runDetail, setRunDetail] = useState<Run | null>(null);
  const [runDetailMetrics, setRunDetailMetrics] = useState<MetricHistoryResponse | null>(null);
  const [runDetailArtifacts, setRunDetailArtifacts] = useState<Artifact[]>([]);
  const [runDetailMetricKey, setRunDetailMetricKey] = useState('');

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState('');
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem(STORAGE_THEME_KEY, theme);
  }, [theme]);

  const chartTheme = useMemo(() => getChartTheme(theme), [theme]);

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
    if (!leaderboardMetric) return [];
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

  const compareHistoryLines = useMemo(
    () =>
      (compareData?.runs ?? [])
      .slice(0, 8)
      .map((run) => ({ run, points: compareHistory[run.run_id] ?? [] }))
      .filter((x) => x.points.length > 0),
    [compareData, compareHistory]
  );

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

  const runDetailChartPoints = useMemo(() => {
    if (!runDetailMetricKey) return null;
    const points = runDetailMetrics?.metrics?.[runDetailMetricKey] ?? [];
    return points.length ? points : null;
  }, [runDetailMetricKey, runDetailMetrics]);

  const sweepChartOption = useMemo(
    () => buildSweepChartOption({ chartTheme, selectedSet, sweepPoints, sweepXAxis, sweepYAxis }),
    [chartTheme, selectedSet, sweepPoints, sweepXAxis, sweepYAxis]
  );

  const parallelActiveDims = useMemo(
    () => parallelDims.filter((dim) => parallelAvailableDims.includes(dim)).slice(0, 6),
    [parallelDims, parallelAvailableDims]
  );

  const parallelChartOption = useMemo(
    () => buildParallelChartOption({ chartTheme, filteredRuns, parallelActiveDims, selectedSet }),
    [chartTheme, filteredRuns, parallelActiveDims, selectedSet]
  );

  const importanceChartOption = useMemo(
    () => buildImportanceChartOption({ chartTheme, importanceRows }),
    [chartTheme, importanceRows]
  );

  const compareChartOption = useMemo(
    () => buildCompareChartOption({ chartTheme, compareHistoryLines, compareMetric }),
    [chartTheme, compareHistoryLines, compareMetric]
  );

  const runDetailChartOption = useMemo(
    () => buildRunDetailChartOption({ chartTheme, runDetailChartPoints, runDetailMetricKey }),
    [chartTheme, runDetailChartPoints, runDetailMetricKey]
  );

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

  const activeExperimentMeta = useMemo(
    () => experiments.find((exp) => exp.name === activeExperiment) ?? null,
    [activeExperiment, experiments]
  );

  function selectExperiment(name: string): void {
    setActiveExperiment(name);
    setProjectPickerOpen(false);
    window.location.hash = `#exp/${encodeURIComponent(name)}`;
  }

  function setRunSelected(runId: string, selected: boolean): void {
    setSelectedRunIds((prev) => {
      if (selected) return [...new Set([...prev, runId])];
      return prev.filter((x) => x !== runId);
    });
  }

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
      {
        title: 'Browse Projects',
        meta: 'Open the hidden project picker',
        run: () => setProjectPickerOpen(true),
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
        setProjectPickerOpen(false);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  const layoutById = new Map(moduleLayout.map((item) => [item.id, item] as const));

  const leaderboardSection = LeaderboardSection({
    leaderboardMetric,
    leaderboardObjective,
    leaderboardRows,
    metricKeys,
    onLeaderboardMetricChange: setLeaderboardMetric,
    onLeaderboardObjectiveChange: setLeaderboardObjective,
    onOpenRun: openRun,
  });

  const sweepSection = SweepSection({
    sweepAxisOptions,
    sweepChartOption,
    sweepTopRuns,
    sweepXAxis,
    sweepYAxis,
    onOpenRun: openRun,
    onSweepXAxisChange: setSweepXAxis,
    onSweepYAxisChange: setSweepYAxis,
  });

  const parallelSection = ParallelSection({
    parallelAvailableDims,
    parallelChartOption,
    parallelDims,
    onOpenRun: openRun,
    onResetAxes: () => setParallelDims(defaultParallelDimensions(parallelAvailableDims, primaryMetric)),
    onToggleDim: (dim) => {
      setParallelDims((prev) => {
        if (prev.includes(dim)) return prev.filter((x) => x !== dim);
        return [...prev, dim].slice(0, 6);
      });
    },
  });

  const importanceSection = ImportanceSection({
    importanceChartOption,
    importanceRows,
    importanceTarget,
    metricKeys,
    onImportanceTargetChange: setImportanceTarget,
  });

  const compareSection = compareData
    ? CompareSection({
        compareChartOption,
        compareHistoryLines,
        compareHistoryLoading,
        compareMetric,
        compareMetricKeys: compareData.metric_keys,
        compareRuns: compareData.runs,
        onClose: () => setCompareData(null),
        onCompareMetricChange: setCompareMetric,
      })
    : null;

  const dashboardModules: DashboardModuleDefinition[] = [
    {
      id: 'leaderboard',
      title: 'Leaderboard',
      visible: true,
      controls: leaderboardSection.controls,
      body: leaderboardSection.body,
    },
    {
      id: 'sweep',
      title: 'Sweep Explorer',
      visible: true,
      controls: sweepSection.controls,
      body: sweepSection.body,
    },
    {
      id: 'parallel',
      title: 'Parallel Coordinates',
      visible: true,
      controls: parallelSection.controls,
      body: parallelSection.body,
    },
    {
      id: 'importance',
      title: 'Feature Importance (Proxy)',
      visible: true,
      controls: importanceSection.controls,
      body: importanceSection.body,
    },
    {
      id: 'compare',
      title: 'Run Comparison',
      visible: Boolean(compareData),
      controls: compareSection?.controls ?? null,
      body: compareSection?.body ?? <div className="module-empty">Choose at least two runs to compare them here.</div>,
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
      <main className="main-content">
        <header className="top-bar">
          <div className="topbar-primary">
            <div className="topbar-brand">
              <div className="logo-icon" aria-hidden="true" />
              <div>
                <h1>MLTracker</h1>
                <p>Runs-first workspace</p>
              </div>
            </div>
            <button
              type="button"
              className="workspace-switcher"
              onClick={() => setProjectPickerOpen(true)}
            >
              <span className="workspace-switcher-label">Project</span>
              <strong>{activeExperiment || 'Select a project'}</strong>
              <small>
                {activeExperimentMeta
                  ? `${activeExperimentMeta.run_count ?? filteredRuns.length} tracked runs`
                  : `${experiments.length} projects available`}
              </small>
            </button>
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

        <div className="content-shell">
          <aside className="run-sidebar">
            <div className="run-sidebar-header">
              <div>
                <div className="sidebar-kicker">Runs</div>
                <h3>{activeExperiment || 'No project selected'}</h3>
                <p>
                  {activeExperiment
                    ? `${filteredRuns.length} visible · ${selectedRunIds.length} selected`
                    : 'Open a project to load its runs.'}
                </p>
              </div>
              <div className="run-sidebar-tools">
                <button
                  type="button"
                  className="action-btn subtle sidebar-project-toggle"
                  onClick={() => setProjectPickerOpen(true)}
                >
                  Projects
                </button>
                <button className="icon-btn" onClick={() => setSelectedRunIds([])}>✕</button>
              </div>
            </div>
            <div className="run-sidebar-list">
              {!activeExperiment && (
                <div className="run-rail-empty">
                  <p>Projects are tucked away until you need them.</p>
                  <button
                    type="button"
                    className="action-btn subtle"
                    onClick={() => setProjectPickerOpen(true)}
                  >
                    Open Projects
                  </button>
                </div>
              )}
              {activeExperiment && filteredRuns.slice(0, 120).map((run) => {
                const runStatus = (run.status ?? 'UNKNOWN').toUpperCase();
                const primaryValue =
                  primaryMetric ? metricValueForRun(run, primaryMetric) : null;

                return (
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
                      onChange={(e) => setRunSelected(run.run_id, e.target.checked)}
                    />
                    <div className="run-rail-main">
                      <div className="run-rail-head">
                        <span className="run-rail-name">{run.name ?? run.run_id}</span>
                        <span className={`run-rail-status rail-status-${runStatus}`}>
                          {run.status ?? 'UNKNOWN'}
                        </span>
                      </div>
                      <div className="run-rail-meta">{run.run_id}</div>
                      <div className="run-rail-foot">
                        <span>{formatDuration(run.duration)}</span>
                        <span>
                          {primaryMetric && primaryValue !== null
                            ? `${primaryMetric}: ${formatMetricValue(primaryValue)}`
                            : 'No primary metric'}
                        </span>
                      </div>
                    </div>
                  </article>
                );
              })}
              {!filteredRuns.length && !loadingRuns && activeExperiment && (
                <div className="run-rail-empty">No runs match current filters.</div>
              )}
            </div>
          </aside>

          <section className="view-container">
            {error && <div className="error-banner">{error}</div>}
            {!activeExperiment ? (
              <div className="empty-state">
                <h2>Select a project</h2>
                <p>Open the project picker to explore runs, metrics, charts, and artifacts.</p>
                <button
                  type="button"
                  className="action-btn"
                  onClick={() => setProjectPickerOpen(true)}
                >
                  Browse Projects
                </button>
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
                                  onChange={(e) => setRunSelected(run.run_id, e.target.checked)}
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

        <RunDetailDrawer
          artifacts={runDetailArtifacts}
          heroMetrics={runDetailHeroMetrics}
          isOpen={runDetailOpen}
          metricKey={runDetailMetricKey}
          metricKeys={runDetailMetricKeys}
          onClose={() => setRunDetailOpen(false)}
          onMetricKeyChange={setRunDetailMetricKey}
          run={runDetail}
          runDetailChartOption={runDetailChartOption}
          runDetailChartPointCount={runDetailChartPoints?.length ?? 0}
        />

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

        {projectPickerOpen && (
          <div
            className="drawer-overlay project-panel-overlay"
            onClick={() => setProjectPickerOpen(false)}
          >
            <aside className="project-panel" onClick={(e) => e.stopPropagation()}>
              <div className="project-panel-head">
                <div>
                  <div className="sidebar-kicker">Projects</div>
                  <h3>Switch workspace</h3>
                  <p>{experiments.length} total projects</p>
                </div>
                <button className="icon-btn" onClick={() => setProjectPickerOpen(false)}>✕</button>
              </div>
              <div className="project-panel-search">
                <input
                  autoFocus
                  className="search-input"
                  placeholder="Search projects..."
                  value={experimentFilter}
                  onChange={(e) => setExperimentFilter(e.target.value)}
                />
              </div>
              <ul className="nav-list project-list">
                {visibleExperiments.map((exp) => (
                  <li key={exp.name}>
                    <button
                      type="button"
                      className={`nav-item project-item ${activeExperiment === exp.name ? 'active' : ''}`}
                      onClick={() => selectExperiment(exp.name)}
                    >
                      <span className="project-item-main">
                        <span className="exp-name-text">{exp.name}</span>
                        <span className="project-item-meta">
                          {exp.run_count ?? 0} runs tracked
                        </span>
                      </span>
                      <span className="exp-run-count">{exp.run_count ?? '-'}</span>
                    </button>
                  </li>
                ))}
                {!visibleExperiments.length && (
                  <li className="run-rail-empty">No projects match your search.</li>
                )}
              </ul>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
