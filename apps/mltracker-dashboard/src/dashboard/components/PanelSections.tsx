import { EChart, type DashboardChartOption } from '../../components/EChart';
import type { CompareHistoryLine, ImportanceRow, LeaderboardRow, SweepAxisMode } from '../types';
import { colorForIndex, formatDuration, formatMetricValue, metricLowerIsBetter } from '../utils';

type LeaderboardSectionProps = {
  leaderboardMetric: string;
  leaderboardObjective: 'auto' | 'min' | 'max';
  leaderboardRows: LeaderboardRow[];
  metricKeys: string[];
  onLeaderboardMetricChange: (value: string) => void;
  onLeaderboardObjectiveChange: (value: 'auto' | 'min' | 'max') => void;
  onOpenRun: (runId: string) => void | Promise<void>;
};

export function LeaderboardSection({
  leaderboardMetric,
  leaderboardObjective,
  leaderboardRows,
  metricKeys,
  onLeaderboardMetricChange,
  onLeaderboardObjectiveChange,
  onOpenRun,
}: LeaderboardSectionProps) {
  const controls = (
    <div className="leaderboard-controls">
      <select className="sort-select" value={leaderboardMetric} onChange={(e) => onLeaderboardMetricChange(e.target.value)}>
        {metricKeys.map((key) => (
          <option key={`lb-${key}`} value={key}>{key}</option>
        ))}
      </select>
      <select
        className="sort-select"
        value={leaderboardObjective}
        onChange={(e) => onLeaderboardObjectiveChange(e.target.value as 'auto' | 'min' | 'max')}
      >
        <option value="auto">Objective: Auto</option>
        <option value="min">Objective: Min</option>
        <option value="max">Objective: Max</option>
      </select>
    </div>
  );

  const body = !leaderboardRows.length ? (
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
              <button type="button" className="leaderboard-run-btn" onClick={() => void onOpenRun(entry.run.run_id)}>
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
  );

  return { controls, body };
}

type SweepSectionProps = {
  sweepAxisOptions: Array<{ value: SweepAxisMode; label: string }>;
  sweepChartOption: DashboardChartOption;
  sweepTopRuns: Array<{ run: { run_id: string; name?: string }; y: number }>;
  sweepXAxis: SweepAxisMode;
  sweepYAxis: SweepAxisMode;
  onOpenRun: (runId: string) => void | Promise<void>;
  onSweepXAxisChange: (value: SweepAxisMode) => void;
  onSweepYAxisChange: (value: SweepAxisMode) => void;
};

export function SweepSection({
  sweepAxisOptions,
  sweepChartOption,
  sweepTopRuns,
  sweepXAxis,
  sweepYAxis,
  onOpenRun,
  onSweepXAxisChange,
  onSweepYAxisChange,
}: SweepSectionProps) {
  const controls = (
    <div className="sweep-controls">
      <select className="sort-select" value={sweepXAxis} onChange={(e) => onSweepXAxisChange(e.target.value as SweepAxisMode)}>
        <option value="">X Axis</option>
        {sweepAxisOptions.map((axis) => (
          <option key={`sx-${axis.value}`} value={axis.value}>{axis.label}</option>
        ))}
      </select>
      <select className="sort-select" value={sweepYAxis} onChange={(e) => onSweepYAxisChange(e.target.value as SweepAxisMode)}>
        <option value="">Y Axis</option>
        {sweepAxisOptions.map((axis) => (
          <option key={`sy-${axis.value}`} value={axis.value}>{axis.label}</option>
        ))}
      </select>
    </div>
  );

  const body = (
    <div className="module-grid">
      <div className="chart-container compact">
        <EChart
          className="echart-surface"
          height={260}
          option={sweepChartOption}
          onClick={(params) => {
            const runId = (params as { data?: { runId?: string } }).data?.runId;
            if (runId) void onOpenRun(runId);
          }}
        />
      </div>
      <div>
        <h4 className="module-subtitle">Top Runs</h4>
        <ul className="compact-list">
          {sweepTopRuns.map((entry) => (
            <li key={`top-${entry.run.run_id}`}>
              <button type="button" className="link-btn" onClick={() => void onOpenRun(entry.run.run_id)}>
                {entry.run.name ?? entry.run.run_id}
              </button>
              <span>{formatMetricValue(entry.y)}</span>
            </li>
          ))}
          {!sweepTopRuns.length && <li>Select numeric axes to rank sweep points.</li>}
        </ul>
      </div>
    </div>
  );

  return { controls, body };
}

type ParallelSectionProps = {
  parallelAvailableDims: string[];
  parallelChartOption: DashboardChartOption;
  parallelDims: string[];
  onOpenRun: (runId: string) => void | Promise<void>;
  onResetAxes: () => void;
  onToggleDim: (dim: string) => void;
};

export function ParallelSection({
  parallelAvailableDims,
  parallelChartOption,
  parallelDims,
  onOpenRun,
  onResetAxes,
  onToggleDim,
}: ParallelSectionProps) {
  const controls = (
    <div className="parallel-controls">
      <button className="action-btn subtle" onClick={onResetAxes}>
        Reset Axes
      </button>
    </div>
  );

  const body = (
    <>
      <div className="chip-selector">
        {parallelAvailableDims.map((dim) => (
          <button
            key={`dim-${dim}`}
            type="button"
            className={`axis-chip ${parallelDims.includes(dim) ? 'active' : ''}`}
            onClick={() => onToggleDim(dim)}
          >
            {dim === '__duration__' ? 'duration' : dim}
          </button>
        ))}
      </div>
      <div className="chart-container compact">
        <EChart
          className="echart-surface"
          height={300}
          option={parallelChartOption}
          onClick={(params) => {
            const runId = (params as { data?: { runId?: string } }).data?.runId;
            if (runId) void onOpenRun(runId);
          }}
        />
      </div>
    </>
  );

  return { controls, body };
}

type ImportanceSectionProps = {
  importanceChartOption: DashboardChartOption;
  importanceRows: ImportanceRow[];
  importanceTarget: string;
  metricKeys: string[];
  onImportanceTargetChange: (value: string) => void;
};

export function ImportanceSection({
  importanceChartOption,
  importanceRows,
  importanceTarget,
  metricKeys,
  onImportanceTargetChange,
}: ImportanceSectionProps) {
  const controls = (
    <div className="importance-controls">
      <select className="sort-select" value={importanceTarget} onChange={(e) => onImportanceTargetChange(e.target.value)}>
        {metricKeys.map((key) => (
          <option key={`importance-${key}`} value={key}>{key}</option>
        ))}
      </select>
    </div>
  );

  const body = (
    <div className="module-grid">
      <div className="chart-container compact">
        <EChart className="echart-surface" height={260} option={importanceChartOption} />
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
  );

  return { controls, body };
}

type CompareSectionProps = {
  compareChartOption: DashboardChartOption;
  compareHistoryLines: CompareHistoryLine[];
  compareHistoryLoading: boolean;
  compareMetric: string;
  compareMetricKeys: string[];
  compareRuns: Array<{ run_id: string; name?: string; status?: string; duration?: number; metrics?: Record<string, number> }>;
  onClose: () => void;
  onCompareMetricChange: (value: string) => void;
};

export function CompareSection({
  compareChartOption,
  compareHistoryLines,
  compareHistoryLoading,
  compareMetric,
  compareMetricKeys,
  compareRuns,
  onClose,
  onCompareMetricChange,
}: CompareSectionProps) {
  const controls = (
    <div className="compare-controls">
      <select className="sort-select" value={compareMetric} onChange={(e) => onCompareMetricChange(e.target.value)}>
        {compareMetricKeys.map((key) => (
          <option key={`compare-${key}`} value={key}>{key}</option>
        ))}
      </select>
      <button className="action-btn subtle" onClick={onClose}>Close</button>
    </div>
  );

  const body = (
    <div className="compare-grid">
      <div className="chart-container compact">
        <EChart className="echart-surface" height={260} option={compareChartOption} />
        <div className="viz-legend">
          {compareHistoryLoading && <span className="module-empty">Loading metric history…</span>}
          {compareHistoryLines.map((entry, idx) => (
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
            {compareRuns.map((run) => (
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
  );

  return { controls, body };
}
