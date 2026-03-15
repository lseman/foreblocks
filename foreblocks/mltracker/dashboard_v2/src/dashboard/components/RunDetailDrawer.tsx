import { Fragment } from 'react';
import { artifactDownloadUrl } from '../../api/client';
import { EChart, type DashboardChartOption } from '../../components/EChart';
import type { Artifact, Run } from '../../types';
import { formatDuration, formatMetricValue } from '../utils';

type RunDetailDrawerProps = {
  artifacts: Artifact[];
  heroMetrics: Array<[string, number]>;
  isOpen: boolean;
  metricKey: string;
  metricKeys: string[];
  onClose: () => void;
  onMetricKeyChange: (value: string) => void;
  run: Run | null;
  runDetailChartOption: DashboardChartOption | null;
  runDetailChartPointCount: number;
};

export function RunDetailDrawer({
  artifacts,
  heroMetrics,
  isOpen,
  metricKey,
  metricKeys,
  onClose,
  onMetricKeyChange,
  run,
  runDetailChartOption,
  runDetailChartPointCount,
}: RunDetailDrawerProps) {
  if (!isOpen || !run) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <h3>{run.name ?? run.run_id}</h3>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>

        <section className="run-hero">
          {heroMetrics.length ? (
            heroMetrics.map(([key, value]) => (
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
            {metricKeys.length > 0 && (
              <select
                className="sort-select"
                value={metricKey}
                onChange={(e) => onMetricKeyChange(e.target.value)}
              >
                {metricKeys.map((key) => (
                  <option key={`detail-metric-${key}`} value={key}>
                    {key}
                  </option>
                ))}
              </select>
            )}
          </div>
          {runDetailChartOption ? (
            <>
              <div className="chart-container compact">
                <EChart className="echart-surface" height={260} option={runDetailChartOption} />
              </div>
              <div className="viz-legend">
                <span className="viz-tag">
                  <i style={{ backgroundColor: '#f6a623' }} />
                  {metricKey}
                </span>
                <span className="viz-tag">{runDetailChartPointCount} points</span>
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
            <span className="kv-v">{run.run_id}</span>
            <span className="kv-k">status</span>
            <span className="kv-v">{run.status ?? '-'}</span>
            <span className="kv-k">duration</span>
            <span className="kv-v">{formatDuration(run.duration)}</span>
            <span className="kv-k">started</span>
            <span className="kv-v">{run.start_time ? new Date(run.start_time).toLocaleString() : '-'}</span>
          </div>
        </div>

        <div className="drawer-section">
          <h4>Params</h4>
          <div className="kv-grid">
            {Object.entries(run.params ?? {}).map(([key, value]) => (
              <Fragment key={`pk-${key}`}>
                <span className="kv-k">{key}</span>
                <span className="kv-v">{String(value)}</span>
              </Fragment>
            ))}
          </div>
        </div>

        <div className="drawer-section">
          <h4>Tags</h4>
          <div className="tags-cloud">
            {Object.entries(run.tags ?? {}).map(([key, value]) => (
              <span key={`tag-${key}`} className="tag-chip">{key}:{String(value)}</span>
            ))}
          </div>
        </div>

        <div className="drawer-section">
          <h4>Metrics (Latest)</h4>
          <div className="kv-grid">
            {Object.entries(run.metrics ?? {}).map(([key, value]) => (
              <Fragment key={`mk-${key}`}>
                <span className="kv-k">{key}</span>
                <span className="kv-v">{formatMetricValue(value)}</span>
              </Fragment>
            ))}
          </div>
        </div>

        <div className="drawer-section">
          <h4>Artifacts</h4>
          <ul className="artifact-list">
            {artifacts.map((artifact) => (
              <li key={artifact.path}>
                <span>{artifact.path}</span>
                <a className="artifact-link" href={artifactDownloadUrl(run.run_id, artifact.path)} target="_blank" rel="noreferrer">
                  Download
                </a>
              </li>
            ))}
            {!artifacts.length && <li className="module-empty">No artifacts logged for this run.</li>}
          </ul>
        </div>

        <div className="drawer-section">
          <h4>Metric History Keys</h4>
          <p className="mono-copy">
            {metricKeys.join(', ') || 'No history'}
          </p>
        </div>
      </aside>
    </div>
  );
}
