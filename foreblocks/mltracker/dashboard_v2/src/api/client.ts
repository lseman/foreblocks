import type {
  Artifact,
  ArtifactsResponse,
  CompareResponse,
  Experiment,
  HealthResponse,
  MetricHistoryResponse,
  Run,
  RunsResponse,
} from '../types';

function isLikelyFilesystemPath(value: string): boolean {
  return value.startsWith('/data/') || value.includes('/mltracker_data');
}

function normalizeApiBase(raw: string | undefined): string {
  const fallback =
    typeof window !== 'undefined' && window.location.protocol === 'file:'
      ? 'http://127.0.0.1:8000/api'
      : '/api';

  const value = (raw ?? '').trim();
  if (!value) return fallback;

  if (/^[a-zA-Z][a-zA-Z\d+.-]*:\/\//.test(value)) {
    return value.replace(/\/+$/, '');
  }

  if (value.startsWith('/')) {
    if (isLikelyFilesystemPath(value)) {
      console.warn(
        `[mltracker] Ignoring filesystem-like VITE_API_BASE="${value}". ` +
          'Use backend URL (e.g. http://127.0.0.1:8000/api).'
      );
      return fallback;
    }
    return value.replace(/\/+$/, '');
  }

  return `http://${value.replace(/\/+$/, '')}`;
}

const API_BASE = normalizeApiBase(import.meta.env.VITE_API_BASE);

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function sendJson<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail;
    throw new Error(detail ?? `HTTP ${res.status} ${res.statusText}`);
  }
  return body as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/health');
}

export async function getExperiments(): Promise<Experiment[]> {
  return getJson<Experiment[]>('/experiments');
}

export async function getRuns(experimentName: string): Promise<Run[]> {
  const query = new URLSearchParams({
    experiment_name: experimentName,
    limit: '200',
    include: 'full',
  });
  const data = await getJson<RunsResponse>(`/runs?${query.toString()}`);
  return data.runs ?? [];
}

export async function searchRuns(experimentName: string, query: string): Promise<Run[]> {
  const params = new URLSearchParams({
    experiment_name: experimentName,
    limit: '300',
    include: 'full',
    q: query,
  });
  const data = await getJson<RunsResponse>(`/runs?${params.toString()}`);
  return data.runs ?? [];
}

export async function getRunDetail(runId: string): Promise<Run> {
  return getJson<Run>(`/runs/${encodeURIComponent(runId)}`);
}

export async function getMetricHistory(
  runId: string,
  metricKey?: string
): Promise<MetricHistoryResponse> {
  const base = `/runs/${encodeURIComponent(runId)}/metrics/history`;
  if (!metricKey) {
    return getJson<MetricHistoryResponse>(base);
  }
  const q = new URLSearchParams({ metric_key: metricKey });
  return getJson<MetricHistoryResponse>(`${base}?${q.toString()}`);
}

export async function getArtifacts(runId: string): Promise<Artifact[]> {
  const data = await getJson<ArtifactsResponse>(`/runs/${encodeURIComponent(runId)}/artifacts`);
  return data.artifacts ?? [];
}

export async function compareRuns(runIds: string[]): Promise<CompareResponse> {
  return sendJson<CompareResponse>('/runs/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_ids: runIds }),
  });
}

export async function deleteRun(runId: string): Promise<void> {
  await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' }).then(
    async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail = (body as { detail?: string }).detail;
        throw new Error(detail ?? `HTTP ${res.status} ${res.statusText}`);
      }
    }
  );
}

export function artifactDownloadUrl(runId: string, artifactPath: string): string {
  const safePath = encodeURIComponent(artifactPath).replace(/%2F/g, '/');
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/artifacts/${safePath}`;
}
