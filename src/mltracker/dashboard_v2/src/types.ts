export type HealthResponse = {
  status: string;
  tracker_path?: string;
  sqlite_path?: string;
  artifacts_path?: string;
};

export type Experiment = {
  name: string;
  run_count?: number;
};

export type MetricMap = Record<string, number>;
export type ParamMap = Record<string, string | number | boolean | null>;
export type TagMap = Record<string, string | number | boolean | null>;

export type Run = {
  run_id: string;
  name?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
  duration?: number;
  experiment_id?: string;
  metrics?: MetricMap;
  params?: ParamMap;
  tags?: TagMap;
};

export type RunsResponse = {
  runs: Run[];
  count?: number;
};

export type MetricHistoryPoint = {
  step: number;
  value: number;
};

export type MetricHistoryResponse = {
  metrics: Record<string, MetricHistoryPoint[]>;
};

export type Artifact = {
  path: string;
  type?: string;
};

export type ArtifactsResponse = {
  artifacts: Artifact[];
};

export type CompareResponse = {
  runs: Run[];
  metric_keys: string[];
  param_keys?: string[];
};

export type MetricCatalogEntry = {
  key: string;
  point_count: number;
  run_count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  max_step: number;
};

export type ParamCatalogEntry = {
  key: string;
  run_count: number;
  numeric_count: number;
  distinct_count: number;
  examples: string[];
  min: number | null;
  max: number | null;
  mean: number | null;
};

export type OverviewRun = {
  run_id: string;
  name?: string | null;
  status?: string;
  start_time?: string;
  end_time?: string | null;
  duration?: number | null;
  experiment_name?: string;
};

export type BestRunSummary = OverviewRun & {
  metric_key: string;
  metric_value: number;
  metric_step: number;
  objective: 'min' | 'max';
};

export type OverviewResponse = {
  experiment: { experiment_id: number; name: string } | null;
  totals: {
    runs: number;
    finished: number;
    running: number;
    failed: number;
    canceled: number;
    success_rate: number;
    avg_duration: number | null;
    recent_24h: number;
  };
  status_counts: Record<string, number>;
  metric_catalog: MetricCatalogEntry[];
  param_catalog: ParamCatalogEntry[];
  primary_metric: string | null;
  best_run: BestRunSummary | null;
  recent_runs: OverviewRun[];
};
