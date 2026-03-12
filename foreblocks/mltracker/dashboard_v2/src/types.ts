export type HealthResponse = {
  status: string;
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
};
