import type { ReactNode } from 'react';
import type { MetricHistoryPoint, Run } from '../types';

export type SortMode = '-start_time' | 'start_time' | '-duration' | 'duration';
export type GroupMode = 'none' | 'status' | 'day';
export type SweepAxisMode = '' | `metric:${string}` | `param:${string}`;
export type DashboardModuleId = 'leaderboard' | 'sweep' | 'parallel' | 'importance' | 'compare';
export type DashboardModuleSpan = 1 | 2 | 3;

export type ViewStateSnapshot = {
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

export type SavedView = {
  name: string;
  createdAt: string;
  state: ViewStateSnapshot;
};

export type RunGroup = {
  label: string | null;
  runs: Run[];
};

export type PaletteAction = {
  title: string;
  meta?: string;
  run: () => void | Promise<void>;
};

export type ParsedUrlState = {
  experimentName: string;
  state: Partial<ViewStateSnapshot>;
};

export type DashboardModuleLayoutItem = {
  id: DashboardModuleId;
  span: DashboardModuleSpan;
};

export type DashboardModuleDefinition = {
  id: DashboardModuleId;
  title: string;
  controls?: ReactNode | null;
  body: ReactNode;
  visible: boolean;
};

export type ChartTheme = {
  axis: string;
  accent: string;
  accentSoft: string;
  blue: string;
  danger: string;
  muted: string;
  split: string;
  success: string;
  surface: string;
  text: string;
  tooltipBg: string;
  tooltipBorder: string;
};

export type LeaderboardRow = {
  run: Run;
  value: number;
};

export type SweepPoint = {
  run: Run;
  x: number;
  y: number;
};

export type ImportanceRow = {
  key: string;
  score: number;
  direction: number;
  samples: number;
};

export type CompareHistoryLine = {
  run: Run;
  points: MetricHistoryPoint[];
};

export const STATUS_ORDER = ['RUNNING', 'FINISHED', 'FAILED', 'CANCELED'];

export const DEFAULT_MODULE_LAYOUT: DashboardModuleLayoutItem[] = [
  { id: 'leaderboard', span: 1 },
  { id: 'sweep', span: 2 },
  { id: 'parallel', span: 3 },
  { id: 'importance', span: 3 },
  { id: 'compare', span: 3 },
];
