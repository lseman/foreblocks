import type { MetricHistoryPoint, Run } from '../types';
import type { DashboardChartOption } from '../components/EChart';
import type { ChartTheme, CompareHistoryLine, ImportanceRow, SweepAxisMode, SweepPoint } from './types';
import {
  colorForIndex,
  dimensionLabel,
  escapeHtml,
  formatMetricValue,
  parallelDimensionValue,
  sweepAxisLabel,
} from './utils';

export function getChartTheme(theme: 'dark' | 'light'): ChartTheme {
  return theme === 'light'
    ? {
        axis: 'rgba(17, 17, 20, 0.28)',
        accent: '#f6a623',
        accentSoft: 'rgba(246, 166, 35, 0.2)',
        blue: '#4f8ef7',
        danger: '#f87171',
        muted: 'rgba(17, 17, 20, 0.52)',
        split: 'rgba(17, 17, 20, 0.08)',
        success: '#34d399',
        surface: '#ffffff',
        text: '#111114',
        tooltipBg: 'rgba(255, 255, 255, 0.98)',
        tooltipBorder: 'rgba(79, 142, 247, 0.24)',
      }
    : {
        axis: 'rgba(240, 240, 242, 0.18)',
        accent: '#f6a623',
        accentSoft: 'rgba(246, 166, 35, 0.18)',
        blue: '#4f8ef7',
        danger: '#f87171',
        muted: 'rgba(240, 240, 242, 0.54)',
        split: 'rgba(255, 255, 255, 0.08)',
        success: '#34d399',
        surface: '#11161f',
        text: '#f0f0f2',
        tooltipBg: 'rgba(12, 18, 27, 0.96)',
        tooltipBorder: 'rgba(79, 142, 247, 0.28)',
      };
}

export function buildSweepChartOption(args: {
  chartTheme: ChartTheme;
  selectedSet: Set<string>;
  sweepPoints: SweepPoint[];
  sweepXAxis: SweepAxisMode;
  sweepYAxis: SweepAxisMode;
}): DashboardChartOption {
  const { chartTheme, selectedSet, sweepPoints, sweepXAxis, sweepYAxis } = args;
  const xName = sweepAxisLabel(sweepXAxis);
  const yName = sweepAxisLabel(sweepYAxis);

  return {
    animationDuration: 260,
    grid: { top: 20, right: 18, bottom: 44, left: 58 },
    tooltip: {
      trigger: 'item',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: chartTheme.text, fontFamily: 'Inter, sans-serif' },
      formatter: (raw) => {
        const params = raw as {
          data?: { runId?: string; runName?: string; status?: string; xValue?: number; yValue?: number };
        };
        const data = params.data;
        if (!data) return '';
        return [
          `<div style="font-weight:700;margin-bottom:4px;">${escapeHtml(data.runName ?? data.runId ?? 'run')}</div>`,
          `<div style="color:${chartTheme.muted};font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:6px;">${escapeHtml(data.runId ?? '')}</div>`,
          `<div style="font-size:11px;margin-bottom:3px;">status: ${escapeHtml(data.status ?? 'UNKNOWN')}</div>`,
          `<div style="font-size:11px;">${escapeHtml(xName)}: ${escapeHtml(formatMetricValue(data.xValue))}</div>`,
          `<div style="font-size:11px;">${escapeHtml(yName)}: ${escapeHtml(formatMetricValue(data.yValue))}</div>`,
        ].join('');
      },
    },
    xAxis: {
      type: 'value',
      name: xName === '-' ? '' : xName,
      nameLocation: 'middle',
      nameGap: 28,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    yAxis: {
      type: 'value',
      name: yName === '-' ? '' : yName,
      nameLocation: 'middle',
      nameGap: 42,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    series: [
      {
        type: 'scatter',
        symbolSize: (value: unknown, params: { data?: { runId?: string } }) =>
          params.data?.runId && selectedSet.has(params.data.runId) ? 11 : 8,
        emphasis: {
          scale: 1.25,
          itemStyle: {
            borderWidth: 2,
            borderColor: chartTheme.surface,
          },
        },
        data: sweepPoints.map((point) => ({
          value: [point.x, point.y],
          runId: point.run.run_id,
          runName: point.run.name ?? point.run.run_id,
          status: point.run.status ?? 'UNKNOWN',
          xValue: point.x,
          yValue: point.y,
          itemStyle: {
            color:
              (point.run.status ?? '').toUpperCase() === 'FAILED'
                ? chartTheme.danger
                : selectedSet.has(point.run.run_id)
                  ? chartTheme.accent
                  : chartTheme.blue,
            opacity: selectedSet.size && !selectedSet.has(point.run.run_id) ? 0.32 : 0.84,
          },
        })),
      },
    ],
  };
}

export function buildParallelChartOption(args: {
  chartTheme: ChartTheme;
  filteredRuns: Run[];
  parallelActiveDims: string[];
  selectedSet: Set<string>;
}): DashboardChartOption {
  const { chartTheme, filteredRuns, parallelActiveDims, selectedSet } = args;
  const axisData = parallelActiveDims.map((dim, idx) => {
    const values = filteredRuns
      .map((run) => parallelDimensionValue(run, dim))
      .filter((value): value is number => value !== null);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;

    return {
      dim: idx,
      name: dimensionLabel(dim),
      min,
      max: min === max ? max + 1 : max,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      nameTextStyle: { color: chartTheme.text, fontSize: 11 },
      splitLine: { lineStyle: { color: chartTheme.split } },
    };
  });

  return {
    animationDuration: 280,
    tooltip: {
      trigger: 'item',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: chartTheme.text, fontFamily: 'Inter, sans-serif' },
      formatter: (raw) => {
        const params = raw as {
          data?: {
            runId?: string;
            runName?: string;
            status?: string;
            dims?: string[];
            value?: number[];
          };
        };
        const data = params.data;
        if (!data) return '';
        const rows = (data.value ?? [])
          .map((value, idx) => {
            const dim = data.dims?.[idx];
            return dim
              ? `<div style="font-size:11px;">${escapeHtml(dimensionLabel(dim))}: ${escapeHtml(formatMetricValue(value))}</div>`
              : '';
          })
          .join('');
        return [
          `<div style="font-weight:700;margin-bottom:4px;">${escapeHtml(data.runName ?? data.runId ?? 'run')}</div>`,
          `<div style="color:${chartTheme.muted};font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:6px;">${escapeHtml(data.runId ?? '')}</div>`,
          `<div style="font-size:11px;margin-bottom:6px;">status: ${escapeHtml(data.status ?? 'UNKNOWN')}</div>`,
          rows,
        ].join('');
      },
    },
    parallel: {
      left: 42,
      right: 24,
      top: 30,
      bottom: 36,
      parallelAxisDefault: {
        type: 'value',
        nameLocation: 'end',
        nameGap: 10,
      },
    },
    parallelAxis: axisData,
    series: [
      {
        type: 'parallel',
        smooth: 0.18,
        lineStyle: {
          width: 1.6,
          opacity: 0.48,
        },
        emphasis: {
          lineStyle: {
            width: 2.8,
            opacity: 0.96,
          },
        },
        data: filteredRuns
          .slice(0, 80)
          .map((run, rowIdx) => {
            const values = parallelActiveDims.map((dim) => parallelDimensionValue(run, dim));
            if (values.some((value) => value === null)) return null;
            const highlighted = selectedSet.has(run.run_id);
            return {
              value: values as number[],
              dims: [...parallelActiveDims],
              runId: run.run_id,
              runName: run.name ?? run.run_id,
              status: run.status ?? 'UNKNOWN',
              lineStyle: {
                color: highlighted ? chartTheme.accent : colorForIndex(rowIdx),
                opacity: selectedSet.size && !highlighted ? 0.16 : highlighted ? 0.95 : 0.46,
                width: highlighted ? 2.4 : 1.4,
              },
            };
          })
          .filter((entry): entry is NonNullable<typeof entry> => entry !== null),
      },
    ],
  };
}

export function buildImportanceChartOption(args: {
  chartTheme: ChartTheme;
  importanceRows: ImportanceRow[];
}): DashboardChartOption {
  const { chartTheme, importanceRows } = args;
  const rows = [...importanceRows].reverse();
  return {
    animationDuration: 260,
    grid: { top: 18, right: 18, bottom: 18, left: 132 },
    tooltip: {
      trigger: 'item',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: chartTheme.text, fontFamily: 'Inter, sans-serif' },
      formatter: (raw) => {
        const params = raw as { data?: { name?: string; value?: number; direction?: number; samples?: number } };
        const data = params.data;
        if (!data) return '';
        return [
          `<div style="font-weight:700;margin-bottom:4px;">${escapeHtml(data.name ?? '')}</div>`,
          `<div style="font-size:11px;">score: ${escapeHtml(formatMetricValue(data.value))}</div>`,
          `<div style="font-size:11px;">signal: ${data.direction && data.direction < 0 ? 'negative' : 'positive'}</div>`,
          `<div style="font-size:11px;">samples: ${escapeHtml(data.samples ?? 0)}</div>`,
        ].join('');
      },
    },
    xAxis: {
      type: 'value',
      max: 1,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    yAxis: {
      type: 'category',
      data: rows.map((row) => row.key),
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.text },
    },
    series: [
      {
        type: 'bar',
        barWidth: 12,
        label: {
          show: true,
          position: 'right',
          color: chartTheme.muted,
          formatter: (raw: { data?: { value?: number } }) => formatMetricValue(raw.data?.value),
        },
        data: rows.map((row) => ({
          value: row.score,
          name: row.key,
          direction: row.direction,
          samples: row.samples,
          itemStyle: {
            borderRadius: [0, 999, 999, 0],
            color: row.direction >= 0 ? chartTheme.success : chartTheme.danger,
          },
        })),
      },
    ],
  };
}

export function buildCompareChartOption(args: {
  chartTheme: ChartTheme;
  compareHistoryLines: CompareHistoryLine[];
  compareMetric: string;
}): DashboardChartOption {
  const { chartTheme, compareHistoryLines, compareMetric } = args;
  return {
    animationDuration: 260,
    grid: { top: 26, right: 22, bottom: 44, left: 58 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: chartTheme.text, fontFamily: 'Inter, sans-serif' },
      formatter: (raw) => {
        const params = Array.isArray(raw) ? raw : [raw];
        const header = params[0] && 'axisValueLabel' in params[0] ? String(params[0].axisValueLabel ?? '') : '';
        const rows = params
          .map((entry) => {
            const seriesName = 'seriesName' in entry ? String(entry.seriesName ?? '') : '';
            const value = 'data' in entry && Array.isArray(entry.data) ? entry.data[1] : undefined;
            const color = 'color' in entry ? String(entry.color ?? chartTheme.blue) : chartTheme.blue;
            return `<div style="font-size:11px;"><span style="display:inline-block;width:8px;height:8px;border-radius:999px;background:${color};margin-right:6px;"></span>${escapeHtml(seriesName)}: ${escapeHtml(formatMetricValue(value))}</div>`;
          })
          .join('');
        return `<div style="font-size:11px;margin-bottom:6px;">step ${escapeHtml(header)}</div>${rows}`;
      },
    },
    legend: {
      top: 0,
      textStyle: { color: chartTheme.muted, fontSize: 11 },
    },
    xAxis: {
      type: 'value',
      name: 'step',
      nameLocation: 'middle',
      nameGap: 28,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    yAxis: {
      type: 'value',
      name: compareMetric || undefined,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    series: compareHistoryLines.map((entry, idx) => ({
      type: 'line',
      name: entry.run.name ?? entry.run.run_id,
      showSymbol: entry.points.length <= 24,
      symbolSize: 6,
      smooth: 0.12,
      lineStyle: {
        width: 2.2,
        color: colorForIndex(idx),
      },
      itemStyle: {
        color: colorForIndex(idx),
      },
      data: entry.points.map((point) => [point.step, point.value]),
    })),
  };
}

export function buildRunDetailChartOption(args: {
  chartTheme: ChartTheme;
  runDetailChartPoints: MetricHistoryPoint[] | null;
  runDetailMetricKey: string;
}): DashboardChartOption | null {
  const { chartTheme, runDetailChartPoints, runDetailMetricKey } = args;
  if (!runDetailChartPoints || !runDetailMetricKey) return null;
  return {
    animationDuration: 240,
    grid: { top: 22, right: 24, bottom: 42, left: 58 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: chartTheme.text, fontFamily: 'Inter, sans-serif' },
      formatter: (raw) => {
        const params = Array.isArray(raw) ? raw : [raw];
        const header = params[0] && 'axisValueLabel' in params[0] ? String(params[0].axisValueLabel ?? '') : '';
        const value = params[0] && 'data' in params[0] && Array.isArray(params[0].data) ? params[0].data[1] : undefined;
        return [
          `<div style="font-size:11px;margin-bottom:6px;">step ${escapeHtml(header)}</div>`,
          `<div style="font-weight:700;">${escapeHtml(runDetailMetricKey)}: ${escapeHtml(formatMetricValue(value))}</div>`,
        ].join('');
      },
    },
    xAxis: {
      type: 'value',
      name: 'step',
      nameLocation: 'middle',
      nameGap: 28,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    yAxis: {
      type: 'value',
      name: runDetailMetricKey,
      axisLine: { lineStyle: { color: chartTheme.axis } },
      axisLabel: { color: chartTheme.muted },
      splitLine: { lineStyle: { color: chartTheme.split } },
    },
    series: [
      {
        type: 'line',
        name: runDetailMetricKey,
        smooth: 0.16,
        showSymbol: runDetailChartPoints.length <= 40,
        symbolSize: 6,
        areaStyle: {
          color: chartTheme.accentSoft,
        },
        lineStyle: {
          color: chartTheme.accent,
          width: 2.4,
        },
        itemStyle: {
          color: chartTheme.blue,
        },
        data: runDetailChartPoints.map((point) => [point.step, point.value]),
      },
    ],
  };
}
