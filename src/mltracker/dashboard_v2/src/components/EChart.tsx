import { useEffect, useRef } from 'react';
import { init, use, type EChartsCoreOption, type EChartsType } from 'echarts/core';
import { BarChart, LineChart, ParallelChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  ParallelComponent,
  TooltipComponent,
  type GridComponentOption,
  type LegendComponentOption,
  type ParallelComponentOption,
  type TooltipComponentOption,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

use([
  BarChart,
  LineChart,
  ParallelChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  ParallelComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export type DashboardChartOption = EChartsCoreOption &
  GridComponentOption &
  LegendComponentOption &
  ParallelComponentOption &
  TooltipComponentOption;

type EChartProps = {
  className?: string;
  height?: number | string;
  onClick?: (params: unknown) => void;
  option: DashboardChartOption;
};

export function EChart({ className, height = 260, onClick, option }: EChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    const chart = init(node, undefined, { renderer: 'canvas' });
    chartRef.current = chart;

    const resize = () => chart.resize();
    const observer = new ResizeObserver(resize);
    observer.observe(node);
    window.addEventListener('resize', resize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', resize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onClick) return;

    const handleClick = (params: unknown) => onClick(params);
    chart.on('click', handleClick);

    return () => {
      chart.off('click', handleClick);
    };
  }, [onClick]);

  return <div ref={containerRef} className={className} style={{ height, width: '100%' }} />;
}
