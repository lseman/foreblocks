function average(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function variance(values, mean = average(values)) {
  return values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
}

function standardDeviation(values, mean = average(values)) {
  return Math.sqrt(variance(values, mean));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function movingAverage(values, windowSize) {
  const radius = Math.max(1, Math.floor(windowSize / 2));
  return values.map((_value, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(values.length, index + radius + 1);
    return average(values.slice(start, end));
  });
}

function centerSeasonalPattern(pattern) {
  const mean = average(pattern);
  return pattern.map((value) => value - mean);
}

function buildSeasonalComponent(values, period) {
  const buckets = Array.from({ length: period }, () => []);
  values.forEach((value, index) => {
    buckets[index % period].push(value);
  });
  const pattern = centerSeasonalPattern(
    buckets.map((bucket) => (bucket.length > 0 ? average(bucket) : 0)),
  );
  return values.map((_value, index) => pattern[index % period]);
}

export function inferSeasonalPeriods(acfPeaks, count) {
  const candidates = acfPeaks.map((point) => point.lag).filter((lag) => lag >= 4 && lag <= Math.max(4, Math.floor(count / 3)));
  const unique = [];
  for (const lag of candidates) {
    if (!unique.some((value) => Math.abs(value - lag) <= 2 || lag % value === 0)) {
      unique.push(lag);
    }
    if (unique.length === 2) {
      break;
    }
  }
  if (unique.length === 0) {
    unique.push(clamp(Math.floor(count / 12), 4, 24));
  }
  return unique;
}

export function buildMstlStyleDecomposition(series, seasonalPeriods) {
  const observed = series.map((point) => point.value);
  const periods = seasonalPeriods.length > 0 ? seasonalPeriods : [12];
  const seasonalComponents = periods.map((period) => ({ period, series: Array.from({ length: observed.length }, () => 0), strength: 0 }));
  let trend = movingAverage(observed, Math.max(5, periods[0] * 2 - 1));

  for (let iteration = 0; iteration < 2; iteration += 1) {
    seasonalComponents.forEach((component, componentIndex) => {
      const otherSeasonal = observed.map((_value, index) => seasonalComponents.reduce((total, current, indexInner) => total + (indexInner === componentIndex ? 0 : current.series[index]), 0));
      const detrended = observed.map((value, index) => value - trend[index] - otherSeasonal[index]);
      component.series = buildSeasonalComponent(detrended, component.period);
    });

    const deseasonalized = observed.map((value, index) => value - seasonalComponents.reduce((total, component) => total + component.series[index], 0));
    trend = movingAverage(observed.map((value, index) => value - seasonalComponents.reduce((total, component) => total + component.series[index], 0)), Math.max(5, Math.min(observed.length - 1, periods[0] * 2 + 1)));
  }

  const combinedSeasonal = observed.map((_value, index) => seasonalComponents.reduce((total, component) => total + component.series[index], 0));
  const residual = observed.map((value, index) => value - trend[index] - combinedSeasonal[index]);
  const deseasonalized = observed.map((value, index) => value - trend[index]);

  seasonalComponents.forEach((component) => {
    const seasonalPlusResidual = component.series.map((value, index) => value + residual[index]);
    component.strength = clamp(1 - variance(residual) / Math.max(1e-12, variance(seasonalPlusResidual)), 0, 1);
  });

  const combinedStrength = clamp(1 - variance(residual) / Math.max(1e-12, variance(deseasonalized)), 0, 1);

  return {
    periods,
    strength: combinedStrength,
    components: seasonalComponents.map((component) => ({ period: component.period, strength: Number(component.strength.toFixed(3)) })),
    data: series.map((point, index) => ({ ...point, observed: Number(observed[index].toFixed(4)), trend: Number(trend[index].toFixed(4)), seasonal: Number(combinedSeasonal[index].toFixed(4)), residual: Number(residual[index].toFixed(4)) })),
    residualStd: Number(standardDeviation(residual, 0).toFixed(4)),
  };
}
