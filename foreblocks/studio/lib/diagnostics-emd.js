import { EPS, clamp, estimateDominantFrequency, hasNonFiniteValue, mean, meanAbs, energy, meanAndStd, roundArray, countZeroCrossings, findExtrema, countExtrema, rmse } from "./diagnostics-utils.js";

const TINY = 1e-10;

function toFloat64(values) {
  return values instanceof Float64Array ? values.slice() : Float64Array.from(values);
}

function isMonotonic(values, tolerance = 1e-10) {
  const n = values.length;
  if (n < 3) return true;

  let nonDecreasing = true;
  let nonIncreasing = true;

  for (let i = 1; i < n; i += 1) {
    const d = values[i] - values[i - 1];
    if (d < -tolerance) nonDecreasing = false;
    if (d > tolerance) nonIncreasing = false;
    if (!nonDecreasing && !nonIncreasing) return false;
  }

  return true;
}

function movingAverage3(values) {
  const n = values.length;
  if (n < 3) return toFloat64(values);

  const out = new Float64Array(n);
  out[0] = (2 * values[0] + values[1]) / 3;
  for (let i = 1; i < n - 1; i += 1) {
    out[i] = (values[i - 1] + values[i] + values[i + 1]) / 3;
  }
  out[n - 1] = (values[n - 2] + 2 * values[n - 1]) / 3;
  return out;
}

function pruneExtrema(indices, sourceValues, signalStd, minSeparation = 2, prominenceFrac = 0.02) {
  if (indices.length <= 2) {
    return indices.slice();
  }

  const prominenceThreshold = prominenceFrac * Math.max(signalStd, EPS);
  const kept = [];

  for (let k = 0; k < indices.length; k += 1) {
    const idx = indices[k];
    const left = Math.max(0, idx - 1);
    const right = Math.min(sourceValues.length - 1, idx + 1);
    const localProminence = Math.max(
      Math.abs(sourceValues[idx] - sourceValues[left]),
      Math.abs(sourceValues[idx] - sourceValues[right]),
    );

    if (localProminence >= prominenceThreshold || k === 0 || k === indices.length - 1) {
      kept.push(idx);
    }
  }

  if (kept.length <= 2) {
    return kept;
  }

  const merged = [kept[0]];
  for (let i = 1; i < kept.length; i += 1) {
    const curr = kept[i];
    const prev = merged[merged.length - 1];

    if (curr - prev < minSeparation) {
      if (Math.abs(sourceValues[curr]) > Math.abs(sourceValues[prev])) {
        merged[merged.length - 1] = curr;
      }
    } else {
      merged.push(curr);
    }
  }

  return merged;
}

function buildExtremaFromDetection(modeValues, detectionValues, kind, signalStd) {
  const raw = findExtrema(detectionValues, kind);
  const prunedIndices = pruneExtrema(raw.indices, modeValues, signalStd, 2, 0.02);
  return {
    indices: prunedIndices,
    values: prunedIndices.map((idx) => modeValues[idx]),
  };
}

function reflectExtrema(indices, extremaValues, count, reflectCount = 2) {
  const m = indices.length;
  if (m < 2) {
    return { x: indices.slice(), y: extremaValues.slice() };
  }

  const leftTake = Math.min(reflectCount, m);
  const rightTake = Math.min(reflectCount, m);

  const x = [];
  const y = [];

  for (let i = leftTake - 1; i >= 0; i -= 1) {
    x.push(-indices[i]);
    y.push(extremaValues[i]);
  }

  for (let i = 0; i < m; i += 1) {
    x.push(indices[i]);
    y.push(extremaValues[i]);
  }

  const last = count - 1;
  for (let i = m - rightTake; i < m; i += 1) {
    x.push(2 * last - indices[i]);
    y.push(extremaValues[i]);
  }

  const zipped = x.map((xi, i) => ({ x: xi, y: y[i] })).sort((a, b) => a.x - b.x);

  const outX = [];
  const outY = [];
  for (let i = 0; i < zipped.length; i += 1) {
    if (i > 0 && zipped[i].x === zipped[i - 1].x) continue;
    outX.push(zipped[i].x);
    outY.push(zipped[i].y);
  }

  return { x: outX, y: outY };
}

function solveTridiagonal(a, b, c, d) {
  const n = d.length;
  const cp = new Float64Array(n);
  const dp = new Float64Array(n);
  const x = new Float64Array(n);

  cp[0] = c[0] / (b[0] + EPS);
  dp[0] = d[0] / (b[0] + EPS);

  for (let i = 1; i < n; i += 1) {
    const denom = b[i] - a[i] * cp[i - 1] + EPS;
    cp[i] = i < n - 1 ? c[i] / denom : 0;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
  }

  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i -= 1) {
    x[i] = dp[i] - cp[i] * x[i + 1];
  }

  return x;
}

function cubicSplineSecondDerivatives(xs, ys) {
  const n = xs.length;
  const m = new Float64Array(n);

  if (n < 3) {
    return m;
  }

  const a = new Float64Array(n);
  const b = new Float64Array(n);
  const c = new Float64Array(n);
  const d = new Float64Array(n);

  b[0] = 1;
  b[n - 1] = 1;
  d[0] = 0;
  d[n - 1] = 0;

  for (let i = 1; i < n - 1; i += 1) {
    const h0 = xs[i] - xs[i - 1];
    const h1 = xs[i + 1] - xs[i];
    a[i] = h0;
    b[i] = 2 * (h0 + h1);
    c[i] = h1;
    d[i] = 6 * ((ys[i + 1] - ys[i]) / (h1 + EPS) - (ys[i] - ys[i - 1]) / (h0 + EPS));
  }

  return solveTridiagonal(a, b, c, d);
}

function evaluateNaturalCubicSpline(xs, ys, second, xq) {
  const n = xs.length;
  if (n === 1) return ys[0];
  if (xq <= xs[0]) return ys[0];
  if (xq >= xs[n - 1]) return ys[n - 1];

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= xq) lo = mid;
    else hi = mid;
  }

  const h = xs[hi] - xs[lo] + EPS;
  const a = (xs[hi] - xq) / h;
  const b = (xq - xs[lo]) / h;

  return (
    a * ys[lo]
    + b * ys[hi]
    + ((a * a * a - a) * second[lo] + (b * b * b - b) * second[hi]) * (h * h) / 6
  );
}

function interpolateEnvelope(indices, extremaValues, count) {
  if (indices.length < 2 || extremaValues.length < 2) {
    return null;
  }

  const reflected = reflectExtrema(indices, extremaValues, count, 2);
  const xs = reflected.x;
  const ys = reflected.y;

  if (xs.length < 2) {
    return null;
  }

  const second = cubicSplineSecondDerivatives(xs, ys);
  const envelope = new Float64Array(count);

  for (let i = 0; i < count; i += 1) {
    envelope[i] = evaluateNaturalCubicSpline(xs, ys, second, i);
  }

  return envelope;
}

function summarizeSignal(values) {
  const maxima = findExtrema(values, "max");
  const minima = findExtrema(values, "min");
  const extremaCount = maxima.indices.length + minima.indices.length;
  const { mean: avg, std } = meanAndStd(values);

  return {
    zeroCrossings: countZeroCrossings(values),
    extremaCount,
    meanAbs: meanAbs(values),
    energy: energy(values),
    mean: avg,
    std,
    maxima,
    minima,
  };
}

function normalizedSdCriterion(previous, current) {
  let total = 0;
  const n = previous.length;
  for (let i = 0; i < n; i += 1) {
    const diff = previous[i] - current[i];
    total += (diff * diff) / (previous[i] * previous[i] + EPS);
  }
  return total / Math.max(1, n);
}

function localMeanQuality(meanEnvelope, mode) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < mode.length; i += 1) {
    num += Math.abs(meanEnvelope[i]);
    den += Math.abs(mode[i]);
  }
  return num / (den + EPS);
}

function subtractInPlace(target, source) {
  for (let i = 0; i < target.length; i += 1) {
    target[i] -= source[i];
  }
}

function reconstruct(components, residual, count) {
  const out = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = residual[i];
  }
  for (let k = 0; k < components.length; k += 1) {
    const values = components[k].rawValues;
    for (let i = 0; i < count; i += 1) {
      out[i] += values[i];
    }
  }
  return out;
}

function siftEmpiricalMode(
  signal,
  maxSifts = 80,
  sdThreshold = 0.08,
  meanRatioThreshold = 0.03,
) {
  const n = signal.length;
  let mode = toFloat64(signal);
  let completedSifts = 0;

  const meanEnvelope = new Float64Array(n);
  const updated = new Float64Array(n);

  for (let iteration = 0; iteration < maxSifts; iteration += 1) {
    const detectionSignal = movingAverage3(mode);
    const { std: signalStd } = meanAndStd(mode);

    const maxima = buildExtremaFromDetection(mode, detectionSignal, "max", signalStd);
    const minima = buildExtremaFromDetection(mode, detectionSignal, "min", signalStd);

    if (maxima.indices.length < 2 || minima.indices.length < 2) {
      return completedSifts > 0 ? { values: mode, sifts: completedSifts } : null;
    }

    const upper = interpolateEnvelope(maxima.indices, maxima.values, n);
    const lower = interpolateEnvelope(minima.indices, minima.values, n);

    if (!upper || !lower) {
      return completedSifts > 0 ? { values: mode, sifts: completedSifts } : null;
    }

    for (let i = 0; i < n; i += 1) {
      meanEnvelope[i] = 0.5 * (upper[i] + lower[i]);
      updated[i] = mode[i] - meanEnvelope[i];
    }

    const sd = normalizedSdCriterion(mode, updated);
    const summary = summarizeSignal(updated);
    const meanRatio = localMeanQuality(meanEnvelope, updated);

    mode = updated.slice();
    completedSifts = iteration + 1;

    if (
      sd < sdThreshold
      && meanRatio < meanRatioThreshold
      && Math.abs(summary.zeroCrossings - summary.extremaCount) <= 1
    ) {
      break;
    }

    if (summary.std < TINY) {
      break;
    }
  }

  return { values: mode, sifts: completedSifts };
}

export function computeEmdDiagnostics(
  series,
  maxImfs = 6,
  maxSifts = 80,
  siftThreshold = 0.08,
) {
  const values = Float64Array.from(series, (point) => point.value);
  const count = values.length;

  if (count < 16) {
    return {
      available: false,
      method: "emd_cubic_spline_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Reflected cubic spline envelopes",
      reason: "EMD needs at least 16 observations.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  if (hasNonFiniteValue(values)) {
    return {
      available: false,
      method: "emd_cubic_spline_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Reflected cubic spline envelopes",
      reason: "EMD requires a complete series of finite values.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const { std: originalStd } = meanAndStd(values);
  if (originalStd < TINY) {
    return {
      available: false,
      method: "emd_cubic_spline_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Reflected cubic spline envelopes",
      reason: "EMD needs a non-constant signal.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const originalEnergy = energy(values) + EPS;
  const components = [];
  const residual = values.slice();
  let totalSifts = 0;

  for (let imfIndex = 0; imfIndex < maxImfs; imfIndex += 1) {
    const residualEnergyShare = energy(residual) / originalEnergy;
    if (residualEnergyShare < 1e-8 || isMonotonic(residual)) {
      break;
    }

    const sifted = siftEmpiricalMode(
      residual,
      maxSifts,
      clamp(siftThreshold, 0.01, 0.5),
      0.03,
    );

    if (!sifted) {
      break;
    }

    const componentValues = sifted.values;
    const summary = summarizeSignal(componentValues);

    if (summary.std < TINY * Math.max(1, originalStd)) {
      break;
    }

    components.push({
      order: components.length + 1,
      name: `IMF ${components.length + 1}`,
      rawValues: componentValues,
      values: roundArray(componentValues, 6),
      energyShare: Number((summary.energy / originalEnergy).toFixed(4)),
      zeroCrossings: summary.zeroCrossings,
      extremaCount: summary.extremaCount,
      meanAbs: Number(summary.meanAbs.toFixed(4)),
      sifts: sifted.sifts,
      frequency: Number(estimateDominantFrequency(componentValues).toFixed(6)),
    });

    subtractInPlace(residual, componentValues);
    totalSifts += sifted.sifts;
  }

  if (components.length === 0) {
    return {
      available: false,
      method: "emd_cubic_spline_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Reflected cubic spline envelopes",
      reason: "The signal became monotonic before an IMF could be extracted.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const rebuilt = reconstruct(components, residual, count);
  const reconstructionError = rmse(rebuilt, values);
  const residualSummary = summarizeSignal(residual);
  const residualEnergy = residualSummary.energy;

  return {
    available: true,
    method: "emd_cubic_spline_sifting",
    methodLabel: "Empirical mode decomposition",
    interpolationLabel: "Reflected cubic spline envelopes",
    reason: "",
    components: components.map(({ rawValues, ...component }) => component),
    residual: {
      name: "Residual",
      values: roundArray(residual, 6),
      energyShare: Number((residualEnergy / originalEnergy).toFixed(4)),
      zeroCrossings: residualSummary.zeroCrossings,
      extremaCount: residualSummary.extremaCount,
      meanAbs: Number(residualSummary.meanAbs.toFixed(4)),
    },
    imfCount: components.length,
    maxImfs,
    maxSifts,
    totalSifts,
    reconstructionError: Number(reconstructionError.toFixed(6)),
    residualEnergyShare: Number((residualEnergy / originalEnergy).toFixed(4)),
    siftThreshold: Number(clamp(siftThreshold, 0.01, 0.5).toFixed(3)),
  };
}

export function buildEmdOverviewData(seriesData, emdDiagnostics) {
  if (!emdDiagnostics?.available) {
    return [];
  }

  const components = Array.isArray(emdDiagnostics.components) ? emdDiagnostics.components : [];
  const residual = Array.isArray(emdDiagnostics.residual?.values) ? emdDiagnostics.residual.values : [];

  return seriesData.map((point, index) => {
    let reconstruction = residual[index] ?? 0;
    for (let i = 0; i < components.length; i += 1) {
      reconstruction += components[i]?.values?.[index] ?? 0;
    }

    return {
      ...point,
      observed: point.value,
      reconstruction,
      residual: residual[index] ?? 0,
    };
  });
}

export function buildEmdComponentSeries(seriesData, component) {
  return seriesData.map((point, index) => ({
    ...point,
    value: component?.values?.[index] ?? 0,
  }));
}
