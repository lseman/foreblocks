import { computeEmdDiagnostics } from "./diagnostics-emd.js";
import { estimateDominantFrequency, hasNonFiniteValue } from "./diagnostics-utils.js";

const EPS = 1e-12;

function mean(values) {
  const n = values.length;
  if (n === 0) return 0;

  let total = 0;
  for (let i = 0; i < n; i += 1) {
    total += values[i];
  }
  return total / n;
}

function standardDeviation(values, meanValue = mean(values)) {
  const n = values.length;
  if (n === 0) return 0;

  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = values[i] - meanValue;
    total += diff * diff;
  }
  return Math.sqrt(total / n);
}

function roundArray(values, digits = 6) {
  return Array.from(values, (value) => Number(value.toFixed(digits)));
}

function meanInteger(values) {
  return Math.round(mean(values));
}

function gaussianNoise() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function buildNoiseVector(length, noiseStd) {
  const noise = new Float64Array(length);
  for (let i = 0; i < length; i += 1) {
    noise[i] = gaussianNoise() * noiseStd;
  }
  return noise;
}

function addNoiseVectorToSeries(series, noise, sign = 1) {
  const out = new Array(series.length);
  for (let i = 0; i < series.length; i += 1) {
    out[i] = {
      ...series[i],
      value: series[i].value + sign * noise[i],
    };
  }
  return out;
}

function averageArrayCollection(collection) {
  const rows = collection.length;
  const length = collection[0]?.length ?? 0;
  const output = new Float64Array(length);

  if (rows === 0 || length === 0) {
    return output;
  }

  for (let j = 0; j < rows; j += 1) {
    const row = collection[j];
    for (let i = 0; i < length; i += 1) {
      output[i] += row[i];
    }
  }

  for (let i = 0; i < length; i += 1) {
    output[i] /= rows;
  }

  return output;
}

function energy(values) {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i] * values[i];
  }
  return total;
}

function countZeroCrossings(values) {
  const n = values.length;
  if (n < 2) return 0;

  let count = 0;
  let prev = values[0];

  for (let i = 1; i < n; i += 1) {
    const curr = values[i];
    if ((prev < 0 && curr >= 0) || (prev > 0 && curr <= 0)) {
      count += 1;
    }
    if (curr !== 0) prev = curr;
  }

  return count;
}

function findExtrema(values, kind = "max", tolerance = 1e-12) {
  const n = values.length;
  if (n < 3) {
    return { indices: [], values: [] };
  }

  const indices = [];
  let i = 1;
  const isMax = kind === "max";

  const better = (a, b) => (isMax ? a > b + tolerance : a < b - tolerance);
  const notWorse = (a, b) => (isMax ? a >= b - tolerance : a <= b + tolerance);

  if (better(values[0], values[1])) {
    indices.push(0);
  }

  while (i < n - 1) {
    const prev = values[i - 1];
    const curr = values[i];
    const next = values[i + 1];

    if (better(curr, prev) && notWorse(curr, next)) {
      indices.push(i);
      i += 1;
      continue;
    }

    if (Math.abs(curr - next) <= tolerance) {
      let left = i;
      let right = i + 1;
      while (right < n - 1 && Math.abs(values[right] - values[right + 1]) <= tolerance) {
        right += 1;
      }

      const leftNeighbor = values[left - 1];
      const plateauValue = values[left];
      const rightNeighbor = values[right + 1];

      if (better(plateauValue, leftNeighbor) && better(plateauValue, rightNeighbor)) {
        indices.push(((left + right) / 2) | 0);
      }

      i = right + 1;
      continue;
    }

    i += 1;
  }

  if (better(values[n - 1], values[n - 2])) {
    indices.push(n - 1);
  }

  const uniqueIndices = [...new Set(indices)].sort((a, b) => a - b);
  return {
    indices: uniqueIndices,
    values: uniqueIndices.map((index) => values[index]),
  };
}

function countExtrema(values) {
  return findExtrema(values, "max").indices.length + findExtrema(values, "min").indices.length;
}

function summarizeValues(values) {
  const absMean = mean(values.map((value) => Math.abs(value)));
  return {
    zeroCrossings: countZeroCrossings(values),
    extremaCount: countExtrema(values),
    meanAbs: absMean,
    energy: energy(values),
  };
}

function rmse(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;

  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = a[i] - b[i];
    total += diff * diff;
  }
  return Math.sqrt(total / n);
}

function computeResidualFromComponents(originalValues, components) {
  const residual = new Float64Array(originalValues.length);

  for (let i = 0; i < originalValues.length; i += 1) {
    let value = originalValues[i];
    for (let k = 0; k < components.length; k += 1) {
      value -= components[k][i];
    }
    residual[i] = value;
  }

  return residual;
}

function reconstructFromComponents(components, residual) {
  const out = new Float64Array(residual.length);

  for (let i = 0; i < residual.length; i += 1) {
    out[i] = residual[i];
  }

  for (let k = 0; k < components.length; k += 1) {
    const comp = components[k];
    for (let i = 0; i < out.length; i += 1) {
      out[i] += comp[i];
    }
  }

  return out;
}

function normalizeRun(run) {
  return {
    ...run,
    components: run.components.map((component) => ({
      ...component,
      values: Float64Array.from(component.values),
    })),
    residual: run.residual
      ? {
          ...run.residual,
          values: Float64Array.from(run.residual.values),
        }
      : null,
  };
}

export function computeEemdDiagnostics(
  series,
  ensembleSize = 16,
  noiseStdRatio = 0.12,
  maxImfs = 6,
  maxSifts = 60,
  siftThreshold = 0.05,
) {
  const originalValues = Float64Array.from(series, (point) => point.value);
  if (hasNonFiniteValue(originalValues)) {
    return computeEmdDiagnostics(series, maxImfs, maxSifts, siftThreshold);
  }

  const signalStd = standardDeviation(originalValues);
  const noiseStd = Math.max(EPS, signalStd * noiseStdRatio);

  if (originalValues.length < 16 || signalStd < 1e-10) {
    return computeEmdDiagnostics(series, maxImfs, maxSifts, siftThreshold);
  }

  const acceptedRuns = [];

  for (let runIndex = 0; runIndex < ensembleSize; runIndex += 1) {
    const noise = buildNoiseVector(series.length, noiseStd);

    const plusSeries = addNoiseVectorToSeries(series, noise, +1);
    const minusSeries = addNoiseVectorToSeries(series, noise, -1);

    const plusRun = computeEmdDiagnostics(plusSeries, maxImfs, maxSifts, siftThreshold);
    const minusRun = computeEmdDiagnostics(minusSeries, maxImfs, maxSifts, siftThreshold);

    if (plusRun.available && Array.isArray(plusRun.components) && plusRun.components.length > 0) {
      acceptedRuns.push(normalizeRun(plusRun));
    }

    if (minusRun.available && Array.isArray(minusRun.components) && minusRun.components.length > 0) {
      acceptedRuns.push(normalizeRun(minusRun));
    }
  }

  if (acceptedRuns.length === 0) {
    return computeEmdDiagnostics(series, maxImfs, maxSifts, siftThreshold);
  }

  const componentCount = Math.min(
    maxImfs,
    ...acceptedRuns.map((run) => run.components.length),
  );

  if (componentCount <= 0) {
    return computeEmdDiagnostics(series, maxImfs, maxSifts, siftThreshold);
  }

  const averagedComponents = [];
  for (let componentIndex = 0; componentIndex < componentCount; componentIndex += 1) {
    const collection = acceptedRuns.map((run) => run.components[componentIndex].values);
    averagedComponents.push(averageArrayCollection(collection));
  }

  const residual = computeResidualFromComponents(originalValues, averagedComponents);
  const reconstructed = reconstructFromComponents(averagedComponents, residual);
  const reconstructionError = rmse(reconstructed, originalValues);

  const originalEnergy = energy(originalValues) + EPS;
  const residualSummary = summarizeValues(Array.from(residual));
  const components = [];

  for (let componentIndex = 0; componentIndex < componentCount; componentIndex += 1) {
    const componentValues = averagedComponents[componentIndex];
    const summary = summarizeValues(Array.from(componentValues));

    components.push({
      order: componentIndex + 1,
      name: `IMF ${componentIndex + 1}`,
      values: roundArray(componentValues, 6),
      energyShare: Number((summary.energy / originalEnergy).toFixed(4)),
      zeroCrossings: summary.zeroCrossings,
      extremaCount: summary.extremaCount,
      meanAbs: Number(summary.meanAbs.toFixed(4)),
      sifts: meanInteger(
        acceptedRuns.map((run) => run.components[componentIndex]?.sifts ?? 0),
      ),
      frequency: Number(estimateDominantFrequency(componentValues).toFixed(6)),
    });
  }

  return {
    available: true,
    method: "eemd_cubic_spline_sifting",
    methodLabel: "Ensemble empirical mode decomposition",
    interpolationLabel: acceptedRuns[0].interpolationLabel ?? "Reflected cubic spline envelopes",
    reason: "",
    components,
    residual: {
      name: "Residual",
      values: roundArray(residual, 6),
      energyShare: Number((residualSummary.energy / originalEnergy).toFixed(4)),
      zeroCrossings: residualSummary.zeroCrossings,
      extremaCount: residualSummary.extremaCount,
      meanAbs: Number(residualSummary.meanAbs.toFixed(4)),
    },
    imfCount: componentCount,
    maxImfs,
    totalSifts: meanInteger(acceptedRuns.map((run) => run.totalSifts ?? 0)),
    reconstructionError: Number(reconstructionError.toFixed(6)),
    residualEnergyShare: Number((residualSummary.energy / originalEnergy).toFixed(4)),
    siftThreshold: Number(siftThreshold.toFixed(3)),
    ensembleSize: acceptedRuns.length,
    noiseStdRatio: Number(noiseStdRatio.toFixed(3)),
  };
}
