export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function variance(values, mean = average(values)) {
  return values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
}

function standardDeviation(values, mean = average(values)) {
  return Math.sqrt(variance(values, mean));
}

function isConstant(values, tolerance = 1e-8) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length < 2) {
    return true;
  }
  return Math.abs(Math.max(...finite) - Math.min(...finite)) <= tolerance;
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function quantile(values, ratio) {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * ratio;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }

  const weight = position - lowerIndex;
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function logGamma(value) {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019571e-6,
    1.5056327351493116e-7,
  ];

  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }

  let accumulator = 0.9999999999998099;
  const shifted = value - 1;
  for (let index = 0; index < coefficients.length; index += 1) {
    accumulator += coefficients[index] / (shifted + index + 1);
  }

  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(accumulator);
}

function betaContinuedFraction(x, a, b) {
  const maxIterations = 200;
  const epsilon = 3e-7;
  const fpMin = 1e-30;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;

  if (Math.abs(d) < fpMin) {
    d = fpMin;
  }

  d = 1 / d;
  let result = d;

  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const m2 = iteration * 2;
    let aa = (iteration * (b - iteration) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) {
      d = fpMin;
    }
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) {
      c = fpMin;
    }
    d = 1 / d;
    result *= d * c;

    aa = (-(a + iteration) * (qab + iteration) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) {
      d = fpMin;
    }
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) {
      c = fpMin;
    }
    d = 1 / d;
    const delta = d * c;
    result *= delta;

    if (Math.abs(delta - 1) < epsilon) {
      break;
    }
  }

  return result;
}

function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }

  const front = Math.exp(
    logGamma(a + b)
      - logGamma(a)
      - logGamma(b)
      + a * Math.log(x)
      + b * Math.log(1 - x),
  );

  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaContinuedFraction(x, a, b)) / a;
  }

  return 1 - (front * betaContinuedFraction(1 - x, b, a)) / b;
}

function fDistributionSurvivalProbability(fStatistic, numeratorDof, denominatorDof) {
  if (!Number.isFinite(fStatistic) || fStatistic <= 0) {
    return 1;
  }

  const x = (numeratorDof * fStatistic) / (numeratorDof * fStatistic + denominatorDof);
  return clamp(1 - regularizedIncompleteBeta(x, numeratorDof / 2, denominatorDof / 2), 0, 1);
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
  const pattern = centerSeasonalPattern(buckets.map((bucket) => (bucket.length > 0 ? average(bucket) : 0)));
  return values.map((_value, index) => pattern[index % period]);
}

function autocorrelation(values, lag, mean, seriesVariance) {
  if (lag <= 0 || lag >= values.length || seriesVariance === 0) {
    return 0;
  }

  let numerator = 0;
  for (let index = lag; index < values.length; index += 1) {
    numerator += (values[index] - mean) * (values[index - lag] - mean);
  }

  return numerator / ((values.length - lag) * seriesVariance);
}

function buildPacf(values, maxLag, mean, seriesVariance) {
  const acf = Array.from({ length: maxLag + 1 }, (_value, lag) => lag === 0 ? 1 : autocorrelation(values, lag, mean, seriesVariance));
  const phi = Array.from({ length: maxLag + 1 }, () => Array.from({ length: maxLag + 1 }, () => 0));
  const pacf = [{ lag: 0, value: 1 }];

  if (maxLag >= 1) {
    phi[1][1] = acf[1];
    pacf.push({ lag: 1, value: acf[1] });
  }

  for (let lag = 2; lag <= maxLag; lag += 1) {
    let numerator = acf[lag];
    let denominator = 1;
    for (let index = 1; index < lag; index += 1) {
      numerator -= phi[lag - 1][index] * acf[lag - index];
      denominator -= phi[lag - 1][index] * acf[index];
    }
    phi[lag][lag] = denominator === 0 ? 0 : numerator / denominator;
    for (let index = 1; index < lag; index += 1) {
      phi[lag][index] = phi[lag - 1][index] - phi[lag][lag] * phi[lag - 1][lag - index];
    }
    pacf.push({ lag, value: Number(phi[lag][lag].toFixed(4)) });
  }

  return pacf;
}

function detectCorrelationPeaks(series, threshold) {
  return series
    .filter((point) => point.lag > 0)
    .filter((point, index, items) => {
      const previous = items[index - 1]?.value ?? 0;
      const next = items[index + 1]?.value ?? 0;
      return Math.abs(point.value) >= threshold && Math.abs(point.value) >= Math.abs(previous) && Math.abs(point.value) >= Math.abs(next);
    })
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 6);
}

function buildIdentityMatrix(size) {
  return Array.from({ length: size }, (_row, rowIndex) => Array.from({ length: size }, (_value, columnIndex) => rowIndex === columnIndex ? 1 : 0));
}

function invertMatrix(matrix) {
  const size = matrix.length;
  const identity = buildIdentityMatrix(size);
  const augmented = matrix.map((row, index) => [...row.map((value) => Number(value)), ...identity[index]]);

  for (let pivot = 0; pivot < size; pivot += 1) {
    let pivotRow = pivot;
    for (let candidate = pivot + 1; candidate < size; candidate += 1) {
      if (Math.abs(augmented[candidate][pivot]) > Math.abs(augmented[pivotRow][pivot])) {
        pivotRow = candidate;
      }
    }

    const pivotValue = augmented[pivotRow][pivot];
    if (Math.abs(pivotValue) < 1e-10) {
      throw new Error("Matrix is singular.");
    }

    if (pivotRow !== pivot) {
      [augmented[pivot], augmented[pivotRow]] = [augmented[pivotRow], augmented[pivot]];
    }

    for (let column = 0; column < augmented[pivot].length; column += 1) {
      augmented[pivot][column] /= pivotValue;
    }

    for (let row = 0; row < size; row += 1) {
      if (row === pivot) {
        continue;
      }
      const factor = augmented[row][pivot];
      for (let column = 0; column < augmented[row].length; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }

  return augmented.map((row) => row.slice(size));
}

function ordinaryLeastSquares(design, target) {
  const rows = design.length;
  const columns = design[0].length;
  const xtx = Array.from({ length: columns }, () => Array.from({ length: columns }, () => 0));
  const xty = Array.from({ length: columns }, () => 0);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      xty[column] += design[row][column] * target[row];
      for (let inner = 0; inner < columns; inner += 1) {
        xtx[column][inner] += design[row][column] * design[row][inner];
      }
    }
  }

  const xtxInverse = invertMatrix(xtx);
  const coefficients = xtxInverse.map((row) => row.reduce((total, value, index) => total + value * xty[index], 0));
  const fitted = design.map((row) => row.reduce((total, value, index) => total + value * coefficients[index], 0));
  const residuals = target.map((value, index) => value - fitted[index]);
  const dof = Math.max(1, rows - columns);
  const sigmaSquared = sum(residuals.map((value) => value ** 2)) / dof;

  return { coefficients, residuals, sigmaSquared, xtxInverse };
}

function difference(values) {
  return values.slice(1).map((value, index) => value - values[index]);
}

function runAdfTest(values) {
  const sampleCount = values.length;
  const diffValues = difference(values);
  const lagOrder = Math.max(1, Math.min(8, Math.floor(Math.cbrt(sampleCount))));
  const design = [];
  const target = [];

  for (let index = lagOrder; index < diffValues.length; index += 1) {
    const row = [1, values[index]];
    for (let lag = 1; lag <= lagOrder; lag += 1) {
      row.push(diffValues[index - lag]);
    }
    design.push(row);
    target.push(diffValues[index]);
  }

  if (design.length <= design[0].length) {
    return { available: false, statistic: 0, criticalValue: -2.86, rejectUnitRoot: false, lagOrder };
  }

  const regression = ordinaryLeastSquares(design, target);
  const betaIndex = 1;
  const standardError = Math.sqrt(Math.max(1e-12, regression.sigmaSquared * regression.xtxInverse[betaIndex][betaIndex]));
  const statistic = regression.coefficients[betaIndex] / standardError;
  const criticalValue = -2.86;

  return { available: true, statistic, criticalValue, rejectUnitRoot: statistic < criticalValue, lagOrder };
}

function runKpssTest(values) {
  const sampleCount = values.length;
  const mean = average(values);
  const residuals = values.map((value) => value - mean);
  const partialSums = [];
  let cumulative = 0;
  for (const residual of residuals) {
    cumulative += residual;
    partialSums.push(cumulative);
  }

  const eta = sum(partialSums.map((value) => value ** 2)) / sampleCount ** 2;
  const bandwidth = Math.max(1, Math.min(sampleCount - 1, Math.floor(12 * ((sampleCount / 100) ** 0.25))));
  let longRunVariance = sum(residuals.map((value) => value ** 2)) / sampleCount;

  for (let lag = 1; lag <= bandwidth; lag += 1) {
    let covariance = 0;
    for (let index = lag; index < sampleCount; index += 1) {
      covariance += residuals[index] * residuals[index - lag];
    }
    covariance /= sampleCount;
    longRunVariance += 2 * (1 - lag / (bandwidth + 1)) * covariance;
  }

  const statistic = eta / Math.max(1e-12, longRunVariance);
  return { available: true, statistic, criticalValue: 0.463, rejectStationarity: statistic > 0.463, bandwidth };
}

function inferFrequencyHint(series) {
  const timestamps = series.map((item) => Date.parse(item.timestamp)).filter((value) => !Number.isNaN(value));
  if (timestamps.length < 3) {
    return "auto";
  }
  const diffs = [];
  for (let index = 1; index < timestamps.length; index += 1) {
    const diff = timestamps[index] - timestamps[index - 1];
    if (diff > 0) {
      diffs.push(diff);
    }
  }
  if (diffs.length === 0) {
    return "auto";
  }

  const medianDiff = median(diffs);
  const hour = 60 * 60 * 1000;
  const day = 24 * hour;
  if (medianDiff <= 1.5 * hour) return "H";
  if (medianDiff <= 1.5 * day) return "D";
  if (medianDiff <= 10 * day) return "W";
  if (medianDiff <= 35 * day) return "M";
  return "auto";
}

function formatPointLabel(timestamp, index) {
  const parsed = Date.parse(timestamp ?? "");
  if (!Number.isNaN(parsed)) {
    return new Date(parsed).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }

  return `T${String(index + 1).padStart(3, "0")}`;
}

const WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const HOUR_LABELS = Array.from({ length: 24 }, (_value, index) => `${String(index).padStart(2, "0")}:00`);

function buildCalendarProfile(entries, label, labels, getBucket, minimumCount = 12, minimumActiveBuckets = 2) {
  if (entries.length < minimumCount) {
    return {
      label,
      available: false,
      effectShare: 0,
      peakLabel: "",
      troughLabel: "",
      span: 0,
      data: [],
    };
  }

  const overallMean = average(entries.map((entry) => entry.value));
  const overallStd = standardDeviation(entries.map((entry) => entry.value), overallMean);
  const totalSs = sum(entries.map((entry) => (entry.value - overallMean) ** 2));
  const buckets = labels.map((bucketLabel, index) => ({
    index,
    label: bucketLabel,
    count: 0,
    sum: 0,
  }));

  entries.forEach((entry) => {
    const bucketIndex = getBucket(entry.date);
    if (bucketIndex < 0 || bucketIndex >= buckets.length) {
      return;
    }
    buckets[bucketIndex].count += 1;
    buckets[bucketIndex].sum += entry.value;
  });

  const data = buckets.map((bucket) => {
    const mean = bucket.count > 0 ? bucket.sum / bucket.count : null;
    const relativeIndex = mean == null
      ? null
      : overallStd > 1e-10
        ? (mean - overallMean) / overallStd
        : 0;
    return {
      bucket: bucket.index,
      label: bucket.label,
      count: bucket.count,
      mean: mean == null ? null : Number(mean.toFixed(4)),
      relativeIndex: relativeIndex == null ? null : Number(relativeIndex.toFixed(4)),
    };
  });

  const active = data.filter((bucket) => bucket.count > 0 && bucket.mean != null);
  if (active.length < minimumActiveBuckets) {
    return {
      label,
      available: false,
      effectShare: 0,
      peakLabel: "",
      troughLabel: "",
      span: 0,
      data,
    };
  }

  const betweenSs = sum(active.map((bucket) => bucket.count * ((bucket.mean ?? overallMean) - overallMean) ** 2));
  const peak = [...active].sort((left, right) => (right.mean ?? -Infinity) - (left.mean ?? -Infinity))[0];
  const trough = [...active].sort((left, right) => (left.mean ?? Infinity) - (right.mean ?? Infinity))[0];
  const span = (peak?.mean ?? overallMean) - (trough?.mean ?? overallMean);

  return {
    label,
    available: true,
    effectShare: Number((totalSs > 1e-12 ? betweenSs / totalSs : 0).toFixed(4)),
    peakLabel: peak?.label ?? "",
    troughLabel: trough?.label ?? "",
    span: Number(span.toFixed(4)),
    data,
  };
}

function computeCalendarDiagnostics(series, frequencyHint = "auto") {
  const entries = series
    .map((point) => {
      const parsed = Date.parse(point.timestamp);
      if (Number.isNaN(parsed)) {
        return null;
      }
      return {
        date: new Date(parsed),
        value: point.value,
      };
    })
    .filter(Boolean);

  if (entries.length < 12) {
    return {
      available: false,
      validTimestampCount: entries.length,
      strongestProfile: null,
      strongestEffectShare: 0,
      label: "insufficient",
      recommendation: "optional",
      weekdayProfile: null,
      monthProfile: null,
      hourProfile: null,
      weekendGap: 0,
    };
  }

  const weekdayProfile = buildCalendarProfile(entries, "Day of week", WEEKDAY_LABELS, (date) => date.getDay(), 14, 3);
  const monthProfile = buildCalendarProfile(entries, "Month of year", MONTH_LABELS, (date) => date.getMonth(), 18, 4);
  const hourProfile = frequencyHint === "H"
    ? buildCalendarProfile(entries, "Hour of day", HOUR_LABELS, (date) => date.getHours(), 24, 6)
    : {
      label: "Hour of day",
      available: false,
      effectShare: 0,
      peakLabel: "",
      troughLabel: "",
      span: 0,
      data: [],
    };

  const profiles = [weekdayProfile, monthProfile, hourProfile].filter((profile) => profile.available);
  if (profiles.length === 0) {
    return {
      available: false,
      validTimestampCount: entries.length,
      strongestProfile: null,
      strongestEffectShare: 0,
      label: "insufficient",
      recommendation: "optional",
      weekdayProfile,
      monthProfile,
      hourProfile,
      weekendGap: 0,
    };
  }

  const strongestProfile = [...profiles].sort((left, right) => right.effectShare - left.effectShare)[0];
  const strongestEffectShare = strongestProfile.effectShare;
  const label = strongestEffectShare >= 0.12
    ? "strong"
    : strongestEffectShare >= 0.05
      ? "moderate"
      : "weak";
  const recommendation = strongestEffectShare >= 0.05 ? "recommended" : "optional";
  const weekdayMeans = weekdayProfile.data.filter((bucket) => bucket.mean != null);
  const weekdaySeries = weekdayMeans.filter((bucket) => bucket.bucket >= 1 && bucket.bucket <= 5).map((bucket) => bucket.mean);
  const weekendSeries = weekdayMeans.filter((bucket) => bucket.bucket === 0 || bucket.bucket === 6).map((bucket) => bucket.mean);
  const weekdayMean = weekdaySeries.length > 0 ? average(weekdaySeries) : 0;
  const weekendMean = weekendSeries.length > 0 ? average(weekendSeries) : 0;
  const weekendGap = Number((Number.isFinite(weekendMean - weekdayMean) ? weekendMean - weekdayMean : 0).toFixed(4));

  return {
    available: true,
    validTimestampCount: entries.length,
    strongestProfile: strongestProfile.label,
    strongestEffectShare: Number(strongestEffectShare.toFixed(4)),
    label,
    recommendation,
    weekdayProfile,
    monthProfile,
    hourProfile,
    weekendGap,
  };
}

function buildRunLengthHistogram(runLengths) {
  const bins = new Map();
  runLengths.forEach((length) => {
    const key = length >= 8 ? "8+" : String(length);
    bins.set(key, (bins.get(key) ?? 0) + 1);
  });

  return [...bins.entries()].map(([runLength, count]) => ({
    runLength,
    count,
  })).sort((left, right) => {
    const leftValue = left.runLength === "8+" ? 8 : Number(left.runLength);
    const rightValue = right.runLength === "8+" ? 8 : Number(right.runLength);
    return leftValue - rightValue;
  });
}

function classifyIntermittency(adi, cv2, zeroShare) {
  if (zeroShare < 0.08 && adi < 1.2) {
    return "dense";
  }
  if (adi >= 1.32 && cv2 >= 0.49) {
    return "lumpy";
  }
  if (adi >= 1.32 && cv2 < 0.49) {
    return "intermittent";
  }
  if (adi < 1.32 && cv2 >= 0.49) {
    return "erratic";
  }
  return "smooth";
}

function buildIntermittencyGuidance(classification, zeroShare, adi, cv2, burstinessRatio) {
  const baselineHints = [];
  const preprocessingSuggestions = [];
  const sparseDemand = classification === "intermittent" || classification === "lumpy";
  const burstyDemand = classification === "erratic" || burstinessRatio >= 1.9;

  if (sparseDemand) {
    baselineHints.push({
      key: "croston_sba",
      label: "Croston / SBA baseline",
      suitability: "high",
      detail: `ADI ${adi.toFixed(3)} and CV² ${cv2.toFixed(3)} indicate sparse arrivals; compare against interval-demand baselines before heavier seq2seq models.`,
    });
  }

  if (sparseDemand || zeroShare >= 0.35) {
    baselineHints.push({
      key: "tsb_zero_occurrence",
      label: "TSB-style zero-occurrence baseline",
      suitability: zeroShare >= 0.5 ? "high" : "medium",
      detail: `Zero share ${(zeroShare * 100).toFixed(1)}% suggests separating occurrence smoothing from demand-size smoothing can be more faithful than dense autoregression.`,
    });
  }

  if (sparseDemand || burstyDemand || zeroShare >= 0.2) {
    baselineHints.push({
      key: "zero_inflated_direct",
      label: "Zero-inflation-friendly direct head",
      suitability: sparseDemand ? "high" : "review",
      detail: "Keep explicit zeros in the target and benchmark a direct zero-aware baseline before assuming every step should behave like dense demand.",
    });
  }

  if (sparseDemand || zeroShare >= 0.2) {
    preprocessingSuggestions.push({
      key: "preserve_zero_runs",
      label: "Preserve zero runs",
      suitability: "high",
      detail: "Avoid interpolating target zeros into small positive values; preserve the occurrence signal instead of smoothing it away.",
    });
    preprocessingSuggestions.push({
      key: "occurrence_indicator",
      label: "Add activity indicator",
      suitability: "medium",
      detail: "A binary occurrence feature can help downstream models separate whether demand happens from how large it is when active.",
    });
  }

  if (burstyDemand) {
    preprocessingSuggestions.push({
      key: "robust_burst_scaling",
      label: "Use burst-robust scaling",
      suitability: "medium",
      detail: "Large bursts are uneven enough that robust scaling or winsorization on auxiliary features is safer than mean-variance normalization alone.",
    });
  }

  if (baselineHints.length === 0) {
    baselineHints.push({
      key: "dense_baseline",
      label: "Dense baseline path",
      suitability: "optional",
      detail: "The target is active often enough that standard direct, recurrent, or transformer baselines remain reasonable defaults.",
    });
  }

  if (preprocessingSuggestions.length === 0) {
    preprocessingSuggestions.push({
      key: "standard_dense_preprocessing",
      label: "Standard dense preprocessing",
      suitability: "optional",
      detail: "No special sparse-demand preprocessing is strongly indicated beyond the existing scaling and differencing recommendations.",
    });
  }

  return {
    baselineHints: baselineHints.slice(0, 3),
    preprocessingSuggestions: preprocessingSuggestions.slice(0, 3),
  };
}

function skewness(values, mean = average(values), std = standardDeviation(values, mean)) {
  if (values.length === 0 || std < 1e-12) {
    return 0;
  }

  return average(values.map((value) => ((value - mean) / std) ** 3));
}

function buildVarianceTransformGuidance(values, volatilityLabel, archPValue, regimeRatio) {
  if (values.length === 0) {
    return {
      preferredTransform: "none",
      skewness: 0,
      minValue: 0,
      zeroShare: 0,
      suggestions: [],
    };
  }

  const minValue = Math.min(...values);
  const zeroShare = values.filter((value) => Math.abs(value) < 1e-12).length / values.length;
  const mean = average(values);
  const std = standardDeviation(values, mean);
  const asymmetry = skewness(values, mean, std);
  const transformPressure = volatilityLabel !== "stable" || archPValue < 0.1 || regimeRatio >= 1.45;
  const logEligible = minValue >= 0;
  const boxCoxEligible = minValue > 0;
  const suggestions = [];

  let logSuitability = "optional";
  if (!logEligible) {
    logSuitability = "avoid";
  } else if (transformPressure && asymmetry >= 0.9) {
    logSuitability = "recommended";
  } else if (transformPressure || asymmetry >= 0.55) {
    logSuitability = "review";
  }

  suggestions.push({
    key: "log1p",
    label: "log1p transform",
    suitability: logSuitability,
    detail: !logEligible
      ? "Negative values are present, so a plain log or log1p transform is not safe without shifting the series."
      : zeroShare > 0
        ? `Zeros are present in ${(zeroShare * 100).toFixed(1)}% of rows, so log1p is safer than plain log while still compressing bursts.`
        : `All values are non-negative and skewness ${asymmetry.toFixed(3)} suggests log compression can stabilize burst magnitudes.`,
  });

  let boxCoxSuitability = "optional";
  if (!boxCoxEligible) {
    boxCoxSuitability = "avoid";
  } else if (transformPressure && asymmetry >= 0.7) {
    boxCoxSuitability = "recommended";
  } else if (transformPressure || asymmetry >= 0.45) {
    boxCoxSuitability = "review";
  }

  suggestions.push({
    key: "box_cox",
    label: "Box-Cox transform",
    suitability: boxCoxSuitability,
    detail: !boxCoxEligible
      ? "Box-Cox requires strictly positive values, so zeros or negatives rule it out without a shift."
      : `Strictly positive support makes Box-Cox feasible; use it when variance grows with level and rolling regime ratio ${regimeRatio.toFixed(3)} keeps widening.`,
  });

  if (!logEligible && transformPressure) {
    suggestions.push({
      key: "yeo_johnson",
      label: "Yeo-Johnson fallback",
      suitability: "recommended",
      detail: "Because negatives are present, Yeo-Johnson is the safer variance-stabilizing alternative when you still want a power transform.",
    });
  } else if (transformPressure) {
    suggestions.push({
      key: "none",
      label: "No extra transform",
      suitability: "review",
      detail: "If interpretability matters more than variance stabilization, compare the raw scale against transformed runs instead of assuming the transform always helps.",
    });
  }

  const preferredTransform = suggestions.find((item) => item.suitability === "recommended")?.key ?? "none";
  return {
    preferredTransform,
    skewness: Number(asymmetry.toFixed(3)),
    minValue: Number(minValue.toFixed(4)),
    zeroShare: Number(zeroShare.toFixed(4)),
    suggestions: suggestions.slice(0, 3),
  };
}

function computeIntermittencyDiagnostics(series) {
  const values = series.map((point) => point.value);
  const count = values.length;
  if (count < 20) {
    return {
      available: false,
      label: "insufficient",
      classification: "unknown",
      recommendation: "optional",
      zeroShare: 0,
      activeShare: 0,
      adi: 0,
      cv2: 0,
      burstinessRatio: 0,
      longestIdleRun: 0,
      activeCount: 0,
      zeroLikeThreshold: 0,
      idleRunHistogram: [],
      activityTimeline: [],
      eventIntervals: [],
      baselineHints: [],
      preprocessingSuggestions: [],
    };
  }

  const absValues = values.map((value) => Math.abs(value));
  const strictlyPositive = absValues.filter((value) => value > 1e-12);
  const medianPositive = strictlyPositive.length > 0 ? median(strictlyPositive) : 0;
  const zeroLikeThreshold = Math.max(1e-8, medianPositive * 0.05, standardDeviation(values, average(values)) * 0.01);
  const activeMask = values.map((value) => Math.abs(value) > zeroLikeThreshold);
  const activeCount = activeMask.filter(Boolean).length;
  const zeroShare = 1 - activeCount / Math.max(count, 1);
  const activeShare = activeCount / Math.max(count, 1);

  const idleRunLengths = [];
  let currentIdleRun = 0;
  activeMask.forEach((isActive) => {
    if (!isActive) {
      currentIdleRun += 1;
      return;
    }
    if (currentIdleRun > 0) {
      idleRunLengths.push(currentIdleRun);
      currentIdleRun = 0;
    }
  });
  if (currentIdleRun > 0) {
    idleRunLengths.push(currentIdleRun);
  }

  const activeIndices = activeMask
    .map((isActive, index) => (isActive ? index : -1))
    .filter((index) => index >= 0);
  const eventIntervals = activeIndices.slice(1).map((index, eventIndex) => ({
    event: `E${String(eventIndex + 2).padStart(2, "0")}`,
    gap: index - activeIndices[eventIndex],
    magnitude: Number(values[index].toFixed(4)),
    timestamp: series[index]?.timestamp ?? `T${index + 1}`,
  }));
  const adi = eventIntervals.length > 0 ? average(eventIntervals.map((item) => item.gap)) : activeCount > 0 ? count / activeCount : count;
  const activeMagnitudes = activeIndices.map((index) => Math.abs(values[index]));
  const meanMagnitude = activeMagnitudes.length > 0 ? average(activeMagnitudes) : 0;
  const magnitudeCv = meanMagnitude > 1e-12 ? standardDeviation(activeMagnitudes, meanMagnitude) / meanMagnitude : 0;
  const cv2 = magnitudeCv ** 2;
  const burstinessRatio = activeMagnitudes.length > 0
    ? quantile(activeMagnitudes, 0.9) / Math.max(quantile(activeMagnitudes, 0.5), 1e-8)
    : 0;
  const longestIdleRun = Math.max(0, ...idleRunLengths);
  const classification = activeCount === 0 ? "zero-only" : classifyIntermittency(adi, cv2, zeroShare);
  const label = activeCount === 0
    ? "zero-heavy"
    : classification === "dense" || classification === "smooth"
      ? "dense"
      : classification === "erratic"
        ? "bursty"
        : "sparse";
  const recommendation = classification === "intermittent" || classification === "lumpy" || zeroShare >= 0.2
    ? "recommended"
    : classification === "erratic"
      ? "review"
      : "optional";
  const guidance = buildIntermittencyGuidance(classification, zeroShare, adi, cv2, burstinessRatio);

  return {
    available: true,
    label,
    classification,
    recommendation,
    zeroShare: Number(zeroShare.toFixed(4)),
    activeShare: Number(activeShare.toFixed(4)),
    adi: Number(adi.toFixed(3)),
    cv2: Number(cv2.toFixed(3)),
    burstinessRatio: Number(burstinessRatio.toFixed(3)),
    longestIdleRun,
    activeCount,
    zeroLikeThreshold: Number(zeroLikeThreshold.toFixed(6)),
    idleRunHistogram: buildRunLengthHistogram(idleRunLengths),
    activityTimeline: series.map((point, index) => ({
      timestamp: point.timestamp ?? `T${index + 1}`,
      label: formatPointLabel(point.timestamp, index),
      value: Number(point.value.toFixed(4)),
      activeValue: activeMask[index] ? Number(point.value.toFixed(4)) : null,
    })),
    eventIntervals: eventIntervals.slice(0, 48),
    baselineHints: guidance.baselineHints,
    preprocessingSuggestions: guidance.preprocessingSuggestions,
  };
}

function computeVolatilityDiagnostics(series) {
  const values = series.map((point) => point.value);
  if (values.length < 24) {
    return {
      available: false,
      label: "insufficient",
      recommendation: "optional",
      archStatistic: 0,
      archPValue: 1,
      archLag: 0,
      clusteringScore: 0,
      regimeRatio: 0,
      medianVolatility: 0,
      maxVolatility: 0,
      rollingVolatility: [],
      shockTimeline: [],
      peakTimestamp: "",
      varianceTransformGuidance: {
        preferredTransform: "none",
        skewness: 0,
        minValue: 0,
        zeroShare: 0,
        suggestions: [],
      },
    };
  }

  const innovations = difference(values);
  const innovationMean = average(innovations);
  const centeredInnovations = innovations.map((value) => value - innovationMean);
  const innovationStd = standardDeviation(centeredInnovations, 0);
  const standardized = centeredInnovations.map((value) => value / Math.max(innovationStd, 1e-8));
  const squared = standardized.map((value) => value ** 2);
  const absInnovations = centeredInnovations.map((value) => Math.abs(value));
  const archLag = Math.min(8, Math.max(2, Math.floor(Math.sqrt(centeredInnovations.length) / 2)));

  let archStatistic = 0;
  let archPValue = 1;
  if (squared.length > archLag + 6) {
    const design = [];
    const target = [];
    for (let index = archLag; index < squared.length; index += 1) {
      const row = [1];
      for (let lag = 1; lag <= archLag; lag += 1) {
        row.push(squared[index - lag]);
      }
      design.push(row);
      target.push(squared[index]);
    }

    const regression = ordinaryLeastSquares(design, target);
    const targetMean = average(target);
    const ssTot = sum(target.map((value) => (value - targetMean) ** 2));
    const ssRes = sum(regression.residuals.map((value) => value ** 2));
    const rSquared = ssTot > 1e-12 ? clamp(1 - ssRes / ssTot, 0, 1) : 0;
    archStatistic = design.length * rSquared;
    archPValue = chiSquareSurvivalApprox(archStatistic, archLag);
  }

  const clusteringScore = average(
    Array.from({ length: Math.min(4, Math.max(1, absInnovations.length - 1)) }, (_value, lagIndex) => {
      const lag = lagIndex + 1;
      return Math.abs(pearsonCorrelation(absInnovations.slice(lag), absInnovations.slice(0, absInnovations.length - lag)));
    }),
  );

  const rollingWindow = clamp(Math.floor(values.length / 8), 8, 32);
  const rollingVolatility = [];
  for (let start = 0; start + rollingWindow <= centeredInnovations.length; start += Math.max(2, Math.floor(rollingWindow / 3))) {
    const windowValues = centeredInnovations.slice(start, start + rollingWindow);
    const volatility = standardDeviation(windowValues, average(windowValues));
    const timestampIndex = Math.min(series.length - 1, start + rollingWindow);
    rollingVolatility.push({
      window: `W${rollingVolatility.length + 1}`,
      timestamp: series[timestampIndex]?.timestamp ?? `T${timestampIndex + 1}`,
      label: formatPointLabel(series[timestampIndex]?.timestamp, timestampIndex),
      volatility: Number(volatility.toFixed(4)),
      meanAbsChange: Number(average(windowValues.map((value) => Math.abs(value))).toFixed(4)),
    });
  }

  const rollingValues = rollingVolatility.map((window) => window.volatility);
  const medianVolatility = rollingValues.length > 0 ? median(rollingValues) : 0;
  const regimeRatio = rollingValues.length > 1
    ? quantile(rollingValues, 0.9) / Math.max(quantile(rollingValues, 0.1), 1e-8)
    : 1;
  const peakWindow = rollingVolatility.reduce((best, window) => (
    !best || window.volatility > best.volatility ? window : best
  ), null);
  const label = archPValue < 0.05 && regimeRatio >= 1.8
    ? "clustered"
    : archPValue < 0.1 || regimeRatio >= 1.45 || clusteringScore >= 0.18
      ? "moderate"
      : "stable";
  const recommendation = label === "stable" ? "optional" : "recommended";
  const varianceTransformGuidance = buildVarianceTransformGuidance(values, label, archPValue, regimeRatio);

  return {
    available: true,
    label,
    recommendation,
    archStatistic: Number(archStatistic.toFixed(3)),
    archPValue: Number(archPValue.toFixed(4)),
    archLag,
    clusteringScore: Number(clusteringScore.toFixed(3)),
    regimeRatio: Number(regimeRatio.toFixed(3)),
    medianVolatility: Number(medianVolatility.toFixed(4)),
    maxVolatility: Number(Math.max(...rollingValues, 0).toFixed(4)),
    rollingVolatility,
    shockTimeline: centeredInnovations.map((value, index) => ({
      timestamp: series[index + 1]?.timestamp ?? `T${index + 2}`,
      label: formatPointLabel(series[index + 1]?.timestamp, index + 1),
      absChange: Number(Math.abs(value).toFixed(4)),
      signedChange: Number(value.toFixed(4)),
    })),
    peakTimestamp: peakWindow?.timestamp ?? "",
    varianceTransformGuidance,
  };
}

function classifyStationarity(adf, kpss, slope) {
  if (adf.rejectUnitRoot && !kpss.rejectStationarity) {
    return { label: "stationary", automationSummary: "Series looks stationary. Keep differencing off and focus on lag window and seasonality structure.", needsDiff: false, needsDetrend: false };
  }
  if (!adf.rejectUnitRoot && kpss.rejectStationarity) {
    return { label: "difference-stationary", automationSummary: "Tests point to a unit root. Enable differencing and use a conservative horizon until the level becomes stable.", needsDiff: true, needsDetrend: Math.abs(slope) > 0.035 };
  }
  if (adf.rejectUnitRoot && kpss.rejectStationarity) {
    return { label: "trend-stationary", automationSummary: "The series is mean-reverting after removing deterministic drift. Detrending is recommended before fitting the backbone.", needsDiff: false, needsDetrend: true };
  }
  return { label: "inconclusive", automationSummary: "Stationarity tests disagree weakly. Prefer robust scaling, monitor drift, and keep preprocessing recommendations under review.", needsDiff: Math.abs(slope) > 0.075, needsDetrend: Math.abs(slope) > 0.035 };
}

function inferSeasonalPeriods(acfPeaks, count) {
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

function buildMstlStyleDecomposition(series, seasonalPeriods) {
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
    trend = movingAverage(deseasonalized, Math.max(5, Math.min(observed.length - 1, periods[0] * 2 + 1)));
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

function isMonotonic(values) {
  const diffs = difference(values);
  return diffs.every((value) => value >= -1e-10) || diffs.every((value) => value <= 1e-10);
}

function countZeroCrossings(values) {
  let count = 0;
  for (let index = 1; index < values.length; index += 1) {
    const previousSign = values[index - 1] >= 0 ? 1 : -1;
    const currentSign = values[index] >= 0 ? 1 : -1;
    if (previousSign !== currentSign) {
      count += 1;
    }
  }
  return count;
}

function findExtrema(values, kind = "max") {
  if (values.length < 3) {
    return { indices: [], values: [] };
  }

  const indices = [];
  if ((kind === "max" && values[0] > values[1]) || (kind === "min" && values[0] < values[1])) {
    indices.push(0);
  }

  for (let index = 1; index < values.length - 1; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    const next = values[index + 1];
    const isExtremum = kind === "max"
      ? ((current >= previous && current > next) || (current > previous && current >= next))
      : ((current <= previous && current < next) || (current < previous && current <= next));

    if (isExtremum) {
      indices.push(index);
    }
  }

  const lastIndex = values.length - 1;
  if ((kind === "max" && values[lastIndex] > values[lastIndex - 1]) || (kind === "min" && values[lastIndex] < values[lastIndex - 1])) {
    indices.push(lastIndex);
  }

  const uniqueIndices = [...new Set(indices)].sort((left, right) => left - right);
  return {
    indices: uniqueIndices,
    values: uniqueIndices.map((index) => values[index]),
  };
}

function interpolateEnvelope(indices, values, count) {
  if (indices.length < 2 || values.length < 2) {
    return null;
  }

  const envelope = Array.from({ length: count }, () => 0);
  const firstIndex = indices[0];
  for (let index = 0; index <= firstIndex; index += 1) {
    envelope[index] = values[0];
  }

  for (let segment = 0; segment < indices.length - 1; segment += 1) {
    const leftIndex = indices[segment];
    const rightIndex = indices[segment + 1];
    const leftValue = values[segment];
    const rightValue = values[segment + 1];
    const width = Math.max(1, rightIndex - leftIndex);

    for (let index = leftIndex; index <= rightIndex; index += 1) {
      const ratio = (index - leftIndex) / width;
      envelope[index] = leftValue * (1 - ratio) + rightValue * ratio;
    }
  }

  const lastIndex = indices[indices.length - 1];
  for (let index = lastIndex; index < count; index += 1) {
    envelope[index] = values[values.length - 1];
  }

  return envelope;
}

function countExtrema(values) {
  return findExtrema(values, "max").indices.length + findExtrema(values, "min").indices.length;
}

function siftEmpiricalMode(signal, maxSifts = 60, threshold = 0.03) {
  let mode = [...signal];
  let completedSifts = 0;

  for (let iteration = 0; iteration < maxSifts; iteration += 1) {
    const maxima = findExtrema(mode, "max");
    const minima = findExtrema(mode, "min");
    if (maxima.indices.length < 3 || minima.indices.length < 3) {
      return completedSifts > 0 ? { values: mode, sifts: completedSifts } : null;
    }

    const upper = interpolateEnvelope(maxima.indices, maxima.values, mode.length);
    const lower = interpolateEnvelope(minima.indices, minima.values, mode.length);
    if (!upper || !lower) {
      return completedSifts > 0 ? { values: mode, sifts: completedSifts } : null;
    }

    const meanEnvelope = upper.map((value, index) => 0.5 * (value + lower[index]));
    const updated = mode.map((value, index) => value - meanEnvelope[index]);
    const delta = sum(updated.map((value, index) => (value - mode[index]) ** 2));
    const norm = sum(mode.map((value) => value ** 2)) + 1e-12;
    const relativeChange = delta / norm;

    mode = updated;
    completedSifts = iteration + 1;

    const zeroCrossings = countZeroCrossings(mode);
    const extremaCount = countExtrema(mode);
    const meanEnvelopeStd = standardDeviation(meanEnvelope, average(meanEnvelope));
    const modeStd = standardDeviation(mode, average(mode));
    if (
      relativeChange < threshold
      || Math.abs(zeroCrossings - extremaCount) <= 1
      || meanEnvelopeStd < threshold * Math.max(modeStd, 1e-12)
    ) {
      break;
    }
  }

  return { values: mode, sifts: completedSifts };
}

function computeEmdDiagnostics(series, maxImfs = 6, maxSifts = 60, siftThreshold = 0.03) {
  const values = series.map((point) => point.value);
  const count = values.length;
  if (count < 16) {
    return {
      available: false,
      method: "emd_linear_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Linear envelopes",
      reason: "EMD needs at least 16 observations.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const mean = average(values);
  const std = standardDeviation(values, mean);
  if (std < 1e-10) {
    return {
      available: false,
      method: "emd_linear_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Linear envelopes",
      reason: "EMD needs a non-constant signal.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const originalEnergy = sum(values.map((value) => value ** 2)) + 1e-12;
  const originalStd = std;
  const components = [];
  let residual = [...values];
  let totalSifts = 0;

  for (let index = 0; index < maxImfs; index += 1) {
    const residualEnergyShare = sum(residual.map((value) => value ** 2)) / originalEnergy;
    if (residualEnergyShare < 1e-6 || isMonotonic(residual)) {
      break;
    }

    const sifted = siftEmpiricalMode(residual, maxSifts, siftThreshold);
    if (!sifted) {
      break;
    }

    const componentValues = sifted.values;
    const energy = sum(componentValues.map((value) => value ** 2));
    const extremaCount = countExtrema(componentValues);
    const zeroCrossings = countZeroCrossings(componentValues);
    const meanAbs = average(componentValues.map((value) => Math.abs(value)));

    components.push({
      order: components.length + 1,
      name: `IMF ${components.length + 1}`,
      values: componentValues.map((value) => Number(value.toFixed(6))),
      energyShare: Number((energy / originalEnergy).toFixed(4)),
      zeroCrossings,
      extremaCount,
      meanAbs: Number(meanAbs.toFixed(4)),
      sifts: sifted.sifts,
    });

    residual = residual.map((value, componentIndex) => value - componentValues[componentIndex]);
    totalSifts += sifted.sifts;

    if (standardDeviation(componentValues, average(componentValues)) < 1e-10 * originalStd) {
      break;
    }
  }

  if (components.length === 0) {
    return {
      available: false,
      method: "emd_linear_sifting",
      methodLabel: "Empirical mode decomposition",
      interpolationLabel: "Linear envelopes",
      reason: "The signal became monotonic before an IMF could be extracted.",
      components: [],
      residual: null,
      imfCount: 0,
      totalSifts: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const reconstruction = values.map((_value, index) => (
    residual[index] + components.reduce((total, component) => total + component.values[index], 0)
  ));
  const reconstructionError = Math.sqrt(average(reconstruction.map((value, index) => (value - values[index]) ** 2)));
  const residualEnergy = sum(residual.map((value) => value ** 2));

  return {
    available: true,
    method: "emd_linear_sifting",
    methodLabel: "Empirical mode decomposition",
    interpolationLabel: "Linear envelopes",
    reason: "",
    components,
    residual: {
      name: "Residual",
      values: residual.map((value) => Number(value.toFixed(6))),
      energyShare: Number((residualEnergy / originalEnergy).toFixed(4)),
      zeroCrossings: countZeroCrossings(residual),
      extremaCount: countExtrema(residual),
      meanAbs: Number(average(residual.map((value) => Math.abs(value))).toFixed(4)),
    },
    imfCount: components.length,
    totalSifts,
    reconstructionError: Number(reconstructionError.toFixed(6)),
    residualEnergyShare: Number((residualEnergy / originalEnergy).toFixed(4)),
    siftThreshold: Number(siftThreshold.toFixed(3)),
  };
}

function smape(actual, forecast) {
  const values = actual.map((value, index) => {
    const denominator = (Math.abs(value) + Math.abs(forecast[index])) / 2;
    return denominator === 0 ? 0 : Math.abs(value - forecast[index]) / denominator;
  });
  return average(values) * 100;
}

function mae(actual, forecast) {
  return average(actual.map((value, index) => Math.abs(value - forecast[index])));
}

function rmse(actual, forecast) {
  return Math.sqrt(average(actual.map((value, index) => (value - forecast[index]) ** 2)));
}

function absoluteErrors(actual, forecast) {
  return actual.map((value, index) => Math.abs(value - forecast[index]));
}

function forecastNaive(train, horizon) {
  return Array.from({ length: horizon }, () => train[train.length - 1]);
}

function forecastSeasonalNaive(train, horizon, period) {
  if (period < 2 || train.length <= period) {
    return null;
  }
  return Array.from({ length: horizon }, (_value, index) => train[train.length - period + (index % period)]);
}

function forecastMovingAverage(train, horizon, windowSize) {
  const window = train.slice(-Math.max(2, Math.min(windowSize, train.length)));
  const mean = average(window);
  return Array.from({ length: horizon }, () => mean);
}

function forecastMean(train, horizon) {
  if (train.length === 0) {
    return null;
  }
  const meanValue = average(train);
  return Array.from({ length: horizon }, () => meanValue);
}

function forecastMedian(train, horizon) {
  if (train.length === 0) {
    return null;
  }
  const sorted = [...train].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
  return Array.from({ length: horizon }, () => median);
}

function forecastSimpleExponentialSmoothing(train, horizon, alpha = 0.2) {
  if (train.length === 0) {
    return null;
  }
  let level = train[0];
  for (let index = 1; index < train.length; index += 1) {
    level = alpha * train[index] + (1 - alpha) * level;
  }
  return Array.from({ length: horizon }, () => level);
}

function forecastDrift(train, horizon) {
  if (train.length < 2) {
    return forecastNaive(train, horizon);
  }
  const slope = (train[train.length - 1] - train[0]) / (train.length - 1);
  return Array.from({ length: horizon }, (_value, index) => train[train.length - 1] + slope * (index + 1));
}

function forecastLinearTrend(train, horizon) {
  if (train.length < 3) {
    return forecastDrift(train, horizon);
  }
  const design = train.map((_value, index) => [1, index]);
  const regression = ordinaryLeastSquares(design, train);
  return Array.from({ length: horizon }, (_value, index) => {
    const t = train.length + index;
    return regression.coefficients[0] + regression.coefficients[1] * t;
  });
}

function rollingOriginBacktest(values, period, horizon) {
  const minTrain = Math.max(48, period * 3, horizon * 4);
  const maxSplits = 5;
  const splitOrigins = [];
  for (let trainEnd = values.length - horizon * maxSplits; trainEnd <= values.length - horizon; trainEnd += horizon) {
    if (trainEnd >= minTrain) {
      splitOrigins.push(trainEnd);
    }
  }
  const origins = splitOrigins.slice(-maxSplits);
  if (origins.length === 0) {
    return { available: false, horizon, splits: 0, models: [], winner: null };
  }

  const modelDefs = [
    { name: "naive", label: "Naive", forecast: (train) => forecastNaive(train, horizon) },
    { name: "seasonal_naive", label: "Seasonal Naive", forecast: (train) => forecastSeasonalNaive(train, horizon, period) },
    { name: "moving_average", label: "Moving Average", forecast: (train) => forecastMovingAverage(train, horizon, period) },
    { name: "mean", label: "Mean", forecast: (train) => forecastMean(train, horizon) },
    { name: "median", label: "Median", forecast: (train) => forecastMedian(train, horizon) },
    { name: "exponential_smoothing", label: "Exp. Smoothing", forecast: (train) => forecastSimpleExponentialSmoothing(train, horizon) },
    { name: "drift", label: "Drift", forecast: (train) => forecastDrift(train, horizon) },
    { name: "linear_trend", label: "Linear Trend", forecast: (train) => forecastLinearTrend(train, horizon) },
  ];

  const metrics = modelDefs.map((model) => ({ name: model.name, label: model.label, mae: [], rmse: [], smape: [], foldSeries: [], horizonErrors: Array.from({ length: horizon }, () => []), residuals: [] }));
  const folds = [];

  origins.forEach((trainEnd, splitIndex) => {
    const train = values.slice(0, trainEnd);
    const actual = values.slice(trainEnd, trainEnd + horizon);
    const fold = {
      split: `F${splitIndex + 1}`,
      splitIndex: splitIndex + 1,
      trainEnd,
      models: [],
    };

    modelDefs.forEach((model, index) => {
      const forecast = model.forecast(train);
      if (!forecast || forecast.length !== actual.length) {
        return;
      }
      const foldMae = mae(actual, forecast);
      const foldRmse = rmse(actual, forecast);
      const foldSmape = smape(actual, forecast);
      const errors = absoluteErrors(actual, forecast);

      metrics[index].mae.push(foldMae);
      metrics[index].rmse.push(foldRmse);
      metrics[index].smape.push(foldSmape);
      metrics[index].foldSeries.push({
        split: fold.split,
        splitIndex: fold.splitIndex,
        mae: foldMae,
        rmse: foldRmse,
        smape: foldSmape,
      });
      errors.forEach((errorValue, horizonIndex) => {
        metrics[index].horizonErrors[horizonIndex].push(errorValue);
      });
      actual.forEach((actualValue, horizonIndex) => {
        metrics[index].residuals.push(actualValue - forecast[horizonIndex]);
      });

      fold.models.push({
        name: model.name,
        label: model.label,
        mae: Number(foldMae.toFixed(4)),
        rmse: Number(foldRmse.toFixed(4)),
        smape: Number(foldSmape.toFixed(2)),
      });
    });

    folds.push(fold);
  });

  const summarized = metrics.filter((model) => model.mae.length > 0).map((model) => ({
    name: model.name,
    label: model.label,
    mae: Number(average(model.mae).toFixed(4)),
    rmse: Number(average(model.rmse).toFixed(4)),
    smape: Number(average(model.smape).toFixed(2)),
    foldSeries: model.foldSeries.map((fold) => ({
      split: fold.split,
      splitIndex: fold.splitIndex,
      mae: Number(fold.mae.toFixed(4)),
      rmse: Number(fold.rmse.toFixed(4)),
      smape: Number(fold.smape.toFixed(2)),
    })),
    horizonProfile: model.horizonErrors.map((errors, horizonIndex) => ({
      horizon: horizonIndex + 1,
      mae: Number((errors.length > 0 ? average(errors) : 0).toFixed(4)),
    })),
    residuals: model.residuals.map((value) => Number(value.toFixed(4))),
  })).sort((left, right) => left.mae - right.mae);

  const foldCurves = folds.map((fold) => {
    const row = {
      split: fold.split,
      splitIndex: fold.splitIndex,
      trainEnd: fold.trainEnd,
    };
    fold.models.forEach((model) => {
      row[`${model.name}_mae`] = model.mae;
      row[`${model.name}_rmse`] = model.rmse;
      row[`${model.name}_smape`] = model.smape;
    });
    return row;
  });

  const horizonProfiles = Array.from({ length: horizon }, (_value, horizonIndex) => {
    const row = { horizon: horizonIndex + 1 };
    summarized.forEach((model) => {
      row[`${model.name}_mae`] = model.horizonProfile[horizonIndex]?.mae ?? 0;
    });
    return row;
  });

  return {
    available: summarized.length > 0,
    horizon,
    splits: origins.length,
    models: summarized,
    foldCurves,
    horizonProfiles,
    winner: summarized[0] ?? null,
  };
}

function pearsonCorrelation(left, right) {
  if (left.length < 3 || right.length < 3 || left.length !== right.length) {
    return 0;
  }
  const meanLeft = average(left);
  const meanRight = average(right);
  let numerator = 0;
  let leftDenominator = 0;
  let rightDenominator = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftCentered = left[index] - meanLeft;
    const rightCentered = right[index] - meanRight;
    numerator += leftCentered * rightCentered;
    leftDenominator += leftCentered ** 2;
    rightDenominator += rightCentered ** 2;
  }
  const denominator = Math.sqrt(leftDenominator * rightDenominator);
  return denominator === 0 ? 0 : numerator / denominator;
}

function erfApprox(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function normalCdf(value) {
  return 0.5 * (1 + erfApprox(value / Math.sqrt(2)));
}

function chiSquareSurvivalApprox(statistic, degreesOfFreedom) {
  if (degreesOfFreedom <= 0) {
    return 1;
  }
  const normalized = Math.max(statistic, 1e-12) / degreesOfFreedom;
  const z = (Math.cbrt(normalized) - (1 - 2 / (9 * degreesOfFreedom))) / Math.sqrt(2 / (9 * degreesOfFreedom));
  return clamp(1 - normalCdf(z), 0, 1);
}

function crossCorrelationAtLag(target, covariate, lag) {
  const alignedTarget = [];
  const alignedCovariate = [];
  for (let index = 0; index < target.length; index += 1) {
    const covariateIndex = index - lag;
    if (covariateIndex < 0 || covariateIndex >= covariate.length) {
      continue;
    }
    const covariateValue = covariate[covariateIndex];
    if (covariateValue == null) {
      continue;
    }
    alignedTarget.push(target[index]);
    alignedCovariate.push(covariateValue);
  }
  if (alignedTarget.length < 12) {
    return { correlation: 0, pairs: alignedTarget.length };
  }
  return { correlation: pearsonCorrelation(alignedTarget, alignedCovariate), pairs: alignedTarget.length };
}

function alignSeriesAtLag(target, covariate, lag) {
  const alignedTarget = [];
  const alignedCovariate = [];
  const indices = [];

  for (let index = 0; index < target.length; index += 1) {
    const covariateIndex = index - lag;
    if (covariateIndex < 0 || covariateIndex >= covariate.length) {
      continue;
    }

    const covariateValue = covariate[covariateIndex];
    if (covariateValue == null) {
      continue;
    }

    alignedTarget.push(target[index]);
    alignedCovariate.push(covariateValue);
    indices.push(index);
  }

  return { alignedTarget, alignedCovariate, indices };
}

function describeLagDirection(lag) {
  if (lag > 0) {
    return "covariate leads target";
  }
  if (lag < 0) {
    return "target leads covariate";
  }
  return "contemporaneous";
}

function computeCointegrationHeuristic(target, covariate, lag) {
  const aligned = alignSeriesAtLag(target, covariate, lag);
  if (aligned.alignedTarget.length < 24) {
    return {
      available: false,
      label: "insufficient",
      recommendation: "optional",
      beta: 0,
      intercept: 0,
      residualScale: 0,
      alignedCount: aligned.alignedTarget.length,
      residualAdfStatistic: 0,
      residualAdfRejectUnitRoot: false,
    };
  }

  try {
    const design = aligned.alignedCovariate.map((value) => [1, value]);
    const regression = ordinaryLeastSquares(design, aligned.alignedTarget);
    const residuals = regression.residuals;
    const residualAdf = runAdfTest(residuals);
    const targetStd = standardDeviation(aligned.alignedTarget, average(aligned.alignedTarget));
    const residualScale = standardDeviation(residuals, average(residuals)) / Math.max(targetStd, 1e-8);
    const label = residualAdf.available && residualAdf.rejectUnitRoot
      ? residualScale <= 0.75 ? "stable" : "possible"
      : residualScale <= 0.45 ? "possible" : "weak";
    const recommendation = label === "stable" ? "strong" : label === "possible" ? "review" : "weak";

    return {
      available: true,
      label,
      recommendation,
      beta: Number(regression.coefficients[1].toFixed(4)),
      intercept: Number(regression.coefficients[0].toFixed(4)),
      residualScale: Number(residualScale.toFixed(3)),
      alignedCount: aligned.alignedTarget.length,
      residualAdfStatistic: Number(residualAdf.statistic.toFixed(3)),
      residualAdfRejectUnitRoot: residualAdf.rejectUnitRoot,
    };
  } catch {
    return {
      available: false,
      label: "degenerate",
      recommendation: "optional",
      beta: 0,
      intercept: 0,
      residualScale: 0,
      alignedCount: aligned.alignedTarget.length,
      residualAdfStatistic: 0,
      residualAdfRejectUnitRoot: false,
    };
  }
}

function computeLeadLagStability(target, covariate, maxLag, preferredLag = 0) {
  const windowSize = clamp(Math.floor(target.length / 3), 24, 96);
  const step = Math.max(6, Math.floor(windowSize / 3));
  const windows = [];

  for (let start = 0; start + windowSize <= target.length; start += step) {
    const targetWindow = target.slice(start, start + windowSize);
    const covariateWindow = covariate.slice(start, start + windowSize);
    const lagZero = crossCorrelationAtLag(targetWindow, covariateWindow, 0);
    let best = { lag: 0, correlation: lagZero.correlation, pairs: lagZero.pairs };

    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
      const result = crossCorrelationAtLag(targetWindow, covariateWindow, lag);
      if (Math.abs(result.correlation) > Math.abs(best.correlation)) {
        best = { lag, correlation: result.correlation, pairs: result.pairs };
      }
    }

    windows.push({
      window: `W${windows.length + 1}`,
      label: `W${windows.length + 1}`,
      bestLag: best.lag,
      bestCorrelation: Number(best.correlation.toFixed(3)),
      score: Number((Math.abs(best.correlation) * (best.pairs / Math.max(windowSize, 1))).toFixed(3)),
      alignedPairs: best.pairs,
    });
  }

  if (windows.length < 2) {
    return {
      available: false,
      label: "insufficient",
      medianLag: preferredLag,
      stableShare: 0,
      lagStd: 0,
      scoreStd: 0,
      direction: describeLagDirection(preferredLag),
      windows,
    };
  }

  const lagValues = windows.map((item) => item.bestLag);
  const scoreValues = windows.map((item) => item.score);
  const medianLag = median(lagValues);
  const stableShare = windows.filter((item) => Math.abs(item.bestLag - medianLag) <= 1).length / windows.length;
  const lagStd = standardDeviation(lagValues, average(lagValues));
  const scoreStd = standardDeviation(scoreValues, average(scoreValues));
  const label = stableShare >= 0.6 && lagStd <= 1.2
    ? "stable"
    : stableShare >= 0.4 && lagStd <= 2.5
      ? "moderate"
      : "shifting";

  return {
    available: true,
    label,
    medianLag: Number(medianLag.toFixed(2)),
    stableShare: Number(stableShare.toFixed(3)),
    lagStd: Number(lagStd.toFixed(3)),
    scoreStd: Number(scoreStd.toFixed(3)),
    direction: describeLagDirection(preferredLag),
    windows,
  };
}

function screenExogenousCovariates(target, covariates, maxLag) {
  return covariates.map((covariate) => {
    const lagZero = crossCorrelationAtLag(target, covariate.values, 0);
    let best = { lag: 0, correlation: lagZero.correlation, pairs: lagZero.pairs };
    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
      const result = crossCorrelationAtLag(target, covariate.values, lag);
      if (Math.abs(result.correlation) > Math.abs(best.correlation)) {
        best = { lag, correlation: result.correlation, pairs: result.pairs };
      }
    }
    const score = Math.abs(best.correlation) * covariate.validRatio;
    const leadLagStability = computeLeadLagStability(target, covariate.values, maxLag, best.lag);
    const cointegration = computeCointegrationHeuristic(target, covariate.values, best.lag);
    let recommendation = "weak";
    if (score >= 0.45 && (leadLagStability.label === "stable" || cointegration.label === "stable")) recommendation = "strong";
    else if (score >= 0.25 || leadLagStability.label === "moderate" || cointegration.label === "possible") recommendation = "medium";
    return {
      name: covariate.name,
      coverage: Number((covariate.validRatio * 100).toFixed(1)),
      lag0Correlation: Number(lagZero.correlation.toFixed(3)),
      bestLag: best.lag,
      bestCorrelation: Number(best.correlation.toFixed(3)),
      score: Number(score.toFixed(3)),
      recommendation,
      missingCount: covariate.missingCount,
      direction: describeLagDirection(best.lag),
      leadLagStability,
      cointegration,
    };
  }).sort((left, right) => right.score - left.score);
}

function alignSeriesPair(target, predictor) {
  const alignedTarget = [];
  const alignedPredictor = [];
  const length = Math.min(target.length, predictor.length);

  for (let index = 0; index < length; index += 1) {
    const targetValue = target[index];
    const predictorValue = predictor[index];
    if (Number.isFinite(targetValue) && Number.isFinite(predictorValue)) {
      alignedTarget.push(targetValue);
      alignedPredictor.push(predictorValue);
    }
  }

  return {
    alignedTarget,
    alignedPredictor,
    alignedCount: alignedTarget.length,
    coverage: length > 0 ? alignedTarget.length / length : 0,
  };
}

function needsDifferencingForGranger(values) {
  if (values.length < 32 || isConstant(values)) {
    return false;
  }

  const mean = average(values);
  const center = (values.length - 1) / 2;
  const denominator = values.reduce((total, _value, index) => total + (index - center) ** 2, 0);
  const slope = denominator > 0
    ? values.reduce((total, value, index) => total + (index - center) * (value - mean), 0) / denominator
    : 0;
  const adf = runAdfTest(values);
  const kpss = runKpssTest(values);

  if (!adf.available || !kpss.available) {
    return false;
  }

  return classifyStationarity(adf, kpss, slope).needsDiff;
}

function buildGrangerLagDesign(target, predictor, lag) {
  const unrestricted = [];
  const restricted = [];
  const response = [];

  for (let index = lag; index < target.length; index += 1) {
    const restrictedRow = [1];
    const unrestrictedRow = [1];
    for (let lagIndex = 1; lagIndex <= lag; lagIndex += 1) {
      restrictedRow.push(target[index - lagIndex]);
      unrestrictedRow.push(target[index - lagIndex]);
    }
    for (let lagIndex = 1; lagIndex <= lag; lagIndex += 1) {
      unrestrictedRow.push(predictor[index - lagIndex]);
    }
    restricted.push(restrictedRow);
    unrestricted.push(unrestrictedRow);
    response.push(target[index]);
  }

  return { restricted, unrestricted, response };
}

function runGrangerTestForLag(target, predictor, lag) {
  if (isConstant(target) || isConstant(predictor)) {
    return null;
  }

  const { restricted, unrestricted, response } = buildGrangerLagDesign(target, predictor, lag);
  const rowCount = response.length;
  const unrestrictedColumns = unrestricted[0]?.length ?? 0;

  if (rowCount <= unrestrictedColumns + 2 || restricted.length === 0) {
    return null;
  }

  try {
    const restrictedFit = ordinaryLeastSquares(restricted, response);
    const unrestrictedFit = ordinaryLeastSquares(unrestricted, response);
    const restrictedSse = sum(restrictedFit.residuals.map((value) => value ** 2));
    const unrestrictedSse = sum(unrestrictedFit.residuals.map((value) => value ** 2));
    const denominatorDof = rowCount - unrestrictedColumns;

    if (unrestrictedSse <= 1e-12 || denominatorDof <= 0) {
      return null;
    }

    const gain = Math.max(0, restrictedSse - unrestrictedSse);
    const fStatistic = (gain / lag) / (unrestrictedSse / denominatorDof);
    const pValue = fDistributionSurvivalProbability(fStatistic, lag, denominatorDof);
    const responseMean = average(response);
    const totalSst = sum(response.map((value) => (value - responseMean) ** 2));
    const unrestrictedR2 = totalSst > 1e-12 ? 1 - unrestrictedSse / totalSst : 0;
    const restrictedR2 = totalSst > 1e-12 ? 1 - restrictedSse / totalSst : 0;

    return {
      lag,
      fStatistic: Number(fStatistic.toFixed(3)),
      pValue: Number(pValue.toFixed(4)),
      deltaR2: Number(Math.max(0, unrestrictedR2 - restrictedR2).toFixed(4)),
      unrestrictedR2: Number(unrestrictedR2.toFixed(4)),
      denominatorDof,
      alignedCount: rowCount,
      sseGain: Number(gain.toFixed(4)),
    };
  } catch {
    return null;
  }
}

function computeSingleGrangerResult(responseValues, predictorValues, name, candidateLagMax) {
  const aligned = alignSeriesPair(responseValues, predictorValues);
  if (aligned.alignedCount < 36 || aligned.coverage < 0.6) {
    return null;
  }

  const responseNeedsDiff = needsDifferencingForGranger(aligned.alignedTarget);
  const predictorNeedsDiff = needsDifferencingForGranger(aligned.alignedPredictor);
  const mode = responseNeedsDiff || predictorNeedsDiff ? "difference" : "level";
  const responseSample = mode === "difference" ? difference(aligned.alignedTarget) : aligned.alignedTarget;
  const predictorSample = mode === "difference" ? difference(aligned.alignedPredictor) : aligned.alignedPredictor;
  const lagMax = clamp(Math.min(candidateLagMax, Math.floor(responseSample.length / 6)), 1, candidateLagMax);

  if (responseSample.length < 28 || predictorSample.length < 28 || lagMax < 1) {
    return null;
  }

  const lagTests = [];
  for (let lag = 1; lag <= lagMax; lag += 1) {
    const test = runGrangerTestForLag(responseSample, predictorSample, lag);
    if (test) {
      lagTests.push(test);
    }
  }

  if (lagTests.length === 0) {
    return null;
  }

  const bestTest = [...lagTests].sort((left, right) => {
    if (left.pValue !== right.pValue) {
      return left.pValue - right.pValue;
    }
    return right.fStatistic - left.fStatistic;
  })[0];
  const significance = bestTest.pValue <= 0.05
    ? "significant"
    : bestTest.pValue <= 0.1
      ? "borderline"
      : "weak";
  const score = clamp((Math.max(0, -Math.log10(Math.max(bestTest.pValue, 1e-6))) * 18 + bestTest.deltaR2 * 120 + aligned.coverage * 22), 0, 100);

  return {
    name,
    mode,
    validCount: aligned.alignedCount,
    coverage: Number((aligned.coverage * 100).toFixed(1)),
    bestLag: bestTest.lag,
    fStatistic: bestTest.fStatistic,
    pValue: bestTest.pValue,
    deltaR2: bestTest.deltaR2,
    score: Number(score.toFixed(1)),
    significance,
    lagTests,
  };
}

function buildGrangerDirectionSummary(results, candidateLagMax, emptyReason) {
  return {
    available: results.length > 0,
    reason: results.length > 0 ? "" : emptyReason,
    results,
    topDriver: results[0] ?? null,
    testedCount: results.length,
    significantCount: results.filter((item) => item.significance === "significant").length,
    testedLagMax: candidateLagMax,
  };
}

function computeGrangerDiagnostics(series, covariates, windowSize = 48, horizon = 12) {
  if (!covariates.length) {
    return {
      available: false,
      reason: "No numeric companion series are available for Granger analysis.",
      results: [],
      topDriver: null,
      testedCount: 0,
      significantCount: 0,
      testedLagMax: 0,
      directions: {
        predictorToTarget: {
          available: false,
          reason: "No numeric companion series are available for Granger analysis.",
          results: [],
          topDriver: null,
          testedCount: 0,
          significantCount: 0,
          testedLagMax: 0,
        },
        targetToPredictor: {
          available: false,
          reason: "No numeric companion series are available for Granger analysis.",
          results: [],
          topDriver: null,
          testedCount: 0,
          significantCount: 0,
          testedLagMax: 0,
        },
      },
    };
  }

  const targetValues = series.map((point) => point.value);
  const candidateLagMax = clamp(
    Math.min(12, Math.floor(targetValues.length / 8), Math.max(2, Math.round(windowSize / 8)), Math.max(2, Math.round(horizon / 2) + 1)),
    1,
    12,
  );

  const predictorToTargetResults = covariates
    .map((covariate) => computeSingleGrangerResult(targetValues, covariate.values, covariate.name, candidateLagMax))
    .filter(Boolean)
    .sort((left, right) => right.score - left.score);

  const targetToPredictorResults = covariates
    .map((covariate) => computeSingleGrangerResult(covariate.values, targetValues, covariate.name, candidateLagMax))
    .filter(Boolean)
    .sort((left, right) => right.score - left.score);

  const predictorToTarget = buildGrangerDirectionSummary(
    predictorToTargetResults,
    candidateLagMax,
    "The companion series do not have enough aligned coverage for a stable Granger test.",
  );
  const targetToPredictor = buildGrangerDirectionSummary(
    targetToPredictorResults,
    candidateLagMax,
    "The target history does not yield stable reverse-direction Granger tests for the companion series.",
  );

  if (!predictorToTarget.available && !targetToPredictor.available) {
    return {
      available: false,
      reason: predictorToTarget.reason,
      results: [],
      topDriver: null,
      testedCount: 0,
      significantCount: 0,
      testedLagMax: candidateLagMax,
      directions: {
        predictorToTarget,
        targetToPredictor,
      },
    };
  }

  return {
    available: true,
    reason: predictorToTarget.reason,
    results: predictorToTarget.results,
    topDriver: predictorToTarget.topDriver,
    testedCount: predictorToTarget.testedCount,
    significantCount: predictorToTarget.significantCount,
    testedLagMax: candidateLagMax,
    directions: {
      predictorToTarget,
      targetToPredictor,
    },
  };
}

function segmentMoments(prefixSum, prefixSquareSum, start, end) {
  const length = end - start;
  if (length <= 0) {
    return { length: 0, mean: 0, variance: 0 };
  }

  const sumSegment = prefixSum[end] - prefixSum[start];
  const squareSumSegment = prefixSquareSum[end] - prefixSquareSum[start];
  const mean = sumSegment / length;
  const variance = Math.max(1e-8, squareSumSegment / length - mean ** 2);
  return { length, mean, variance };
}

function gaussianSegmentCost(prefixSum, prefixSquareSum, start, end) {
  const moments = segmentMoments(prefixSum, prefixSquareSum, start, end);
  if (moments.length <= 1) {
    return 0;
  }
  return moments.length * Math.log(moments.variance);
}

function computeCompactSpectrum(values, maxBins = 24) {
  if (values.length < 8) {
    return Array.from({ length: maxBins }, () => 0);
  }

  const mean = average(values);
  const centered = values.map((value) => value - mean);
  const bins = Array.from({ length: maxBins }, (_value, index) => {
    const frequencyIndex = index + 1;
    if (frequencyIndex > Math.floor(values.length / 2)) {
      return 0;
    }

    let real = 0;
    let imaginary = 0;
    for (let sampleIndex = 0; sampleIndex < values.length; sampleIndex += 1) {
      const angle = (2 * Math.PI * frequencyIndex * sampleIndex) / values.length;
      real += centered[sampleIndex] * Math.cos(angle);
      imaginary -= centered[sampleIndex] * Math.sin(angle);
    }

    return Math.max(0, (real ** 2 + imaginary ** 2) / values.length);
  });

  const totalPower = sum(bins);
  if (totalPower <= 1e-12) {
    return Array.from({ length: maxBins }, () => 1 / maxBins);
  }

  return bins.map((value) => value / totalPower);
}

function kullbackLeiblerDivergence(left, right) {
  const epsilon = 1e-12;
  let divergence = 0;
  for (let index = 0; index < left.length; index += 1) {
    const p = Math.max(epsilon, left[index]);
    const q = Math.max(epsilon, right[index]);
    divergence += p * Math.log(p / q);
  }
  return divergence;
}

function jensenShannonDivergence(left, right) {
  const midpoint = left.map((value, index) => (value + right[index]) / 2);
  return 0.5 * kullbackLeiblerDivergence(left, midpoint) + 0.5 * kullbackLeiblerDivergence(right, midpoint);
}

function coefficientOfVariation(values) {
  if (values.length < 2) {
    return 0;
  }
  const mean = average(values);
  if (Math.abs(mean) < 1e-8) {
    return 0;
  }
  return standardDeviation(values, mean) / Math.abs(mean);
}

function detectChangePointsSegmentation(series) {
  const values = series.map((point) => point.value);
  const count = values.length;
  const minimumSegment = Math.max(12, Math.floor(count / 10));
  if (count < minimumSegment * 2 + 4) {
    return {
      available: false,
      threshold: 0,
      points: [],
      segments: [],
      label: "insufficient",
      maxScore: 0,
      method: "penalized_gaussian_segmentation",
      methodLabel: "Penalized segmentation",
    };
  }

  const prefixSum = [0];
  const prefixSquareSum = [0];
  values.forEach((value) => {
    prefixSum.push(prefixSum[prefixSum.length - 1] + value);
    prefixSquareSum.push(prefixSquareSum[prefixSquareSum.length - 1] + value ** 2);
  });

  const globalStd = Math.sqrt(segmentMoments(prefixSum, prefixSquareSum, 0, count).variance);
  const penalty = Math.max(3, 2.5 * Math.log(count) * (1 + Math.min(2, globalStd / Math.max(1e-6, Math.abs(average(values))))));
  const maxBreaks = Math.max(1, Math.min(5, Math.floor(count / minimumSegment) - 1));
  const costs = Array.from({ length: count + 1 }, () => Infinity);
  const previousIndex = Array.from({ length: count + 1 }, () => -1);
  const breakCounts = Array.from({ length: count + 1 }, () => 0);
  costs[0] = -penalty;

  for (let end = minimumSegment; end <= count; end += 1) {
    for (let start = 0; start <= end - minimumSegment; start += 1) {
      if (!Number.isFinite(costs[start])) {
        continue;
      }
      if (start > 0 && end - start < minimumSegment) {
        continue;
      }
      const candidateBreakCount = breakCounts[start] + (start > 0 ? 1 : 0);
      if (candidateBreakCount > maxBreaks) {
        continue;
      }

      const candidateCost = costs[start] + gaussianSegmentCost(prefixSum, prefixSquareSum, start, end) + penalty;
      if (candidateCost < costs[end] - 1e-9 || (Math.abs(candidateCost - costs[end]) <= 1e-9 && candidateBreakCount < breakCounts[end])) {
        costs[end] = candidateCost;
        previousIndex[end] = start;
        breakCounts[end] = candidateBreakCount;
      }
    }
  }

  const boundaries = [count];
  let cursor = count;
  while (cursor > 0 && previousIndex[cursor] >= 0) {
    boundaries.push(previousIndex[cursor]);
    cursor = previousIndex[cursor];
  }

  const sortedBoundaries = [...new Set(boundaries)].sort((left, right) => left - right);
  const segments = [];
  for (let index = 1; index < sortedBoundaries.length; index += 1) {
    const start = sortedBoundaries[index - 1];
    const end = sortedBoundaries[index];
    if (end <= start) {
      continue;
    }
    const moments = segmentMoments(prefixSum, prefixSquareSum, start, end);
    segments.push({
      start,
      end,
      mean: Number(moments.mean.toFixed(3)),
      variance: Number(moments.variance.toFixed(3)),
      timestampStart: series[start]?.timestamp ?? `T${start + 1}`,
      timestampEnd: series[Math.max(start, end - 1)]?.timestamp ?? `T${end}`,
    });
  }

  const selected = [];
  for (let index = 1; index < segments.length; index += 1) {
    const leftSegment = segments[index - 1];
    const rightSegment = segments[index];
    const pooledStd = Math.sqrt(Math.max(1e-8, (leftSegment.variance + rightSegment.variance) / 2));
    const magnitude = rightSegment.mean - leftSegment.mean;
    const varianceShift = Math.log(Math.max(rightSegment.variance, 1e-8) / Math.max(leftSegment.variance, 1e-8));
    const score = Math.abs(magnitude) / pooledStd + 0.35 * Math.abs(varianceShift);
    selected.push({
      index: rightSegment.start,
      timestamp: series[rightSegment.start]?.timestamp ?? `T${rightSegment.start + 1}`,
      value: Number((series[rightSegment.start]?.value ?? 0).toFixed(3)),
      score: Number(score.toFixed(3)),
      magnitude: Number(magnitude.toFixed(3)),
      varianceShift: Number(varianceShift.toFixed(3)),
      direction: magnitude >= 0 ? "upshift" : "downshift",
    });
  }

  const threshold = selected.length > 0 ? Math.max(1.1, median(selected.map((point) => point.score))) : penalty;
  return {
    available: true,
    threshold: Number(threshold.toFixed(3)),
    points: selected.slice(0, 5),
    segments,
    label: selected.length > 0 ? "structural breaks detected" : "stable regime",
    maxScore: Number(Math.max(...selected.map((point) => point.score), 0).toFixed(3)),
    method: "penalized_gaussian_segmentation",
    methodLabel: "Penalized segmentation",
  };
}

function detectChangePointsMeanShift(series) {
  const values = series.map((point) => point.value);
  const count = values.length;
  const minimumSegment = Math.max(12, Math.floor(count / 10));
  if (count < minimumSegment * 2 + 4) {
    return {
      available: false,
      threshold: 0,
      points: [],
      segments: [],
      label: "insufficient",
      maxScore: 0,
      method: "windowed_mean_shift_scan",
      methodLabel: "Windowed mean-shift",
    };
  }

  const prefixSum = [0];
  const prefixSquareSum = [0];
  values.forEach((value) => {
    prefixSum.push(prefixSum[prefixSum.length - 1] + value);
    prefixSquareSum.push(prefixSquareSum[prefixSquareSum.length - 1] + value ** 2);
  });

  const candidates = [];
  for (let split = minimumSegment; split <= count - minimumSegment; split += 1) {
    const left = segmentMoments(prefixSum, prefixSquareSum, 0, split);
    const right = segmentMoments(prefixSum, prefixSquareSum, split, count);
    const pooledStd = Math.sqrt(Math.max(1e-8, (left.variance + right.variance) / 2));
    const score = Math.abs(left.mean - right.mean) / pooledStd;
    candidates.push({
      split,
      score,
      magnitude: right.mean - left.mean,
    });
  }

  const scoreMean = average(candidates.map((candidate) => candidate.score));
  const scoreStd = standardDeviation(candidates.map((candidate) => candidate.score), scoreMean);
  const threshold = Math.max(1.15, scoreMean + scoreStd * 0.65);
  const localMaxima = candidates.filter((candidate, index) => {
    const previous = candidates[index - 1]?.score ?? -Infinity;
    const next = candidates[index + 1]?.score ?? -Infinity;
    return candidate.score >= threshold && candidate.score >= previous && candidate.score >= next;
  });

  const selected = [];
  localMaxima.sort((left, right) => right.score - left.score).forEach((candidate) => {
    if (selected.some((point) => Math.abs(point.index - candidate.split) < minimumSegment)) {
      return;
    }
    selected.push({
      index: candidate.split,
      timestamp: series[candidate.split]?.timestamp ?? `T${candidate.split + 1}`,
      value: Number((series[candidate.split]?.value ?? 0).toFixed(3)),
      score: Number(candidate.score.toFixed(3)),
      magnitude: Number(candidate.magnitude.toFixed(3)),
      varianceShift: 0,
      direction: candidate.magnitude >= 0 ? "upshift" : "downshift",
    });
  });

  selected.sort((left, right) => left.index - right.index);
  const boundaries = [0, ...selected.map((point) => point.index), count];
  const segments = [];
  for (let index = 1; index < boundaries.length; index += 1) {
    const start = boundaries[index - 1];
    const end = boundaries[index];
    if (end <= start) {
      continue;
    }
    const moments = segmentMoments(prefixSum, prefixSquareSum, start, end);
    segments.push({
      start,
      end,
      mean: Number(moments.mean.toFixed(3)),
      variance: Number(moments.variance.toFixed(3)),
      timestampStart: series[start]?.timestamp ?? `T${start + 1}`,
      timestampEnd: series[Math.max(start, end - 1)]?.timestamp ?? `T${end}`,
    });
  }

  return {
    available: true,
    threshold: Number(threshold.toFixed(3)),
    points: selected.slice(0, 5),
    segments,
    label: selected.length > 0 ? "structural breaks detected" : "stable regime",
    maxScore: Number(Math.max(...candidates.map((candidate) => candidate.score), 0).toFixed(3)),
    method: "windowed_mean_shift_scan",
    methodLabel: "Windowed mean-shift",
  };
}

function detectChangePoints(series, method = "segmentation") {
  if (method === "mean_shift") {
    return detectChangePointsMeanShift(series);
  }
  return detectChangePointsSegmentation(series);
}

function runLjungBoxTest(residuals, requestedLag) {
  const count = residuals.length;
  if (count < 18) {
    return {
      available: false,
      label: "insufficient",
      statistic: 0,
      pValue: 1,
      lagCount: 0,
      rejectWhiteNoise: false,
      residualAcf: [],
      maxResidualAcf: 0,
    };
  }

  const mean = average(residuals);
  const residualVariance = variance(residuals, mean);
  if (residualVariance < 1e-12) {
    return {
      available: true,
      label: "white",
      statistic: 0,
      pValue: 1,
      lagCount: 0,
      rejectWhiteNoise: false,
      residualAcf: [],
      maxResidualAcf: 0,
    };
  }

  const lagCount = Math.min(requestedLag, Math.max(3, Math.floor(Math.sqrt(count))));
  const residualAcf = [];
  let statistic = 0;
  for (let lag = 1; lag <= lagCount; lag += 1) {
    const rho = autocorrelation(residuals, lag, mean, residualVariance);
    residualAcf.push({ lag, value: Number(rho.toFixed(3)) });
    statistic += (rho ** 2) / Math.max(1, count - lag);
  }
  statistic *= count * (count + 2);
  const pValue = chiSquareSurvivalApprox(statistic, lagCount);
  const rejectWhiteNoise = pValue < 0.05;

  return {
    available: true,
    label: rejectWhiteNoise ? "autocorrelated" : "white-ish",
    statistic: Number(statistic.toFixed(3)),
    pValue: Number(pValue.toFixed(4)),
    lagCount,
    rejectWhiteNoise,
    residualAcf,
    maxResidualAcf: Number(Math.max(...residualAcf.map((point) => Math.abs(point.value)), 0).toFixed(3)),
  };
}

function evaluateRollingStability(series, covariates, seasonalPeriods, horizon) {
  const values = series.map((point) => point.value);
  const count = values.length;
  const dominantPeriod = seasonalPeriods[0] ?? 12;
  const windowLength = clamp(Math.max(36, dominantPeriod * 3, horizon * 4), 36, Math.max(36, count - 6));
  if (count < windowLength + 12) {
    return {
      available: false,
      windows: [],
      seasonalityLabel: "insufficient",
      exogenousLabel: covariates.length > 0 ? "insufficient" : "unavailable",
      topExogenousName: covariates[0]?.name ?? "",
      windowLength,
      dominantPeriod,
      seasonalityRange: 0,
      exogenousShare: 0,
      exogenousLagStd: 0,
      spectralDriftMedian: 0,
      spectralDriftStd: 0,
      dominantPeriodStd: 0,
      dominantPeriodCv: 0,
      regimeConsistencyScore: 0,
      method: "rolling_spectral_regime_stability",
    };
  }

  const step = Math.max(6, Math.floor(windowLength / 2));
  const globalSpectrum = computeCompactSpectrum(values);
  const windows = [];
  for (let start = 0; start + windowLength <= count; start += step) {
    const end = start + windowLength;
    const windowSeries = series.slice(start, end);
    const windowValues = windowSeries.map((point) => point.value);
    const windowMean = average(windowValues);
    const windowVariance = variance(windowValues, windowMean);
    const maxLag = Math.min(24, Math.max(8, Math.floor(windowValues.length / 4)));
    const windowAcf = Array.from({ length: maxLag + 1 }, (_value, index) => ({
      lag: index,
      value: index === 0 ? 1 : autocorrelation(windowValues, index, windowMean, windowVariance),
    }));
    const windowPeaks = detectCorrelationPeaks(windowAcf, 1.96 / Math.sqrt(windowValues.length));
    const windowPeriods = inferSeasonalPeriods(windowPeaks, windowValues.length);
    const windowDecomposition = buildMstlStyleDecomposition(windowSeries, windowPeriods);
    const windowSpectrum = computeCompactSpectrum(windowValues, globalSpectrum.length);
    const spectralDrift = jensenShannonDivergence(windowSpectrum, globalSpectrum);

    const windowCovariates = covariates.map((covariate) => {
      const windowValuesCovariate = covariate.values.slice(start, end);
      const validCount = windowValuesCovariate.filter((value) => value != null).length;
      return {
        ...covariate,
        values: windowValuesCovariate,
        validRatio: windowValuesCovariate.length > 0 ? validCount / windowValuesCovariate.length : 0,
        missingCount: windowValuesCovariate.length - validCount,
      };
    });
    const screened = covariates.length > 0
      ? screenExogenousCovariates(windowValues, windowCovariates, Math.min(12, windowPeriods[0] ?? dominantPeriod))
      : [];
    const topExogenous = screened[0] ?? null;

    windows.push({
      window: `W${windows.length + 1}`,
      start,
      end,
      timestamp: windowSeries[windowSeries.length - 1]?.timestamp ?? `T${end}`,
      seasonalityStrength: Number(windowDecomposition.strength.toFixed(3)),
      dominantPeriod: windowPeriods[0] ?? dominantPeriod,
      spectralDrift: Number(spectralDrift.toFixed(3)),
      exogenousName: topExogenous?.name ?? "none",
      exogenousScore: Number((topExogenous?.score ?? 0).toFixed(3)),
      exogenousLag: topExogenous?.bestLag ?? 0,
    });
  }

  const seasonalityStrengths = windows.map((window) => window.seasonalityStrength);
  const spectralDrifts = windows.map((window) => window.spectralDrift);
  const dominantPeriods = windows.map((window) => window.dominantPeriod);
  const seasonalityRange = Math.max(...seasonalityStrengths) - Math.min(...seasonalityStrengths);
  const seasonalityStd = standardDeviation(seasonalityStrengths, average(seasonalityStrengths));
  const spectralDriftMedian = median(spectralDrifts);
  const spectralDriftStd = spectralDrifts.length > 1 ? standardDeviation(spectralDrifts, average(spectralDrifts)) : 0;
  const dominantPeriodStd = dominantPeriods.length > 1 ? standardDeviation(dominantPeriods, average(dominantPeriods)) : 0;
  const dominantPeriodCv = coefficientOfVariation(dominantPeriods);
  const seasonalityLabel = spectralDriftMedian < 0.06 && dominantPeriodCv < 0.15 && seasonalityRange < 0.2 && seasonalityStd < 0.1
    ? "stable"
    : spectralDriftMedian < 0.12 && dominantPeriodCv < 0.28 && seasonalityRange < 0.35
      ? "moderately stable"
      : "shifting";

  const exogenousWindows = windows.filter((window) => window.exogenousName !== "none");
  let exogenousLabel = covariates.length > 0 ? "shifting" : "unavailable";
  let topExogenousName = "";
  let exogenousShare = 0;
  let exogenousLagStd = 0;
  let exogenousScoreCv = 0;
  if (exogenousWindows.length > 0) {
    const counts = new Map();
    exogenousWindows.forEach((window) => {
      counts.set(window.exogenousName, (counts.get(window.exogenousName) ?? 0) + 1);
    });
    const [dominantName, dominantCount] = [...counts.entries()].sort((left, right) => right[1] - left[1])[0];
    topExogenousName = dominantName;
    exogenousShare = dominantCount / exogenousWindows.length;
    const dominantWindows = exogenousWindows.filter((window) => window.exogenousName === dominantName);
    const lags = dominantWindows.map((window) => window.exogenousLag);
    const scores = dominantWindows.map((window) => window.exogenousScore);
    exogenousLagStd = lags.length > 1 ? standardDeviation(lags, average(lags)) : 0;
    exogenousScoreCv = coefficientOfVariation(scores);
    exogenousLabel = exogenousShare >= 0.7 && exogenousLagStd <= 2 && exogenousScoreCv <= 0.35
      ? "stable"
      : exogenousShare >= 0.5 && exogenousLagStd <= 4
        ? "moderately stable"
        : "shifting";
  }

  const regimeConsistencyScore = clamp(
    Math.round(
      100 * (
        0.45 * (1 - Math.min(1, spectralDriftMedian / 0.18))
        + 0.25 * (1 - Math.min(1, dominantPeriodCv / 0.4))
        + 0.15 * (1 - Math.min(1, seasonalityRange / 0.5))
        + 0.15 * (covariates.length === 0 ? 1 : Math.min(1, exogenousShare))
      )
    ),
    0,
    100,
  );

  return {
    available: true,
    windows,
    seasonalityLabel,
    exogenousLabel,
    topExogenousName,
    windowLength,
    dominantPeriod,
    seasonalityRange: Number(seasonalityRange.toFixed(3)),
    exogenousShare: Number((exogenousShare * 100).toFixed(1)),
    exogenousLagStd: Number(exogenousLagStd.toFixed(3)),
    exogenousScoreCv: Number(exogenousScoreCv.toFixed(3)),
    spectralDriftMedian: Number(spectralDriftMedian.toFixed(3)),
    spectralDriftStd: Number(spectralDriftStd.toFixed(3)),
    dominantPeriodStd: Number(dominantPeriodStd.toFixed(3)),
    dominantPeriodCv: Number(dominantPeriodCv.toFixed(3)),
    regimeConsistencyScore,
    method: "rolling_spectral_regime_stability",
  };
}

function computeSpectralDiagnostics(series) {
  const values = series.map((point) => point.value);
  const count = values.length;
  if (count < 16) {
    return {
      available: false,
      dominantPeriod: 0,
      dominantFrequency: 0,
      normalizedPeakPower: 0,
      totalPower: 0,
      peaks: [],
      spectrum: [],
    };
  }

  const mean = average(values);
  const centered = values.map((value) => value - mean);
  const maxFrequencyIndex = Math.max(2, Math.floor(count / 2));
  const spectrum = [];

  for (let frequencyIndex = 1; frequencyIndex <= maxFrequencyIndex; frequencyIndex += 1) {
    let real = 0;
    let imaginary = 0;
    for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
      const angle = (2 * Math.PI * frequencyIndex * sampleIndex) / count;
      real += centered[sampleIndex] * Math.cos(angle);
      imaginary -= centered[sampleIndex] * Math.sin(angle);
    }

    const power = (real ** 2 + imaginary ** 2) / count;
    const frequency = frequencyIndex / count;
    const period = frequency > 0 ? count / frequencyIndex : 0;

    if (period >= 2 && period <= Math.max(48, Math.floor(count / 2))) {
      spectrum.push({
        bin: frequencyIndex,
        frequency: Number(frequency.toFixed(4)),
        period: Number(period.toFixed(2)),
        power: Number(power.toFixed(4)),
      });
    }
  }

  if (spectrum.length === 0) {
    return {
      available: false,
      dominantPeriod: 0,
      dominantFrequency: 0,
      normalizedPeakPower: 0,
      totalPower: 0,
      peaks: [],
      spectrum: [],
    };
  }

  const totalPower = sum(spectrum.map((point) => point.power));
  const meanPower = average(spectrum.map((point) => point.power));
  const stdPower = standardDeviation(spectrum.map((point) => point.power), meanPower);
  const peakThreshold = meanPower + stdPower * 0.75;
  const peaks = spectrum
    .filter((point, index) => {
      const previous = spectrum[index - 1]?.power ?? -Infinity;
      const next = spectrum[index + 1]?.power ?? -Infinity;
      return point.power >= peakThreshold && point.power >= previous && point.power >= next;
    })
    .sort((left, right) => right.power - left.power)
    .slice(0, 6)
    .map((point) => ({
      ...point,
      normalizedPower: Number((point.power / Math.max(totalPower, 1e-12)).toFixed(4)),
    }));

  const dominant = peaks[0] ?? [...spectrum].sort((left, right) => right.power - left.power)[0];
  return {
    available: true,
    dominantPeriod: Number(dominant.period.toFixed(2)),
    dominantFrequency: dominant.frequency,
    normalizedPeakPower: Number((dominant.power / Math.max(totalPower, 1e-12)).toFixed(4)),
    totalPower: Number(totalPower.toFixed(4)),
    peaks,
    spectrum: spectrum.map((point) => ({
      ...point,
      normalizedPower: Number((point.power / Math.max(totalPower, 1e-12)).toFixed(4)),
    })),
  };
}

function computeForecastabilityDiagnostics(spectralDiagnostics, acfPeaks) {
  if (!spectralDiagnostics?.available || !spectralDiagnostics.spectrum.length) {
    return {
      available: false,
      spectralEntropy: 1,
      forecastabilityScore: 0,
      label: "insufficient",
      complexityLabel: "unknown",
      peakConcentration: 0,
      acfSignal: 0,
    };
  }

  const probabilities = spectralDiagnostics.spectrum
    .map((point) => point.normalizedPower)
    .filter((value) => value > 0);
  const entropy = probabilities.length > 1
    ? -sum(probabilities.map((value) => value * Math.log(value))) / Math.log(probabilities.length)
    : 0;
  const peakConcentration = spectralDiagnostics.peaks
    .slice(0, 3)
    .reduce((total, peak) => total + peak.normalizedPower, 0);
  const acfSignal = Math.max(...(acfPeaks ?? []).map((point) => Math.abs(point.value)), 0);
  const forecastabilityScore = clamp(
    100 * (
      0.6 * (1 - entropy)
      + 0.25 * Math.min(1, peakConcentration * 2)
      + 0.15 * Math.min(1, acfSignal)
    ),
    0,
    100,
  );
  const label = forecastabilityScore >= 70
    ? "high"
    : forecastabilityScore >= 45
      ? "moderate"
      : "low";
  const complexityLabel = entropy <= 0.35
    ? "ordered"
    : entropy <= 0.65
      ? "mixed"
      : "diffuse";

  return {
    available: true,
    spectralEntropy: Number(entropy.toFixed(3)),
    forecastabilityScore: Number(forecastabilityScore.toFixed(1)),
    label,
    complexityLabel,
    peakConcentration: Number(peakConcentration.toFixed(3)),
    acfSignal: Number(acfSignal.toFixed(3)),
  };
}

function chooseNearestCandidate(value, candidates, maxValue = Infinity) {
  const viable = candidates.filter((candidate) => candidate <= maxValue);
  const pool = viable.length > 0 ? viable : candidates;
  return pool.reduce((best, candidate) => (
    Math.abs(candidate - value) < Math.abs(best - value) ? candidate : best
  ), pool[0]);
}

function computePatchingDiagnostics(series, seasonalPeriods, acfPeaks, spectralDiagnostics, forecastabilityDiagnostics) {
  const values = series.map((point) => point.value);
  const count = values.length;
  if (count < 36) {
    return {
      available: false,
      patchingLabel: "insufficient",
      patchingScore: 0,
      recommendedPatchLen: 0,
      recommendedStride: 0,
      multiscaleLabel: "unknown",
      multiscaleScore: 0,
      dominantPeriod: 0,
      repeatability: 0,
      roughness: 0,
      scaleBandData: [],
      recommendedPatchSet: [],
      notes: [],
    };
  }

  const seriesMean = average(values);
  const seriesStd = standardDeviation(values, seriesMean);
  const innovations = difference(values);
  const innovationMean = innovations.length > 0 ? average(innovations) : 0;
  const innovationStd = innovations.length > 1 ? standardDeviation(innovations, innovationMean) : 0;
  const roughness = innovationStd / Math.max(seriesStd, 1e-8);
  const repeatability = Math.max(...(acfPeaks ?? []).map((point) => Math.abs(point.value)), 0);
  const dominantPeriod = spectralDiagnostics?.dominantPeriod > 0
    ? spectralDiagnostics.dominantPeriod
    : seasonalPeriods?.[0] ?? acfPeaks?.[0]?.lag ?? 12;
  const candidatePatchLengths = [4, 6, 8, 12, 16, 24, 32];
  const candidateStrides = [1, 2, 4, 6, 8, 12, 16];
  const maxPatchLength = Math.max(4, Math.min(32, Math.floor(count / 3)));
  let basePatchLength = dominantPeriod >= 24 ? dominantPeriod / 3 : dominantPeriod / 2;
  if (roughness > 1.05) {
    basePatchLength *= 0.75;
  } else if (roughness < 0.55) {
    basePatchLength *= 1.15;
  }
  const recommendedPatchLen = chooseNearestCandidate(basePatchLength, candidatePatchLengths, maxPatchLength);
  const recommendedStride = Math.min(
    recommendedPatchLen,
    chooseNearestCandidate(Math.max(1, recommendedPatchLen / 2), candidateStrides, recommendedPatchLen),
  );

  const spectrum = spectralDiagnostics?.spectrum ?? [];
  const shortBandShare = spectrum
    .filter((point) => point.period <= 8)
    .reduce((total, point) => total + point.normalizedPower, 0);
  const mediumBandShare = spectrum
    .filter((point) => point.period > 8 && point.period <= 24)
    .reduce((total, point) => total + point.normalizedPower, 0);
  const longBandShare = spectrum
    .filter((point) => point.period > 24)
    .reduce((total, point) => total + point.normalizedPower, 0);
  const scaleSpread = [shortBandShare, mediumBandShare, longBandShare].filter((share) => share >= 0.18).length;
  const topPeriods = Array.from(new Set(
    (spectralDiagnostics?.peaks ?? [])
      .slice(0, 4)
      .map((peak) => Math.round(peak.period))
      .filter((period) => period >= 2),
  ));
  if (topPeriods.length === 0) {
    seasonalPeriods.forEach((period) => {
      if (!topPeriods.includes(period)) {
        topPeriods.push(period);
      }
    });
  }
  const peakConcentration = spectralDiagnostics?.peaks
    ?.slice(0, 3)
    .reduce((total, peak) => total + peak.normalizedPower, 0) ?? 0;
  const forecastabilityPenalty = forecastabilityDiagnostics?.label === "low"
    ? 10
    : forecastabilityDiagnostics?.label === "moderate"
      ? 4
      : 0;
  const patchingScore = clamp(
    Math.round(
      100 * (
        0.3 * Math.min(1, peakConcentration / 0.22)
        + 0.25 * Math.min(1, repeatability / 0.45)
        + 0.25 * (1 - Math.min(1, roughness / 1.45))
        + 0.2 * Math.min(1, count / 240)
      ),
    ) - forecastabilityPenalty,
    0,
    100,
  );
  const patchingLabel = patchingScore >= 68
    ? "good"
    : patchingScore >= 48
      ? "moderate"
      : "weak";
  const multiscaleScore = clamp(
    Math.round(
      100 * (
        0.45 * Math.min(1, scaleSpread / 3)
        + 0.35 * Math.min(1, Math.max(0, topPeriods.length - 1) / 3)
        + 0.2 * Math.min(1, shortBandShare > 0.14 && longBandShare > 0.14 ? 1 : mediumBandShare / 0.35)
      ),
    ),
    0,
    100,
  );
  const multiscaleLabel = multiscaleScore >= 65
    ? "high"
    : multiscaleScore >= 45
      ? "moderate"
      : "low";
  const recommendedPatchSet = Array.from(new Set([
    chooseNearestCandidate(Math.max(4, recommendedPatchLen / 2), candidatePatchLengths, maxPatchLength),
    recommendedPatchLen,
    chooseNearestCandidate(Math.max(4, dominantPeriod), candidatePatchLengths, maxPatchLength),
  ])).sort((left, right) => left - right);

  return {
    available: true,
    patchingLabel,
    patchingScore,
    recommendedPatchLen,
    recommendedStride,
    multiscaleLabel,
    multiscaleScore,
    dominantPeriod: Number(dominantPeriod.toFixed(2)),
    repeatability: Number(repeatability.toFixed(3)),
    roughness: Number(roughness.toFixed(3)),
    scaleBandData: [
      { band: "Short", share: Number(shortBandShare.toFixed(4)), detail: "period <= 8" },
      { band: "Medium", share: Number(mediumBandShare.toFixed(4)), detail: "8 < period <= 24" },
      { band: "Long", share: Number(longBandShare.toFixed(4)), detail: "period > 24" },
    ],
    recommendedPatchSet,
    notes: [
      {
        key: "patching-fit",
        label: "Patch tokenization fit",
        suitability: patchingLabel,
        detail: `Patching looks ${patchingLabel} because repeatability is ${repeatability.toFixed(3)} and roughness is ${roughness.toFixed(3)} over ${count} observations.`,
      },
      {
        key: "patch-size",
        label: "Suggested patch geometry",
        suitability: "recommended",
        detail: `Start near length ${recommendedPatchLen} with stride ${recommendedStride}; this anchors each token to the dominant cycle around period ${dominantPeriod.toFixed(1)} without over-compressing local structure.`,
      },
      {
        key: "multiscale-fit",
        label: "Multi-scale utility",
        suitability: multiscaleLabel,
        detail: `Multi-scale usefulness is ${multiscaleLabel} because spectral energy is distributed short/medium/long as ${(shortBandShare * 100).toFixed(1)}% / ${(mediumBandShare * 100).toFixed(1)}% / ${(longBandShare * 100).toFixed(1)}%.`,
      },
    ],
  };
}

function buildAutomationDiagnostics({
  analysis,
  datasetSummary,
  benchmark,
  exogenousScreening,
  grangerDiagnostics,
  changePoints,
  residualDiagnostics,
  stabilityDiagnostics,
  spectralDiagnostics,
  forecastabilityDiagnostics,
  patchingDiagnostics,
  calendarDiagnostics,
  intermittencyDiagnostics,
  volatilityDiagnostics,
}) {
  const rowCount = datasetSummary?.rowCount ?? analysis.count;
  const validObservationCount = datasetSummary?.validObservationCount ?? analysis.count;
  const missingTargetCount = datasetSummary?.missingTargetCount ?? 0;
  const invalidTargetCount = datasetSummary?.invalidTargetCount ?? 0;
  const missingTimestampCount = datasetSummary?.missingTimestampCount ?? 0;
  const droppedRowCount = datasetSummary?.droppedRowCount ?? 0;
  const completeness = rowCount > 0 ? validObservationCount / rowCount : 0;
  const missingRate = rowCount > 0 ? (missingTargetCount + invalidTargetCount) / rowCount : 0;
  const droppedRate = rowCount > 0 ? droppedRowCount / rowCount : 0;
  const spectralAgreementDelta = spectralDiagnostics?.available
    ? Math.abs((spectralDiagnostics.dominantPeriod || 0) - (analysis.dominantAcfLag || 0))
    : 0;
  const dataQualityScore = clamp(
    Math.round(
      100
      - missingRate * 120
      - droppedRate * 80
      - Math.min(18, missingTimestampCount * 1.5)
      - Math.min(12, invalidTargetCount * 1.5),
    ),
    0,
    100,
  );
  const consensusScore = clamp(
    Math.round(
      30
      + (spectralDiagnostics?.available ? Math.max(0, 20 - spectralAgreementDelta * 6) : 8)
      + (stabilityDiagnostics?.seasonalityLabel === "stable" ? 18 : stabilityDiagnostics?.seasonalityLabel === "moderately stable" ? 10 : 2)
      + (!residualDiagnostics?.rejectWhiteNoise ? 16 : 4)
      + ((changePoints?.points?.length ?? 0) === 0 ? 10 : (changePoints?.points?.length ?? 0) <= 1 ? 6 : 0)
      + (benchmark?.winner ? 8 : 0),
    ),
    0,
    100,
  );
  const automationConfidence = clamp(
    Math.round(dataQualityScore * 0.45 + consensusScore * 0.55),
    0,
    100,
  );
  const readinessScore = clamp(
    Math.round(
      automationConfidence
      - (analysis.stationarityLabel === "inconclusive" ? 8 : 0)
      - ((changePoints?.points?.length ?? 0) > 1 ? 8 : 0)
      - (residualDiagnostics?.rejectWhiteNoise ? 6 : 0),
    ),
    0,
    100,
  );
  const complexityPenalty = forecastabilityDiagnostics?.label === "low" ? 12 : forecastabilityDiagnostics?.label === "moderate" ? 4 : 0;
  const adjustedReadiness = clamp(readinessScore - complexityPenalty, 0, 100);

  const recommendedActions = [];
  const risks = [];
  const highlights = [];

  recommendedActions.push({
    key: "apply_blueprint",
    title: "Apply full blueprint",
    detail: `Switch to ${analysis.recommendedFamily} with window ${analysis.suggestedLagWindow} and horizon ${analysis.recommendedHorizon}.`,
    priority: adjustedReadiness >= 70 ? "high" : "medium",
    targetSection: "automation",
  });

  recommendedActions.push({
    key: "auto_prep",
    title: "Auto-configure preprocessing",
    detail: `Use ${analysis.scalingMethod} scaling${analysis.needsDiff ? ", differencing" : ""}${analysis.needsDetrend ? ", detrending" : ""}.`,
    priority: analysis.needsDiff || analysis.needsDetrend || missingRate > 0 ? "high" : "medium",
    targetSection: "stationarity",
  });

  recommendedActions.push({
    key: "apply_window",
    title: `Use suggested window ${analysis.suggestedLagWindow}`,
    detail: `Derived from PACF ${analysis.dominantPacfLag}, ACF ${analysis.dominantAcfLag}, and seasonal structure.`,
    priority: residualDiagnostics?.rejectWhiteNoise ? "high" : "medium",
    window: analysis.suggestedLagWindow,
    targetSection: "explore",
    targetTab: "autocorrelation",
  });

  if (spectralDiagnostics?.available && spectralDiagnostics.dominantPeriod >= 2) {
    recommendedActions.push({
      key: "apply_spectral_window",
      title: `Use spectral window ${clamp(Math.round(spectralDiagnostics.dominantPeriod) * 2, 24, 96)}`,
      detail: `Dominant spectral period ${spectralDiagnostics.dominantPeriod} with ${(spectralDiagnostics.normalizedPeakPower * 100).toFixed(1)}% normalized power.`,
      priority: spectralAgreementDelta <= 2 ? "high" : "medium",
      window: clamp(Math.round(spectralDiagnostics.dominantPeriod) * 2, 24, 96),
      targetSection: "explore",
      targetTab: "spectral",
    });
  }

  if ((changePoints?.points?.length ?? 0) > 0) {
    recommendedActions.push({
      key: "inspect_regimes",
      title: "Inspect regime shifts",
      detail: `${changePoints.points.length} structural break(s) detected. Consider shorter training windows or segmented retraining.`,
      priority: changePoints.points.length > 1 ? "high" : "medium",
      targetSection: "regime",
    });
    risks.push({
      text: `${changePoints.points.length} structural break(s) may make one global model unstable.`,
      targetSection: "regime",
    });
  } else {
    highlights.push({ text: "No major structural breaks detected across the retained series.", targetSection: "regime" });
  }

  if (residualDiagnostics?.rejectWhiteNoise) {
    risks.push({
      text: `Winning baseline residuals remain autocorrelated (p=${residualDiagnostics.pValue}).`,
      targetSection: "backtesting",
    });
  } else if (residualDiagnostics?.available) {
    highlights.push({ text: `Winning baseline residuals are close to white noise (p=${residualDiagnostics.pValue}).`, targetSection: "backtesting" });
  }

  if (missingRate > 0 || missingTimestampCount > 0) {
    risks.push({
      text: `Data completeness is ${(completeness * 100).toFixed(1)}%; imputation and schema checks should stay enabled.`,
      targetSection: "stationarity",
    });
  } else {
    highlights.push({ text: `Data completeness is ${(completeness * 100).toFixed(1)}% with no target/timestamp gaps.`, targetSection: "automation" });
  }

  if (stabilityDiagnostics?.seasonalityLabel === "stable") {
    highlights.push({ text: `Seasonality is stable over rolling windows around period ${stabilityDiagnostics.dominantPeriod}.`, targetSection: "regime" });
  } else if (stabilityDiagnostics?.seasonalityLabel) {
    risks.push({
      text: `Seasonality is ${stabilityDiagnostics.seasonalityLabel}; fixed seasonal assumptions may drift.`,
      targetSection: "explore",
      targetTab: "decomposition",
    });
  }

  if (exogenousScreening?.[0]?.recommendation === "strong") {
    highlights.push({
      text: `Top exogenous candidate ${exogenousScreening[0].name} has strong global lead-lag signal with ${exogenousScreening[0].leadLagStability?.label ?? "unknown"} lag stability.`,
      targetSection: "explore",
      targetTab: "exogenous",
    });
    recommendedActions.push({
      key: "inspect_exogenous",
      title: `Promote ${exogenousScreening[0].name}`,
      detail: `Best lag ${exogenousScreening[0].bestLag} (${exogenousScreening[0].direction}) with score ${exogenousScreening[0].score} and cointegration ${exogenousScreening[0].cointegration?.label ?? "unavailable"}.`,
      priority: stabilityDiagnostics?.exogenousLabel === "stable" ? "high" : "medium",
      targetSection: "explore",
      targetTab: "exogenous",
    });
  }

  if (forecastabilityDiagnostics?.available) {
    if (forecastabilityDiagnostics.label === "low") {
      risks.push({
        text: `Forecastability is low with spectral entropy ${forecastabilityDiagnostics.spectralEntropy}; simpler baselines may outperform complex backbones.`,
        targetSection: "explore",
        targetTab: "spectral",
      });
      recommendedActions.push({
        key: "inspect_spectral",
        title: "Review low forecastability",
        detail: `Entropy ${forecastabilityDiagnostics.spectralEntropy} and forecastability ${forecastabilityDiagnostics.forecastabilityScore}/100 suggest diffuse signal structure.`,
        priority: "high",
        targetSection: "explore",
        targetTab: "spectral",
      });
    } else {
      highlights.push({
        text: `Forecastability is ${forecastabilityDiagnostics.label} with score ${forecastabilityDiagnostics.forecastabilityScore}/100.`,
        targetSection: "explore",
        targetTab: "spectral",
      });
    }
  }

  if (patchingDiagnostics?.available) {
    if (analysis.recommendedFamily === "transformer" && patchingDiagnostics.patchingLabel === "weak") {
      risks.push({
        text: `Transformer patching looks weak for this signal; roughness ${patchingDiagnostics.roughness} may make large token compression brittle.`,
        targetSection: "prep",
        targetTab: "patching",
      });
    } else if (patchingDiagnostics.patchingLabel !== "weak") {
      highlights.push({
        text: `Patch tokenization looks ${patchingDiagnostics.patchingLabel} with recommended length ${patchingDiagnostics.recommendedPatchLen} and stride ${patchingDiagnostics.recommendedStride}.`,
        targetSection: "prep",
        targetTab: "patching",
      });
    }

    recommendedActions.push({
      key: "inspect_patching",
      title: "Review patching geometry",
      detail: `Patch fit ${patchingDiagnostics.patchingLabel}; start near L=${patchingDiagnostics.recommendedPatchLen}, S=${patchingDiagnostics.recommendedStride}, with multi-scale usefulness ${patchingDiagnostics.multiscaleLabel}.`,
      priority: patchingDiagnostics.patchingLabel === "good" || patchingDiagnostics.multiscaleLabel === "high" ? "medium" : "low",
      targetSection: "prep",
      targetTab: "patching",
    });
  }

  if (calendarDiagnostics?.available && calendarDiagnostics.recommendation === "recommended") {
    highlights.push({
      text: `${calendarDiagnostics.strongestProfile} effects look ${calendarDiagnostics.label} with effect share ${(calendarDiagnostics.strongestEffectShare * 100).toFixed(1)}%.`,
      targetSection: "explore",
      targetTab: "calendar",
    });
  }

  if (intermittencyDiagnostics?.available && intermittencyDiagnostics.recommendation !== "optional") {
    risks.push({
      text: `Target intermittency looks ${intermittencyDiagnostics.classification} with zero share ${(intermittencyDiagnostics.zeroShare * 100).toFixed(1)}% and ADI ${intermittencyDiagnostics.adi}.`,
      targetSection: "prep",
      targetTab: "intermittency",
    });
    recommendedActions.push({
      key: "inspect_intermittency",
      title: "Inspect sparse-demand structure",
      detail: `Intermittency class ${intermittencyDiagnostics.classification} with longest idle run ${intermittencyDiagnostics.longestIdleRun}; ${intermittencyDiagnostics.baselineHints?.[0]?.label ?? "zero-aware baseline"} is the first comparison to run.`,
      priority: intermittencyDiagnostics.recommendation === "recommended" ? "high" : "medium",
      targetSection: "prep",
      targetTab: "intermittency",
    });
  } else if (intermittencyDiagnostics?.available) {
    highlights.push({
      text: `Target activity looks ${intermittencyDiagnostics.classification} with active share ${(intermittencyDiagnostics.activeShare * 100).toFixed(1)}%.`,
      targetSection: "prep",
      targetTab: "intermittency",
    });
  }

  if (volatilityDiagnostics?.available && volatilityDiagnostics.label !== "stable") {
    risks.push({
      text: `Volatility is ${volatilityDiagnostics.label} with ARCH p=${volatilityDiagnostics.archPValue} and regime ratio ${volatilityDiagnostics.regimeRatio}.`,
      targetSection: "prep",
      targetTab: "volatility",
    });
    recommendedActions.push({
      key: "inspect_volatility",
      title: "Inspect volatility clustering",
      detail: `Volatility regime ratio ${volatilityDiagnostics.regimeRatio}, clustering score ${volatilityDiagnostics.clusteringScore}, preferred transform ${volatilityDiagnostics.varianceTransformGuidance?.preferredTransform ?? "none"}.`,
      priority: volatilityDiagnostics.recommendation === "recommended" ? "high" : "medium",
      targetSection: "prep",
      targetTab: "volatility",
    });
  } else if (volatilityDiagnostics?.available) {
    highlights.push({
      text: `Volatility looks stable with ARCH p=${volatilityDiagnostics.archPValue}.`,
      targetSection: "prep",
      targetTab: "volatility",
    });
  }

  if (grangerDiagnostics?.available && grangerDiagnostics.topDriver?.significance === "significant") {
    highlights.push({
      text: `${grangerDiagnostics.topDriver.name} Granger-causes the target at lag ${grangerDiagnostics.topDriver.bestLag} with p=${grangerDiagnostics.topDriver.pValue}.`,
      targetSection: "explore",
      targetTab: "granger",
    });
    recommendedActions.push({
      key: "inspect_granger",
      title: `Inspect Granger driver ${grangerDiagnostics.topDriver.name}`,
      detail: `Best lag ${grangerDiagnostics.topDriver.bestLag}, F=${grangerDiagnostics.topDriver.fStatistic}, p=${grangerDiagnostics.topDriver.pValue}, using ${grangerDiagnostics.topDriver.mode} mode.`,
      priority: "medium",
      targetSection: "explore",
      targetTab: "granger",
    });
  }

  const readinessLabel = adjustedReadiness >= 80 ? "ready" : adjustedReadiness >= 60 ? "usable" : adjustedReadiness >= 40 ? "caution" : "fragile";
  const topActions = recommendedActions
    .sort((left, right) => {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      return (priorityOrder[left.priority] ?? 3) - (priorityOrder[right.priority] ?? 3);
    })
    .slice(0, 6);

  return {
    readinessScore: adjustedReadiness,
    readinessLabel,
    dataQualityScore,
    consensusScore,
    automationConfidence,
    topActions,
    risks: risks.slice(0, 5),
    highlights: highlights.slice(0, 5),
  };
}

export function computeSeriesDiagnostics({ series, covariates = [], horizon = 12, windowSize = 48, changePointMethod = "segmentation", datasetSummary = null }) {
  const values = series.map((item) => item.value);
  const count = values.length;
  const mean = average(values);
  const seriesVariance = variance(values, mean);
  const std = Math.sqrt(seriesVariance);
  const center = (count - 1) / 2;
  const slope = values.reduce((total, value, index) => total + (index - center) * (value - mean), 0) / values.reduce((total, _value, index) => total + (index - center) ** 2, 0);
  const sorted = [...values].sort((left, right) => left - right);
  const q1 = sorted[Math.floor(count * 0.25)];
  const q3 = sorted[Math.floor(count * 0.75)];
  const iqr = q3 - q1;
  const outliers = values.filter((value) => value < q1 - 1.5 * iqr || value > q3 + 1.5 * iqr).length;
  const lag = Math.min(24, Math.max(6, Math.floor(count / 8)));
  const lagCorrelation = values.slice(0, count - lag).reduce((total, value, index) => total + (value - mean) * (values[index + lag] - mean), 0) / Math.max(1, (count - lag) * seriesVariance);
  const volatility = std / Math.max(1e-6, Math.abs(mean));
  const trendLabel = Math.abs(slope) < 0.02 ? "stable" : slope > 0 ? "upward" : "downward";
  const maxLag = Math.min(48, Math.max(12, Math.floor(count / 4)));
  const acf = Array.from({ length: maxLag + 1 }, (_value, index) => ({ lag: index, value: index === 0 ? 1 : autocorrelation(values, index, mean, seriesVariance) }));
  const pacf = buildPacf(values, maxLag, mean, seriesVariance);
  const threshold = 1.96 / Math.sqrt(count);
  const pacfPeaks = detectCorrelationPeaks(pacf, threshold);
  const acfPeaks = detectCorrelationPeaks(acf, threshold);
  const dominantPacfLag = pacfPeaks.find((point) => point.lag >= 6)?.lag ?? pacfPeaks[0]?.lag ?? 1;
  const dominantAcfLag = acfPeaks.find((point) => point.lag >= 6)?.lag ?? acfPeaks[0]?.lag ?? dominantPacfLag;
  const adf = runAdfTest(values);
  const kpss = runKpssTest(values);
  const stationarity = classifyStationarity(adf, kpss, slope);
  const frequencyHint = inferFrequencyHint(series);
  const calendarDiagnostics = computeCalendarDiagnostics(series, frequencyHint);
  const intermittencyDiagnostics = computeIntermittencyDiagnostics(series);
  const volatilityDiagnostics = computeVolatilityDiagnostics(series);
  const seasonalPeriods = inferSeasonalPeriods(acfPeaks, count);
  const decomposition = buildMstlStyleDecomposition(series, seasonalPeriods);
  const seasonalityStrength = decomposition.strength;
  const seasonal = seasonalityStrength > 0.25;
  const noisy = volatility > 0.12;
  const evaluatedHorizon = clamp(horizon || Math.round(windowSize / 4), 3, 24);
  const benchmark = rollingOriginBacktest(values, seasonalPeriods[0], evaluatedHorizon);
  const exogenousScreening = screenExogenousCovariates(values, covariates, Math.min(24, Math.max(6, seasonalPeriods[0] || evaluatedHorizon)));
  const changePoints = detectChangePoints(series, changePointMethod);
  const winningResiduals = benchmark.winner?.residuals ?? [];
  const residualDiagnostics = runLjungBoxTest(winningResiduals, Math.min(12, evaluatedHorizon + 4));
  const stabilityDiagnostics = evaluateRollingStability(series, covariates, seasonalPeriods, evaluatedHorizon);
  const spectralDiagnostics = computeSpectralDiagnostics(series);
  const forecastabilityDiagnostics = computeForecastabilityDiagnostics(spectralDiagnostics, acfPeaks);
  const patchingDiagnostics = computePatchingDiagnostics(series, seasonalPeriods, acfPeaks, spectralDiagnostics, forecastabilityDiagnostics);
  const emdDiagnostics = computeEmdDiagnostics(series);
  const grangerDiagnostics = computeGrangerDiagnostics(series, covariates, windowSize, horizon);
  const baseWindow = clamp(Math.round(count / 5), 24, 96);
  const suggestedLagWindow = clamp(Math.max(baseWindow, dominantPacfLag * 3, dominantAcfLag * 2, seasonal ? seasonalPeriods[0] * 2 : 24), 24, 96);
  const recommendedHorizon = clamp(Math.round(suggestedLagWindow / 4), 6, 24);
  const scalingMethod = noisy || stationarity.label === "inconclusive" || volatilityDiagnostics.recommendation === "recommended" ? "robust" : "standard";
  let recommendedFamily = count < 140 ? "direct" : seasonal ? "transformer" : noisy ? "gru" : "lstm";
  if (forecastabilityDiagnostics.label === "low") {
    recommendedFamily = "direct";
  }
  if (intermittencyDiagnostics.recommendation === "recommended") {
    recommendedFamily = "direct";
  }
  if (benchmark.winner?.name === "naive" || benchmark.winner?.name === "moving_average") {
    recommendedFamily = "direct";
  } else if (benchmark.winner?.name === "seasonal_naive" && seasonal) {
    recommendedFamily = count > 220 ? "transformer" : "lstm";
  }
  const automationFlags = [
    stationarity.needsDiff ? "Enable differencing" : "Keep differencing off",
    stationarity.needsDetrend ? "Enable detrending" : "No deterministic detrending needed",
    noisy ? "Use robust scaling" : "Standard scaling is sufficient",
    seasonal ? `Seasonality strength ${(seasonalityStrength * 100).toFixed(0)}%` : "Weak seasonal structure",
    calendarDiagnostics.available ? `Calendar effects ${calendarDiagnostics.label}` : "Calendar effects unavailable",
    intermittencyDiagnostics.available ? `Intermittency ${intermittencyDiagnostics.classification}` : "Intermittency unavailable",
    volatilityDiagnostics.available ? `Volatility ${volatilityDiagnostics.label}` : "Volatility unavailable",
    benchmark.winner ? `Best baseline: ${benchmark.winner.label}` : "Baseline ladder pending",
    exogenousScreening[0]?.recommendation === "strong" ? `Top exogenous: ${exogenousScreening[0].name} (${exogenousScreening[0].leadLagStability?.label ?? "lag pending"})` : "No strong exogenous signal yet",
    grangerDiagnostics.available ? `Granger top driver: ${grangerDiagnostics.topDriver?.name ?? "none"}` : "Granger causality unavailable",
    changePoints.points.length > 0 ? `${changePoints.points.length} change point(s)` : "No major change points",
    residualDiagnostics.rejectWhiteNoise ? "Winning baseline leaves autocorrelation" : "Winning baseline residuals look white-ish",
    stabilityDiagnostics.seasonalityLabel === "stable" ? "Seasonality is stable over time" : "Seasonality shifts across windows",
    forecastabilityDiagnostics.available ? `Forecastability ${forecastabilityDiagnostics.label}` : "Forecastability pending",
    patchingDiagnostics.available ? `Patching ${patchingDiagnostics.patchingLabel}` : "Patching pending",
    patchingDiagnostics.available ? `Multiscale ${patchingDiagnostics.multiscaleLabel}` : "Multi-scale pending",
  ];
  const analysis = {
    count,
    mean: mean.toFixed(1),
    std: std.toFixed(2),
    outliers,
    outlierRate: outliers / count,
    seasonal,
    seasonalityStrength: Number(seasonalityStrength.toFixed(3)),
    seasonalPeriods,
    noisy,
    trendLabel,
    stationarityLabel: stationarity.label,
    automationSummary: stationarity.automationSummary,
    automationFlags,
    adfStatistic: adf.statistic.toFixed(3),
    adfCriticalValue: adf.criticalValue.toFixed(2),
    adfRejectUnitRoot: adf.rejectUnitRoot,
    adfLagOrder: adf.lagOrder,
    kpssStatistic: kpss.statistic.toFixed(3),
    kpssCriticalValue: kpss.criticalValue.toFixed(3),
    kpssRejectStationarity: kpss.rejectStationarity,
    kpssBandwidth: kpss.bandwidth,
    lagCorrelation: lagCorrelation.toFixed(3),
    pacfThreshold: threshold.toFixed(3),
    pacfPeaks: pacfPeaks.map((point) => ({ lag: point.lag, value: Number(point.value.toFixed(3)) })),
    acfPeaks: acfPeaks.map((point) => ({ lag: point.lag, value: Number(point.value.toFixed(3)) })),
    dominantPacfLag,
    dominantAcfLag,
    recommendedFamily,
    recommendedWindow: suggestedLagWindow,
    suggestedLagWindow,
    recommendedHorizon,
    scalingMethod,
    needsDetrend: stationarity.needsDetrend,
    needsDiff: stationarity.needsDiff,
    needsFilter: volatility > 0.09,
    useTimeFeatures: seasonal || calendarDiagnostics.recommendation === "recommended",
    frequencyHint,
    transformerAttentionMode: seasonal ? "standard" : "linear",
    changePointCount: changePoints.points.length,
    changePointLabel: changePoints.label,
    residualWhitenessLabel: residualDiagnostics.label,
    residualWhitenessPValue: residualDiagnostics.pValue,
    seasonalityStabilityLabel: stabilityDiagnostics.seasonalityLabel,
    exogenousStabilityLabel: stabilityDiagnostics.exogenousLabel,
    forecastabilityScore: forecastabilityDiagnostics.forecastabilityScore,
    forecastabilityLabel: forecastabilityDiagnostics.label,
    spectralEntropy: forecastabilityDiagnostics.spectralEntropy,
    complexityLabel: forecastabilityDiagnostics.complexityLabel,
    intermittencyLabel: intermittencyDiagnostics.label,
    intermittencyClass: intermittencyDiagnostics.classification,
    volatilityLabel: volatilityDiagnostics.label,
    varianceTransformLabel: volatilityDiagnostics.varianceTransformGuidance?.preferredTransform ?? "none",
    patchingLabel: patchingDiagnostics.patchingLabel,
    patchingScore: patchingDiagnostics.patchingScore,
    recommendedPatchLen: patchingDiagnostics.recommendedPatchLen,
    recommendedPatchStride: patchingDiagnostics.recommendedStride,
    multiscaleLabel: patchingDiagnostics.multiscaleLabel,
  };
  const automationDiagnostics = buildAutomationDiagnostics({
    analysis,
    datasetSummary,
    benchmark,
    exogenousScreening,
    grangerDiagnostics,
    changePoints,
    residualDiagnostics,
    stabilityDiagnostics,
    spectralDiagnostics,
    forecastabilityDiagnostics,
    patchingDiagnostics,
    calendarDiagnostics,
    intermittencyDiagnostics,
    volatilityDiagnostics,
  });

  return {
    analysis,
    acfCurve: acf.slice(1).map((point) => ({ lag: point.lag, acf: Number(point.value.toFixed(3)) })),
    pacfCurve: pacf.slice(1).map((point) => ({ lag: point.lag, pacf: Number(point.value.toFixed(3)) })),
    decomposition,
    benchmark,
    exogenousScreening,
    grangerDiagnostics,
    changePoints,
    residualDiagnostics,
    stabilityDiagnostics,
    spectralDiagnostics,
    forecastabilityDiagnostics,
    patchingDiagnostics,
    emdDiagnostics,
    grangerDiagnostics,
    calendarDiagnostics,
    intermittencyDiagnostics,
    volatilityDiagnostics,
    automationDiagnostics,
  };
}
