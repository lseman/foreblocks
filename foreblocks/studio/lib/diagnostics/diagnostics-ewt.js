import { estimateDominantFrequency, hasNonFiniteValue } from "./diagnostics-utils.js";

const EPS = 1e-12;
const TWO_PI = 2 * Math.PI;

function mean(values) {
  const n = values.length;
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) total += values[i];
  return total / n;
}

function meanAbs(values) {
  const n = values.length;
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) total += Math.abs(values[i]);
  return total / n;
}

function energy(values) {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    total += v * v;
  }
  return total;
}

function meanAndStd(values) {
  const n = values.length;
  if (n === 0) return { mean: 0, std: 0 };

  let m = 0;
  let m2 = 0;
  let count = 0;

  for (let i = 0; i < n; i += 1) {
    count += 1;
    const x = values[i];
    const delta = x - m;
    m += delta / count;
    const delta2 = x - m;
    m2 += delta * delta2;
  }

  return {
    mean: m,
    std: Math.sqrt(m2 / n),
  };
}

function roundArray(values, digits = 6) {
  return Array.from(values, (value) => Number(value.toFixed(digits)));
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
  if (n < 3) return { indices: [], values: [] };

  const indices = [];
  const isMax = kind === "max";

  const better = (a, b) => (isMax ? a > b + tolerance : a < b - tolerance);
  const notWorse = (a, b) => (isMax ? a >= b - tolerance : a <= b + tolerance);

  if (better(values[0], values[1])) {
    indices.push(0);
  }

  let i = 1;
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

function nextPowerOfTwo(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function complexMul(ar, ai, br, bi) {
  return [ar * br - ai * bi, ar * bi + ai * br];
}

function fft(re, im, inverse = false) {
  const n = re.length;
  if (n <= 1) {
    return {
      re: Float64Array.from(re),
      im: Float64Array.from(im),
    };
  }

  const outRe = Float64Array.from(re);
  const outIm = Float64Array.from(im);

  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;

    if (i < j) {
      const tr = outRe[i];
      const ti = outIm[i];
      outRe[i] = outRe[j];
      outIm[i] = outIm[j];
      outRe[j] = tr;
      outIm[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? TWO_PI : -TWO_PI) / len;
    const wlenCos = Math.cos(ang);
    const wlenSin = Math.sin(ang);

    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;

      for (let j2 = 0; j2 < len / 2; j2 += 1) {
        const uRe = outRe[i + j2];
        const uIm = outIm[i + j2];
        const vRe = outRe[i + j2 + len / 2];
        const vIm = outIm[i + j2 + len / 2];

        const [tRe, tIm] = complexMul(wRe, wIm, vRe, vIm);

        outRe[i + j2] = uRe + tRe;
        outIm[i + j2] = uIm + tIm;
        outRe[i + j2 + len / 2] = uRe - tRe;
        outIm[i + j2 + len / 2] = uIm - tIm;

        const nextWRe = wRe * wlenCos - wIm * wlenSin;
        const nextWIm = wRe * wlenSin + wIm * wlenCos;
        wRe = nextWRe;
        wIm = nextWIm;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i += 1) {
      outRe[i] /= n;
      outIm[i] /= n;
    }
  }

  return { re: outRe, im: outIm };
}

function fftReal(signal) {
  const n = nextPowerOfTwo(signal.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < signal.length; i += 1) {
    re[i] = signal[i];
  }
  return fft(re, im, false);
}

function ifftReal(re, im, originalLength) {
  const inv = fft(re, im, true);
  return inv.re.slice(0, originalLength);
}

function magnitudeSpectrum(re, im) {
  const n = re.length;
  const half = Math.floor(n / 2);
  const mag = new Float64Array(half + 1);
  for (let i = 0; i <= half; i += 1) {
    mag[i] = Math.hypot(re[i], im[i]);
  }
  return mag;
}

function smoothArray(values, windowSize = 5) {
  const n = values.length;
  if (windowSize <= 1 || n <= 2) return Float64Array.from(values);

  const radius = Math.max(1, Math.floor(windowSize / 2));
  const out = new Float64Array(n);

  for (let i = 0; i < n; i += 1) {
    let total = 0;
    let count = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(n - 1, i + radius); j += 1) {
      total += values[j];
      count += 1;
    }
    out[i] = total / count;
  }

  return out;
}

function detectSpectralPeaks(spectrum, maxBands = 6, minPeakDistance = 8, relativeThreshold = 0.05) {
  const n = spectrum.length;
  if (n < 3) return [];

  let maxValue = 0;
  for (let i = 1; i < n; i += 1) {
    if (spectrum[i] > maxValue) maxValue = spectrum[i];
  }
  const threshold = maxValue * relativeThreshold;

  const peaks = [];
  for (let i = 1; i < n - 1; i += 1) {
    const prev = spectrum[i - 1];
    const curr = spectrum[i];
    const next = spectrum[i + 1];
    if (curr >= prev && curr > next && curr >= threshold) {
      peaks.push({ index: i, value: curr });
    }
  }

  peaks.sort((a, b) => b.value - a.value);

  const selected = [];
  for (let i = 0; i < peaks.length; i += 1) {
    const peak = peaks[i];
    let tooClose = false;
    for (let j = 0; j < selected.length; j += 1) {
      if (Math.abs(selected[j].index - peak.index) < minPeakDistance) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      selected.push(peak);
      if (selected.length >= maxBands) break;
    }
  }

  selected.sort((a, b) => a.index - b.index);
  return selected;
}

function argMinInRange(values, left, right) {
  let bestIndex = left;
  let bestValue = values[left];
  for (let i = left + 1; i <= right; i += 1) {
    if (values[i] < bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

// Finds the spectral valley (minimum) between each consecutive pair of peaks.
// spectrum must be a typed or plain array of magnitude values.
function buildBoundariesFromPeaks(peaks, spectrum) {
  if (peaks.length <= 1) return [];
  const boundaries = [];
  for (let i = 0; i < peaks.length - 1; i += 1) {
    const left = peaks[i].index;
    const right = peaks[i + 1].index;
    boundaries.push(argMinInRange(spectrum, left, right));
  }
  return boundaries;
}

function betaTransition(x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x * x * (3 - 2 * x);
}

function empiricalMeyerScaling(omega, boundary, gamma) {
  const wp = boundary;
  const lower = (1 - gamma) * wp;
  const upper = (1 + gamma) * wp;

  if (omega <= lower) return 1;
  if (omega >= upper) return 0;

  const x = (omega - lower) / Math.max(EPS, upper - lower);
  return Math.cos((Math.PI / 2) * betaTransition(x));
}

function empiricalMeyerWavelet(omega, leftBoundary, rightBoundary, gamma) {
  const leftLow = (1 - gamma) * leftBoundary;
  const leftHigh = (1 + gamma) * leftBoundary;
  const rightLow = (1 - gamma) * rightBoundary;
  const rightHigh = (1 + gamma) * rightBoundary;

  if (omega <= leftLow || omega >= rightHigh) return 0;
  if (omega >= leftHigh && omega <= rightLow) return 1;

  if (omega > leftLow && omega < leftHigh) {
    const x = (omega - leftLow) / Math.max(EPS, leftHigh - leftLow);
    return Math.sin((Math.PI / 2) * betaTransition(x));
  }

  const x = (omega - rightLow) / Math.max(EPS, rightHigh - rightLow);
  return Math.cos((Math.PI / 2) * betaTransition(x));
}

function makeFrequencyGrid(n) {
  const omega = new Float64Array(n);
  for (let k = 0; k < n; k += 1) {
    let freqIndex = k;
    if (k > n / 2) freqIndex = n - k;
    omega[k] = Math.PI * freqIndex / (n / 2);
  }
  return omega;
}

function buildFilterBank(n, boundaries, gamma = 0.1) {
  const omega = makeFrequencyGrid(n);
  const positiveBoundaries = boundaries.map((b) => Math.PI * b / (n / 2));

  const filters = [];

  if (positiveBoundaries.length === 0) {
    const flat = new Float64Array(n);
    flat.fill(1);
    filters.push(flat);
    return filters;
  }

  const scaling = new Float64Array(n);
  for (let k = 0; k < n; k += 1) {
    scaling[k] = empiricalMeyerScaling(omega[k], positiveBoundaries[0], gamma);
  }
  filters.push(scaling);

  for (let i = 0; i < positiveBoundaries.length - 1; i += 1) {
    const filt = new Float64Array(n);
    for (let k = 0; k < n; k += 1) {
      filt[k] = empiricalMeyerWavelet(
        omega[k],
        positiveBoundaries[i],
        positiveBoundaries[i + 1],
        gamma,
      );
    }
    filters.push(filt);
  }

  const lastBoundary = positiveBoundaries[positiveBoundaries.length - 1];
  const highpass = new Float64Array(n);
  const low = (1 - gamma) * lastBoundary;
  const high = (1 + gamma) * lastBoundary;

  for (let k = 0; k < n; k += 1) {
    const w = omega[k];
    if (w <= low) {
      highpass[k] = 0;
    } else if (w >= high) {
      highpass[k] = 1;
    } else {
      const x = (w - low) / Math.max(EPS, high - low);
      highpass[k] = Math.sin((Math.PI / 2) * betaTransition(x));
    }
  }
  filters.push(highpass);

  return filters;
}

function applyFilter(re, im, filter) {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    outRe[i] = re[i] * filter[i];
    outIm[i] = im[i] * filter[i];
  }
  return { re: outRe, im: outIm };
}

function rmse(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return Math.sqrt(total / n);
}

function summarizeComponent(values, originalEnergy) {
  return {
    energyShare: Number((energy(values) / (originalEnergy + EPS)).toFixed(4)),
    zeroCrossings: countZeroCrossings(values),
    extremaCount: countExtrema(values),
    meanAbs: Number(meanAbs(values).toFixed(4)),
  };
}

function reconstructSignal(components, residual) {
  const n = residual.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) out[i] = residual[i];

  for (let k = 0; k < components.length; k += 1) {
    const c = components[k];
    for (let i = 0; i < n; i += 1) {
      out[i] += c[i];
    }
  }

  return out;
}

const UNAVAILABLE = (reason) => ({
  available: false,
  method: "ewt_empirical_meyer",
  methodLabel: "Empirical wavelet transform",
  interpolationLabel: "Empirical Meyer filter bank",
  reason,
  components: [],
  residual: null,
  bandCount: 0,
  reconstructionError: 0,
  residualEnergyShare: 0,
  boundaries: [],
});

export function computeEwtDiagnostics(
  series,
  maxBands = 5,
  smoothingWindow = 7,
  detectThreshold = 0.05,
  gamma = 0.1,
) {
  const values = Float64Array.from(series, (point) => point.value);
  const count = values.length;

  if (count < 16) {
    return UNAVAILABLE("EWT needs at least 16 observations.");
  }

  if (hasNonFiniteValue(values)) {
    return UNAVAILABLE("EWT requires a complete series of finite values.");
  }

  const { mean: signalMean, std: signalStd } = meanAndStd(values);
  if (signalStd < 1e-10) {
    return UNAVAILABLE("EWT needs a non-constant signal.");
  }

  const centered = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    centered[i] = values[i] - signalMean;
  }

  const spectrum = fftReal(centered);
  const fftLength = spectrum.re.length;
  const mag = magnitudeSpectrum(spectrum.re, spectrum.im);
  const smoothedMag = smoothArray(mag, smoothingWindow);

  const peaks = detectSpectralPeaks(
    smoothedMag,
    Math.max(1, maxBands - 1),
    Math.max(4, Math.floor(count / 64)),
    detectThreshold,
  );

  if (peaks.length === 0) {
    return UNAVAILABLE("Could not detect spectral peaks for adaptive partitioning.");
  }

  // FIX 2: A single peak means no inter-peak boundaries can be formed, so no
  // oscillatory bands can be separated. Report unavailable rather than returning
  // available: true with an empty component list.
  if (peaks.length === 1) {
    return UNAVAILABLE("Only one spectral peak detected; cannot form oscillatory bands.");
  }

  // FIX 1: removed dead `buildBoundariesFromPeaks` function that incorrectly
  // received a number instead of an array as its second argument.
  // `buildBoundariesFromPeaks` now IS the canonical implementation (renamed,
  // corrected, singular).
  const boundariesBins = buildBoundariesFromPeaks(peaks, smoothedMag);
  const filters = buildFilterBank(fftLength, boundariesBins, gamma);

  const rawBands = [];
  for (let k = 0; k < filters.length; k += 1) {
    const filtered = applyFilter(spectrum.re, spectrum.im, filters[k]);
    rawBands.push(ifftReal(filtered.re, filtered.im, count));
  }

  // First band = low-frequency scaling / trend component.
  // Remaining bands = oscillatory EWT components, sorted high → low frequency.
  const trend = rawBands[0];
  const componentsRaw = rawBands.slice(1);

  componentsRaw.sort((a, b) => countZeroCrossings(b) - countZeroCrossings(a));

  const originalEnergy = energy(values);
  const components = [];

  for (let i = 0; i < componentsRaw.length; i += 1) {
    const vals = componentsRaw[i];
    const summary = summarizeComponent(vals, originalEnergy);
    components.push({
      order: i + 1,
      name: `EWT ${i + 1}`,
      values: roundArray(vals, 6),
      energyShare: summary.energyShare,
      zeroCrossings: summary.zeroCrossings,
      extremaCount: summary.extremaCount,
      meanAbs: summary.meanAbs,
      band: i + 1,
      centerFrequency: Number(estimateDominantFrequency(vals).toFixed(6)),
    });
  }

  const residual = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    residual[i] = trend[i] + signalMean;
  }

  const reconstructed = reconstructSignal(componentsRaw, residual);
  const reconstructionError = rmse(reconstructed, values);
  const residualEnergy = energy(residual);

  return {
    available: true,
    method: "ewt_empirical_meyer",
    methodLabel: "Empirical wavelet transform",
    interpolationLabel: "Empirical Meyer filter bank",
    reason: "",
    components,
    residual: {
      name: "Residual",
      values: roundArray(residual, 6),
      energyShare: Number((residualEnergy / (originalEnergy + EPS)).toFixed(4)),
      zeroCrossings: countZeroCrossings(residual),
      extremaCount: countExtrema(residual),
      meanAbs: Number(meanAbs(residual).toFixed(4)),
    },
    bandCount: components.length,
    reconstructionError: Number(reconstructionError.toFixed(6)),
    residualEnergyShare: Number((residualEnergy / (originalEnergy + EPS)).toFixed(4)),
    boundaries: boundariesBins.map((bin) => ({
      bin,
      normalizedFrequency: Number((bin / Math.max(1, mag.length - 1)).toFixed(4)),
    })),
    peaks: peaks.map((peak) => ({
      bin: peak.index,
      amplitude: Number(peak.value.toFixed(6)),
      normalizedFrequency: Number((peak.index / Math.max(1, mag.length - 1)).toFixed(4)),
    })),
    maxBands,
    smoothingWindow,
    detectThreshold: Number(detectThreshold.toFixed(3)),
    gamma: Number(gamma.toFixed(3)),
  };
}
