import { hasNonFiniteValue } from "./diagnostics-utils.js";

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

function fftShift(re, im) {
  const n = re.length;
  const half = n >> 1;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);

  for (let i = 0; i < half; i += 1) {
    outRe[i] = re[i + half];
    outIm[i] = im[i + half];
    outRe[i + half] = re[i];
    outIm[i + half] = im[i];
  }

  return { re: outRe, im: outIm };
}

function ifftShift(re, im) {
  return fftShift(re, im);
}

// FIX 1: Half-period mirror extension matching Dragomiretskiy & Zosso (2014).
// Layout: [reverse of first half | original signal | reverse of second half]
// Total length = 2 * n. cropCenter(result, n) then recovers exactly the
// positions corresponding to the original signal.
function mirrorExtend(values) {
  const n = values.length;
  const half = Math.floor(n / 2);
  const out = new Float64Array(2 * n);

  // First n/2 samples: reversed first half of the signal
  for (let i = 0; i < half; i += 1) {
    out[i] = values[half - 1 - i];
  }

  // Middle n samples: the original signal
  for (let i = 0; i < n; i += 1) {
    out[half + i] = values[i];
  }

  // Last n/2 samples: reversed second half of the signal
  for (let i = 0; i < half; i += 1) {
    out[half + n + i] = values[n - 1 - i];
  }

  return out;
}

function cropCenter(values, originalLength) {
  const start = Math.floor((values.length - originalLength) / 2);
  return values.slice(start, start + originalLength);
}

function frequencyGridShifted(n) {
  const out = new Float64Array(n);
  const half = n / 2;
  for (let i = 0; i < n; i += 1) {
    out[i] = (i - half) / n;
  }
  return out;
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

function reconstructSignal(components, residual) {
  const out = new Float64Array(residual.length);
  for (let i = 0; i < residual.length; i += 1) {
    out[i] = residual[i];
  }

  for (let k = 0; k < components.length; k += 1) {
    const c = components[k];
    for (let i = 0; i < out.length; i += 1) {
      out[i] += c[i];
    }
  }

  return out;
}

function modeSummary(values, originalEnergy) {
  return {
    energyShare: Number((energy(values) / (originalEnergy + EPS)).toFixed(4)),
    zeroCrossings: countZeroCrossings(values),
    extremaCount: countExtrema(values),
    meanAbs: Number(meanAbs(values).toFixed(4)),
  };
}

function initializeOmegas(K, dcMode) {
  const omegas = new Float64Array(K);
  if (dcMode) omegas[0] = 0;

  const startIndex = dcMode ? 1 : 0;
  const remaining = K - startIndex;

  for (let k = startIndex; k < K; k += 1) {
    const frac = remaining <= 1 ? 0.25 : (k - startIndex + 1) / (remaining + 1);
    omegas[k] = 0.5 * frac;
  }

  return omegas;
}

function vmdDecompose(
  signal,
  {
    K = 4,
    alpha = 2000,
    tau = 0,
    tol = 1e-7,
    maxIterations = 500,
    dcMode = false,
  } = {},
) {
  const extended = mirrorExtend(signal);

  const spectrum = fftReal(extended);
  const shifted = fftShift(spectrum.re, spectrum.im);
  const fRe = shifted.re;
  const fIm = shifted.im;

  // N is the padded FFT size — all spectral arrays use this consistently.
  const N = fRe.length;
  const freqs = frequencyGridShifted(N);

  const uHatRe = Array.from({ length: K }, () => new Float64Array(N));
  const uHatIm = Array.from({ length: K }, () => new Float64Array(N));
  const uHatPrevRe = Array.from({ length: K }, () => new Float64Array(N));
  const uHatPrevIm = Array.from({ length: K }, () => new Float64Array(N));

  const lambdaRe = new Float64Array(N);
  const lambdaIm = new Float64Array(N);

  const omegas = initializeOmegas(K, dcMode);
  const sumOthersRe = new Float64Array(N);
  const sumOthersIm = new Float64Array(N);

  let iterations = 0;
  let converged = false;
  let finalDiff = Infinity;

  for (let iter = 0; iter < maxIterations; iter += 1) {
    iterations = iter + 1;

    for (let k = 0; k < K; k += 1) {
      uHatPrevRe[k].set(uHatRe[k]);
      uHatPrevIm[k].set(uHatIm[k]);
    }

    for (let k = 0; k < K; k += 1) {
      sumOthersRe.fill(0);
      sumOthersIm.fill(0);

      for (let j = 0; j < K; j += 1) {
        if (j === k) continue;
        const uRe = uHatRe[j];
        const uIm = uHatIm[j];
        for (let n = 0; n < N; n += 1) {
          sumOthersRe[n] += uRe[n];
          sumOthersIm[n] += uIm[n];
        }
      }

      const omegaK = dcMode && k === 0 ? 0 : omegas[k];
      const curRe = uHatRe[k];
      const curIm = uHatIm[k];

      for (let n = 0; n < N; n += 1) {
        const freq = freqs[n];
        const diffFreq = freq - omegaK;
        const denom = 1 + 2 * alpha * diffFreq * diffFreq;

        const rhsRe = fRe[n] - sumOthersRe[n] + 0.5 * lambdaRe[n];
        const rhsIm = fIm[n] - sumOthersIm[n] + 0.5 * lambdaIm[n];

        curRe[n] = rhsRe / denom;
        curIm[n] = rhsIm / denom;
      }

      if (!(dcMode && k === 0)) {
        let num = 0;
        let den = 0;
        for (let n = Math.floor(N / 2); n < N; n += 1) {
          const p = curRe[n] * curRe[n] + curIm[n] * curIm[n];
          num += freqs[n] * p;
          den += p;
        }
        omegas[k] = den > EPS ? num / den : omegas[k];
      }
    }

    if (tau > 0) {
      for (let n = 0; n < N; n += 1) {
        let sumRe = 0;
        let sumIm = 0;
        for (let k = 0; k < K; k += 1) {
          sumRe += uHatRe[k][n];
          sumIm += uHatIm[k][n];
        }
        lambdaRe[n] += tau * (fRe[n] - sumRe);
        lambdaIm[n] += tau * (fIm[n] - sumIm);
      }
    }

    let diff = 0;
    let base = 0;
    for (let k = 0; k < K; k += 1) {
      for (let n = 0; n < N; n += 1) {
        const dr = uHatRe[k][n] - uHatPrevRe[k][n];
        const di = uHatIm[k][n] - uHatPrevIm[k][n];
        diff += dr * dr + di * di;
        base += uHatPrevRe[k][n] * uHatPrevRe[k][n] + uHatPrevIm[k][n] * uHatPrevIm[k][n];
      }
    }

    finalDiff = diff / (base + EPS);
    if (finalDiff < tol) {
      converged = true;
      break;
    }
  }

  const modes = [];
  for (let k = 0; k < K; k += 1) {
    const shiftedBack = ifftShift(uHatRe[k], uHatIm[k]);
    // IFFT over the full padded size N, then crop back to the extended signal
    // length, then extract the center window = original signal length.
    const timeMode = ifftReal(shiftedBack.re, shiftedBack.im, N);
    modes.push(cropCenter(timeMode, signal.length));
  }

  return {
    modes,
    omegas,
    iterations,
    converged,
    finalDiff,
  };
}

export function computeVmdDiagnostics(
  series,
  modeCount = 4,
  alpha = 2000,
  tau = 0,
  tolerance = 1e-7,
  maxIterations = 500,
  dcMode = false,
) {
  const values = Float64Array.from(series, (point) => point.value);
  const count = values.length;

  if (count < 16) {
    return {
      available: false,
      method: "vmd_admm",
      methodLabel: "Variational mode decomposition",
      interpolationLabel: "FFT-domain ADMM updates",
      reason: "VMD needs at least 16 observations.",
      components: [],
      residual: null,
      modeCount: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  if (hasNonFiniteValue(values)) {
    return {
      available: false,
      method: "vmd_admm",
      methodLabel: "Variational mode decomposition",
      interpolationLabel: "FFT-domain ADMM updates",
      reason: "VMD requires a complete series of finite values.",
      components: [],
      residual: null,
      modeCount: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const { mean: signalMean, std: signalStd } = meanAndStd(values);
  if (signalStd < 1e-10) {
    return {
      available: false,
      method: "vmd_admm",
      methodLabel: "Variational mode decomposition",
      interpolationLabel: "FFT-domain ADMM updates",
      reason: "VMD needs a non-constant signal.",
      components: [],
      residual: null,
      modeCount: 0,
      reconstructionError: 0,
      residualEnergyShare: 0,
    };
  }

  const centered = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    centered[i] = values[i] - signalMean;
  }

  const result = vmdDecompose(centered, {
    K: Math.max(1, modeCount | 0),
    alpha: Math.max(1, alpha),
    tau: Math.max(0, tau),
    tol: Math.max(1e-12, tolerance),
    maxIterations: Math.max(10, maxIterations | 0),
    dcMode,
  });

  const originalEnergy = energy(values);

  // FIX 2 & 3: Keep omegas paired with their modes before sorting so that
  // center frequencies remain correctly associated after reordering.
  const pairs = result.modes.map((mode, i) => ({ mode, omega: result.omegas[i] }));
  pairs.sort((a, b) => countZeroCrossings(b.mode) - countZeroCrossings(a.mode));

  const reconstructedCentered = new Float64Array(count);
  for (let k = 0; k < pairs.length; k += 1) {
    const mode = pairs[k].mode;
    for (let i = 0; i < count; i += 1) {
      reconstructedCentered[i] += mode[i];
    }
  }

  const residual = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    residual[i] = values[i] - (reconstructedCentered[i] + signalMean);
  }

  const components = [];
  for (let k = 0; k < pairs.length; k += 1) {
    const mode = new Float64Array(count);
    for (let i = 0; i < count; i += 1) {
      mode[i] = pairs[k].mode[i];
    }

    const summary = modeSummary(mode, originalEnergy);
    components.push({
      order: k + 1,
      name: `VMD ${k + 1}`,
      values: roundArray(mode, 6),
      energyShare: summary.energyShare,
      zeroCrossings: summary.zeroCrossings,
      extremaCount: summary.extremaCount,
      meanAbs: summary.meanAbs,
      centerFrequency: Number(Math.abs(pairs[k].omega).toFixed(6)),
    });
  }

  const residualWithMean = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    residualWithMean[i] = residual[i] + signalMean;
  }

  const reconstructed = reconstructSignal(
    pairs.map((p) => p.mode),
    residualWithMean,
  );

  const reconstructionError = rmse(reconstructed, values);
  const residualSummary = modeSummary(residualWithMean, originalEnergy);
  const residualEnergyShare = Number((energy(residualWithMean) / (originalEnergy + EPS)).toFixed(4));

  return {
    available: true,
    method: "vmd_admm",
    methodLabel: "Variational mode decomposition",
    interpolationLabel: "FFT-domain ADMM updates",
    reason: "",
    components,
    residual: {
      name: "Residual",
      values: roundArray(residualWithMean, 6),
      energyShare: residualEnergyShare,
      zeroCrossings: residualSummary.zeroCrossings,
      extremaCount: residualSummary.extremaCount,
      meanAbs: residualSummary.meanAbs,
    },
    modeCount: components.length,
    reconstructionError: Number(reconstructionError.toFixed(6)),
    residualEnergyShare,
    alpha: Number(alpha.toFixed(3)),
    tau: Number(tau.toFixed(3)),
    tolerance: Number(tolerance.toExponential(2)),
    maxIterations,
    iterationsUsed: result.iterations,
    converged: result.converged,
    finalDiff: Number(result.finalDiff.toExponential(3)),
    dcMode,
  };
}
