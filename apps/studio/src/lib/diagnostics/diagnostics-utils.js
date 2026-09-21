export const EPS = 1e-12;
export const TWO_PI = 2 * Math.PI;

export function clamp(value, min, max) {
  return value < min ? min : value > max ? max : value;
}

export function hasNonFiniteValue(values) {
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) {
      return true;
    }
  }
  return false;
}

export function mean(values) {
  const n = values.length;
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    total += values[i];
  }
  return total / n;
}

export function meanAbs(values) {
  const n = values.length;
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    total += Math.abs(values[i]);
  }
  return total / n;
}

export function energy(values) {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    total += v * v;
  }
  return total;
}

export function meanAndStd(values) {
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

export function standardDeviation(values, meanValue = mean(values)) {
  const n = values.length;
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = values[i] - meanValue;
    total += diff * diff;
  }
  return Math.sqrt(total / n);
}

export function roundArray(values, digits = 6) {
  return Array.from(values, (value) => Number(value.toFixed(digits)));
}

export function countZeroCrossings(values) {
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

export function findExtrema(values, kind = "max", tolerance = 1e-12) {
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

export function countExtrema(values) {
  return findExtrema(values, "max").indices.length + findExtrema(values, "min").indices.length;
}

export function nextPowerOfTwo(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function complexMul(ar, ai, br, bi) {
  return [ar * br - ai * bi, ar * bi + ai * br];
}

export function fft(re, im, inverse = false) {
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

export function fftReal(signal) {
  const n = nextPowerOfTwo(signal.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < signal.length; i += 1) {
    re[i] = signal[i];
  }
  return fft(re, im, false);
}

export function ifftReal(re, im, originalLength) {
  const inv = fft(re, im, true);
  return inv.re.slice(0, originalLength);
}

export function fftShift(re, im) {
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

export function ifftShift(re, im) {
  return fftShift(re, im);
}

export function mirrorExtend(values) {
  const n = values.length;
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i += 1) {
    out[i] = values[n - 1 - i];
    out[n + i] = values[i];
  }
  return out;
}

export function cropCenter(values, originalLength) {
  const start = Math.floor((values.length - originalLength) / 2);
  return values.slice(start, start + originalLength);
}

export function frequencyGridShifted(n) {
  const out = new Float64Array(n);
  const half = n / 2;
  for (let i = 0; i < n; i += 1) {
    out[i] = (i - half) / n;
  }
  return out;
}

export function magnitudeSpectrum(re, im) {
  const n = Math.min(re.length, im.length);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = Math.hypot(re[i], im[i]);
  }
  return out;
}

export function estimateDominantFrequency(values) {
  const n = values.length;
  if (n < 4) {
    return 0;
  }

  const meanValue = mean(values);
  const centered = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    centered[i] = values[i] - meanValue;
  }

  const spectrum = fftReal(centered);
  const half = spectrum.re.length >> 1;
  let bestIndex = 1;
  let bestMagnitude = 0;

  for (let k = 1; k < half; k += 1) {
    const magnitude = Math.hypot(spectrum.re[k], spectrum.im[k]);
    if (magnitude > bestMagnitude) {
      bestMagnitude = magnitude;
      bestIndex = k;
    }
  }

  return bestIndex / spectrum.re.length;
}

export function rmse(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = a[i] - b[i];
    total += diff * diff;
  }
  return Math.sqrt(total / n);
}
