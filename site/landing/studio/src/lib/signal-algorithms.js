function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function isFiniteNumber(value) {
    return Number.isFinite(value);
}

function average(values) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (isFiniteNumber(v)) {
            sum += v;
            count += 1;
        }
    }
    return count > 0 ? sum / count : 0;
}

function variance(values, mean = average(values)) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (isFiniteNumber(v)) {
            const d = v - mean;
            sum += d * d;
            count += 1;
        }
    }
    return count > 0 ? sum / count : 0;
}

function standardDeviation(values, mean = average(values)) {
    return Math.sqrt(variance(values, mean));
}

function median(values) {
    const finite = [];
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (isFiniteNumber(v)) {
            finite.push(v);
        }
    }
    if (finite.length === 0) {
        return 0;
    }
    finite.sort((a, b) => a - b);
    const mid = Math.floor(finite.length / 2);
    return finite.length % 2 === 0
        ? (finite[mid - 1] + finite[mid]) / 2
        : finite[mid];
}

function mad(values, center = median(values)) {
    const deviations = new Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        deviations[i] = isFiniteNumber(v) ? Math.abs(v - center) : null;
    }
    return median(deviations);
}

function round3(value) {
    return Number.isFinite(value) ? Number(value.toFixed(3)) : 0;
}

function inferPeriod(valuesLength, analysis) {
    const dominantAcfLag = Number.isFinite(analysis?.dominantAcfLag) ? analysis.dominantAcfLag : 12;
    const dominantPacfLag = Number.isFinite(analysis?.dominantPacfLag) ? analysis.dominantPacfLag : 6;
    const raw = Math.max(dominantAcfLag, dominantPacfLag);
    return clamp(Math.round(raw), 4, Math.min(24, Math.max(4, valuesLength - 1)));
}

function chooseOddWindow(size, minValue, maxValue) {
    let window = clamp(Math.round(size), minValue, maxValue);
    if (window % 2 === 0) {
        window += 1;
    }
    return clamp(window, minValue, maxValue % 2 === 0 ? maxValue - 1 : maxValue);
}

function weightedMovingAverage(values, weights, window) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const effectiveWindow = chooseOddWindow(window, 3, Math.max(3, n % 2 === 0 ? n - 1 : n));
    const radius = Math.floor(effectiveWindow / 2);

    const prefixWeightedSum = new Float64Array(n + 1);
    const prefixWeight = new Float64Array(n + 1);

    for (let i = 0; i < n; i += 1) {
        const value = values[i];
        const baseWeight = isFiniteNumber(weights?.[i]) ? Math.max(0, weights[i]) : 1;
        const effectiveWeight = isFiniteNumber(value) ? baseWeight : 0;

        prefixWeightedSum[i + 1] = prefixWeightedSum[i] + (effectiveWeight > 0 ? effectiveWeight * value : 0);
        prefixWeight[i + 1] = prefixWeight[i] + effectiveWeight;
    }

    const result = new Array(n);
    for (let i = 0; i < n; i += 1) {
        const start = Math.max(0, i - radius);
        const end = Math.min(n, i + radius + 1);

        const weightedSum = prefixWeightedSum[end] - prefixWeightedSum[start];
        const totalWeight = prefixWeight[end] - prefixWeight[start];

        if (totalWeight > 1e-12) {
            result[i] = weightedSum / totalWeight;
        } else {
            result[i] = isFiniteNumber(values[i]) ? values[i] : 0;
        }
    }

    return result;
}

function smoothSeasonalSubseries(detrended, weights, period, subseriesWindow) {
    const n = detrended.length;
    const seasonal = new Array(n).fill(0);

    for (let phase = 0; phase < period; phase += 1) {
        const subValues = [];
        const subWeights = [];
        const subIndices = [];

        for (let i = phase; i < n; i += period) {
            subValues.push(isFiniteNumber(detrended[i]) ? detrended[i] : 0);
            subWeights.push(isFiniteNumber(weights?.[i]) ? weights[i] : 1);
            subIndices.push(i);
        }

        if (subValues.length === 0) {
            continue;
        }

        const maxWindow = subValues.length % 2 === 0 ? subValues.length - 1 : subValues.length;
        const safeMaxWindow = Math.max(3, maxWindow);
        const localWindow = chooseOddWindow(subseriesWindow, 3, safeMaxWindow);
        const smoothed = weightedMovingAverage(subValues, subWeights, localWindow);

        for (let j = 0; j < subIndices.length; j += 1) {
            seasonal[subIndices[j]] = smoothed[j];
        }
    }

    return seasonal;
}

function normalizeSeasonal(seasonal, period) {
    const n = seasonal.length;
    if (n === 0) {
        return seasonal;
    }

    const phaseMeans = new Array(period).fill(0);
    const phaseCounts = new Array(period).fill(0);

    for (let i = 0; i < n; i += 1) {
        const value = seasonal[i];
        if (isFiniteNumber(value)) {
            const phase = i % period;
            phaseMeans[phase] += value;
            phaseCounts[phase] += 1;
        }
    }

    for (let phase = 0; phase < period; phase += 1) {
        phaseMeans[phase] = phaseCounts[phase] > 0 ? phaseMeans[phase] / phaseCounts[phase] : 0;
    }

    const overallMean = average(phaseMeans);
    const centeredPhaseMeans = phaseMeans.map((value) => value - overallMean);

    const result = new Array(n);
    for (let i = 0; i < n; i += 1) {
        result[i] = centeredPhaseMeans[i % period];
    }

    return result;
}

function subtractArrays(a, b) {
    const n = Math.min(a.length, b.length);
    const out = new Array(n);
    for (let i = 0; i < n; i += 1) {
        const av = isFiniteNumber(a[i]) ? a[i] : 0;
        const bv = isFiniteNumber(b[i]) ? b[i] : 0;
        out[i] = av - bv;
    }
    return out;
}

function addArrays(a, b) {
    const n = Math.min(a.length, b.length);
    const out = new Array(n);
    for (let i = 0; i < n; i += 1) {
        const av = isFiniteNumber(a[i]) ? a[i] : 0;
        const bv = isFiniteNumber(b[i]) ? b[i] : 0;
        out[i] = av + bv;
    }
    return out;
}

function robustBisquareWeights(residuals) {
    const scale = mad(residuals) * 1.4826;
    if (!Number.isFinite(scale) || scale < 1e-12) {
        return new Array(residuals.length).fill(1);
    }

    const c = 6 * scale;
    const weights = new Array(residuals.length);

    for (let i = 0; i < residuals.length; i += 1) {
        const r = Math.abs(isFiniteNumber(residuals[i]) ? residuals[i] : 0);
        if (r >= c) {
            weights[i] = 0;
        } else {
            const t = 1 - (r / c) ** 2;
            weights[i] = t * t;
        }
    }

    return weights;
}

function stlLite(values, period, options = {}) {
    const n = values.length;
    if (n === 0) {
        return {
            trend: [],
            seasonal: [],
            residual: [],
        };
    }

    const trendWindow = chooseOddWindow(
        options.trendWindow ?? Math.max(7, period * 2 + 1),
        3,
        Math.max(3, n % 2 === 0 ? n - 1 : n),
    );

    const subseriesWindow = chooseOddWindow(
        options.seasonalWindow ?? Math.max(5, Math.floor(period / 2) * 2 + 1),
        3,
        Math.max(3, Math.ceil(n / Math.max(1, period)) % 2 === 0
            ? Math.ceil(n / Math.max(1, period)) - 1
            : Math.ceil(n / Math.max(1, period))),
    );

    const innerIterations = clamp(options.innerIterations ?? 2, 1, 10);
    const outerIterations = clamp(options.outerIterations ?? 2, 1, 10);

    let robustnessWeights = new Array(n).fill(1);
    let trend = weightedMovingAverage(values, robustnessWeights, trendWindow);
    let seasonal = new Array(n).fill(0);
    let residual = new Array(n).fill(0);

    for (let outer = 0; outer < outerIterations; outer += 1) {
        for (let inner = 0; inner < innerIterations; inner += 1) {
            const detrended = subtractArrays(values, trend);
            const rawSeasonal = smoothSeasonalSubseries(detrended, robustnessWeights, period, subseriesWindow);
            seasonal = normalizeSeasonal(rawSeasonal, period);

            const deseasonalized = subtractArrays(values, seasonal);
            trend = weightedMovingAverage(deseasonalized, robustnessWeights, trendWindow);
        }

        residual = subtractArrays(values, addArrays(trend, seasonal));
        robustnessWeights = robustBisquareWeights(residual);
    }

    residual = subtractArrays(values, addArrays(trend, seasonal));
    return { trend, seasonal, residual };
}

export function buildDecomposition(data, analysis) {
    if (!Array.isArray(data) || data.length === 0) {
        return {
            periods: [],
            strength: 0,
            components: [],
            data: [],
            residualStd: 0,
        };
    }

    const values = data.map((point) => (isFiniteNumber(point?.value) ? point.value : 0));
    const period = inferPeriod(values.length, analysis);

    const { trend, seasonal, residual } = stlLite(values, period, {
        trendWindow: Math.max(7, period * 2 + 1),
        seasonalWindow: Math.max(5, Math.min(13, period)),
        innerIterations: 2,
        outerIterations: 2,
    });

    const decompositionData = data.map((point, index) => ({
        ...point,
        trend: round3(trend[index]),
        seasonal: round3(seasonal[index]),
        residual: round3(residual[index]),
    }));

    const observedVariance = variance(values);
    const remainderVariance = variance(residual);
    const strength = observedVariance > 1e-12
        ? clamp(1 - remainderVariance / observedVariance, 0, 1)
        : 0;

    return {
        periods: [period],
        strength,
        components: [{ period, strength }],
        data: decompositionData.map((point) => ({
            ...point,
            observed: point.value,
        })),
        residualStd: standardDeviation(residual),
    };
}

export function shiftValues(values, lag) {
    const n = values.length;
    const result = new Array(n);
    for (let i = 0; i < n; i += 1) {
        const sourceIndex = i - lag;
        result[i] = sourceIndex >= 0 && isFiniteNumber(values[sourceIndex]) ? values[sourceIndex] : null;
    }
    return result;
}

export function buildRollingHistory(values, windowSize, reducer) {
    const effectiveWindow = Math.max(2, Math.round(windowSize));
    const minimumCount = Math.max(2, Math.min(4, effectiveWindow - 1));

    return values.map((_value, index) => {
        const start = Math.max(0, index - effectiveWindow);
        const window = [];
        for (let i = start; i < index; i += 1) {
            const item = values[i];
            if (isFiniteNumber(item)) {
                window.push(item);
            }
        }
        return window.length >= minimumCount ? reducer(window) : null;
    });
}

