function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function average(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function median(values) {
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function medianAbsoluteDeviation(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length === 0) {
        return 0;
    }

    const med = median(finite);
    const deviations = finite.map((value) => Math.abs(value - med));
    return median(deviations);
}

function standardDeviation(values, mean) {
    return Math.sqrt(values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length);
}

function ensureOddWindow(value, minimum = 3) {
    const normalized = Math.max(minimum, Math.round(Number.isFinite(value) ? value : minimum));
    return normalized % 2 === 1 ? normalized : normalized + 1;
}

function solveLinearSystem(matrix, vector) {
    const size = matrix.length;
    const augmented = matrix.map((row, rowIndex) => [...row, vector[rowIndex]]);

    for (let pivot = 0; pivot < size; pivot += 1) {
        let pivotRow = pivot;
        for (let row = pivot + 1; row < size; row += 1) {
            if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[pivotRow][pivot])) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][pivot]) < 1e-10) {
            return null;
        }

        if (pivotRow !== pivot) {
            [augmented[pivot], augmented[pivotRow]] = [augmented[pivotRow], augmented[pivot]];
        }

        const pivotValue = augmented[pivot][pivot];
        for (let column = pivot; column <= size; column += 1) {
            augmented[pivot][column] /= pivotValue;
        }

        for (let row = 0; row < size; row += 1) {
            if (row === pivot) {
                continue;
            }
            const factor = augmented[row][pivot];
            for (let column = pivot; column <= size; column += 1) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }

    return augmented.map((row) => row[size]);
}

function fitPolynomial(xs, ys, order) {
    if (xs.length === 0 || ys.length === 0) {
        return null;
    }

    const degree = Math.min(order, xs.length - 1);
    const size = degree + 1;
    const system = Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
    const rhs = Array.from({ length: size }, () => 0);

    for (let row = 0; row < size; row += 1) {
        for (let column = 0; column < size; column += 1) {
            system[row][column] = xs.reduce((total, value) => total + value ** (row + column), 0);
        }
        rhs[row] = xs.reduce((total, value, index) => total + ys[index] * (value ** row), 0);
    }

    return solveLinearSystem(system, rhs);
}

function evaluatePolynomial(coefficients, x) {
    if (!coefficients) {
        return 0;
    }
    return coefficients.reduce((total, coefficient, power) => total + coefficient * (x ** power), 0);
}

function buildPolynomialTrend(values, order) {
    const xs = values.map((_value, index) => (values.length <= 1 ? 0 : (index / (values.length - 1)) * 2 - 1));
    const finite = values
        .map((value, index) => ({ value, x: xs[index] }))
        .filter((item) => Number.isFinite(item.value));

    if (finite.length < order + 1) {
        return values.map(() => 0);
    }

    const coefficients = fitPolynomial(
        finite.map((item) => item.x),
        finite.map((item) => item.value),
        order,
    );

    return xs.map((x) => evaluatePolynomial(coefficients, x));
}

function applyMovingAverage(values, windowSize) {
    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    return values.map((_value, index) => {
        const window = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidate = values[index + offset];
            if (Number.isFinite(candidate)) {
                window.push(candidate);
            }
        }
        return window.length > 0 ? average(window) : values[index];
    });
}

function applyMedianFilter(values, windowSize) {
    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    return values.map((value, index) => {
        const window = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidate = values[index + offset];
            if (Number.isFinite(candidate)) {
                window.push(candidate);
            }
        }
        if (window.length === 0) {
            return value;
        }
        const sorted = [...window].sort((left, right) => left - right);
        const middle = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle];
    });
}

function applyWienerLikeFilter(values, windowSize) {
    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const localStats = values.map((_value, index) => {
        const window = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidate = values[index + offset];
            if (Number.isFinite(candidate)) {
                window.push(candidate);
            }
        }
        const mean = window.length > 0 ? average(window) : values[index];
        const variance = window.length > 1
            ? average(window.map((candidate) => (candidate - mean) ** 2))
            : 0;
        return { mean, variance };
    });
    const noiseFloor = median(localStats.map((item) => item.variance));

    return values.map((value, index) => {
        const { mean, variance } = localStats[index];
        if (!Number.isFinite(value) || variance < 1e-10) {
            return mean;
        }
        const gain = Math.max(0, variance - noiseFloor) / Math.max(variance, 1e-8);
        return mean + gain * (value - mean);
    });
}

function applyGaussianFilter(values, windowSize) {
    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const sigma = Math.max(0.8, radius / 2);
    const kernel = Array.from({ length: radius * 2 + 1 }, (_value, index) => {
        const x = index - radius;
        return Math.exp(-(x ** 2) / (2 * sigma ** 2));
    });
    const kernelSum = kernel.reduce((sum, value) => sum + value, 0);
    return values.map((value, index) => {
        const window = [];
        const weights = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidate = values[index + offset];
            if (Number.isFinite(candidate)) {
                window.push(candidate);
                weights.push(kernel[offset + radius]);
            }
        }
        if (window.length === 0) {
            return value;
        }
        const weightedSum = window.reduce((sum, candidate, idx) => sum + candidate * weights[idx], 0);
        const weightTotal = weights.reduce((sum, w) => sum + w, 0);
        return weightTotal > 0 ? weightedSum / weightTotal : value;
    });
}

function applyEma(values, windowSize) {
    const alpha = 2 / (Math.max(2, windowSize) + 1);
    let level = values.length > 0 ? values[0] : 0;
    return values.map((value) => {
        if (!Number.isFinite(value)) {
            return level;
        }
        level = alpha * value + (1 - alpha) * level;
        return level;
    });
}

function tricubeWeight(distance) {
    if (distance >= 1) {
        return 0;
    }
    const value = 1 - distance ** 3;
    return value ** 3;
}

function applyLowessApproximation(values, fraction) {
    const count = values.length;
    if (count < 5) {
        return values.map((value) => (Number.isFinite(value) ? value : 0));
    }

    const span = clamp(Math.round(count * fraction), 5, Math.max(5, count - (count + 1) % 2));
    const radius = Math.max(2, Math.floor(span / 2));

    const regress = (weights) => values.map((value, index) => {
        const xs = [];
        const ys = [];
        const ws = [];

        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidateIndex = index + offset;
            const candidateValue = values[candidateIndex];
            if (!Number.isFinite(candidateValue)) {
                continue;
            }
            const spanWeight = tricubeWeight(Math.abs(offset) / Math.max(radius, 1));
            xs.push(offset);
            ys.push(candidateValue);
            ws.push(spanWeight * weights[candidateIndex]);
        }

        if (xs.length < 3) {
            return value;
        }

        const sumWeights = ws.reduce((total, weight) => total + weight, 0);
        const xMean = xs.reduce((total, x, idx) => total + x * ws[idx], 0) / Math.max(sumWeights, 1e-8);
        const yMean = ys.reduce((total, y, idx) => total + y * ws[idx], 0) / Math.max(sumWeights, 1e-8);
        const numerator = xs.reduce((total, x, idx) => total + ws[idx] * (x - xMean) * (ys[idx] - yMean), 0);
        const denominator = xs.reduce((total, x, idx) => total + ws[idx] * (x - xMean) ** 2, 0);
        const slope = denominator > 1e-10 ? numerator / denominator : 0;
        return yMean - slope * xMean;
    });

    const initialPass = regress(Array.from({ length: count }, () => 1));
    const residuals = values.map((value, index) => {
        if (!Number.isFinite(value) || !Number.isFinite(initialPass[index])) {
            return 0;
        }
        return Math.abs(value - initialPass[index]);
    });
    const scale = Math.max(medianAbsoluteDeviation(residuals), 1e-8) * 6;
    const robustWeights = residuals.map((residual) => 1 / (1 + (residual / scale) ** 2));

    return regress(robustWeights);
}

function applySavGol(values, windowSize, polyorder) {
    const effectiveWindow = ensureOddWindow(windowSize, 5);
    const radius = Math.floor(effectiveWindow / 2);
    return values.map((value, index) => {
        const xs = [];
        const ys = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const sample = values[index + offset];
            if (Number.isFinite(sample)) {
                xs.push(offset);
                ys.push(sample);
            }
        }

        if (xs.length <= 2) {
            return value;
        }

        const fitted = fitPolynomial(xs, ys, Math.min(polyorder, xs.length - 1));
        return fitted ? evaluatePolynomial(fitted, 0) : value;
    });
}

export function buildFilterPreview(data, settings) {
    const points = (data ?? []).filter((point) => Number.isFinite(point?.value));
    if (points.length === 0) {
        return {
            available: false,
            chartData: [],
            rawStd: 0,
            outputStd: 0,
            filterWindow: ensureOddWindow(settings.filterWindow ?? 9),
            filterPolyorder: Math.min(settings.filterPolyorder ?? 2, Math.max(1, ensureOddWindow(settings.filterWindow ?? 9) - 2)),
        };
    }

    const values = points.map((point) => point.value);
    const filterWindow = ensureOddWindow(settings.filterWindow ?? 9, 5);
    const filterPolyorder = clamp(settings.filterPolyorder ?? 2, 1, Math.max(1, filterWindow - 2));
    const lowessFraction = clamp(settings.lowessFraction ?? 0.12, 0.04, 0.45);
    const filterEnabled = settings.filterMethod !== "none";
    const detrendEnabled = settings.detrenderMethod === "polynomial";
    const filtered = !filterEnabled
        ? values
        : settings.filterMethod === "moving_average"
            ? applyMovingAverage(values, filterWindow)
            : settings.filterMethod === "median"
                ? applyMedianFilter(values, filterWindow)
                : settings.filterMethod === "wiener"
                    ? applyWienerLikeFilter(values, filterWindow)
                    : settings.filterMethod === "gaussian"
                        ? applyGaussianFilter(values, filterWindow)
                        : settings.filterMethod === "ema"
                            ? applyEma(values, filterWindow)
                            : settings.filterMethod === "lowess"
                                ? applyLowessApproximation(values, lowessFraction)
                                : applySavGol(values, filterWindow, filterPolyorder);
    const trend = detrendEnabled
        ? buildPolynomialTrend(filtered, clamp(settings.detrenderOrder ?? 2, 1, 6))
        : filtered.map(() => 0);
    const output = filtered.map((value, index) => value - trend[index]);
    const rawStd = standardDeviation(values, average(values));
    const outputStd = standardDeviation(output, average(output));

    return {
        available: true,
        rawStd,
        outputStd,
        filterWindow,
        filterPolyorder,
        lowessFraction,
        chartData: points.map((point, index) => ({
            label: point.month ?? point.timestamp ?? `T${index + 1}`,
            timestamp: point.timestamp ?? `T${index + 1}`,
            raw: Number(values[index].toFixed(4)),
            filtered: Number(filtered[index].toFixed(4)),
            trend: detrendEnabled ? Number(trend[index].toFixed(4)) : null,
            output: Number(output[index].toFixed(4)),
        })),
    };
}
