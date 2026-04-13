function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function isFiniteNumber(value) {
    return Number.isFinite(value);
}

function ensureOddWindow(value, minimum = 3) {
    const normalized = Math.max(minimum, Math.round(isFiniteNumber(value) ? value : minimum));
    return normalized % 2 === 1 ? normalized : normalized + 1;
}

function average(values) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (isFiniteNumber(value)) {
            sum += value;
            count += 1;
        }
    }
    return count > 0 ? sum / count : 0;
}

function variance(values, mean = average(values)) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (isFiniteNumber(value)) {
            const diff = value - mean;
            sum += diff * diff;
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
        const value = values[i];
        if (isFiniteNumber(value)) {
            finite.push(value);
        }
    }

    if (finite.length === 0) {
        return 0;
    }

    finite.sort((left, right) => left - right);
    const middle = Math.floor(finite.length / 2);
    return finite.length % 2 === 0
        ? (finite[middle - 1] + finite[middle]) / 2
        : finite[middle];
}

function medianAbsoluteDeviation(values) {
    const med = median(values);
    const deviations = new Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        deviations[i] = isFiniteNumber(value) ? Math.abs(value - med) : null;
    }
    return median(deviations);
}

function round4(value) {
    return Number.isFinite(value) ? Number(value.toFixed(4)) : 0;
}

export function solveLinearSystem(matrix, vector) {
    const size = matrix.length;
    const augmented = matrix.map((row, rowIndex) => [...row, vector[rowIndex]]);

    for (let pivot = 0; pivot < size; pivot += 1) {
        let pivotRow = pivot;
        for (let row = pivot + 1; row < size; row += 1) {
            if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[pivotRow][pivot])) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][pivot]) < 1e-12) {
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

function fitPolynomial(xs, ys, order, weights = null) {
    if (xs.length === 0 || ys.length === 0) {
        return null;
    }

    const degree = Math.min(order, xs.length - 1);
    const size = degree + 1;
    const system = Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
    const rhs = Array.from({ length: size }, () => 0);

    for (let i = 0; i < xs.length; i += 1) {
        const x = xs[i];
        const y = ys[i];
        const w = weights ? weights[i] : 1;

        if (!isFiniteNumber(x) || !isFiniteNumber(y) || !isFiniteNumber(w) || w <= 0) {
            continue;
        }

        const powers = new Array(size * 2).fill(1);
        for (let p = 1; p < powers.length; p += 1) {
            powers[p] = powers[p - 1] * x;
        }

        for (let row = 0; row < size; row += 1) {
            rhs[row] += w * y * powers[row];
            for (let column = 0; column < size; column += 1) {
                system[row][column] += w * powers[row + column];
            }
        }
    }

    return solveLinearSystem(system, rhs);
}

function evaluatePolynomial(coefficients, x) {
    if (!coefficients) {
        return 0;
    }
    let total = 0;
    let powerValue = 1;
    for (let i = 0; i < coefficients.length; i += 1) {
        total += coefficients[i] * powerValue;
        powerValue *= x;
    }
    return total;
}

function buildPolynomialTrend(values, order) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const xs = new Array(n);
    for (let i = 0; i < n; i += 1) {
        xs[i] = n <= 1 ? 0 : (i / (n - 1)) * 2 - 1;
    }

    const finiteXs = [];
    const finiteYs = [];
    for (let i = 0; i < n; i += 1) {
        if (isFiniteNumber(values[i])) {
            finiteXs.push(xs[i]);
            finiteYs.push(values[i]);
        }
    }

    if (finiteXs.length < order + 1) {
        return values.map(() => 0);
    }

    const coefficients = fitPolynomial(finiteXs, finiteYs, order);
    return xs.map((x) => evaluatePolynomial(coefficients, x));
}

function buildPrefixSums(values) {
    const n = values.length;
    const prefixSum = new Float64Array(n + 1);
    const prefixCount = new Float64Array(n + 1);
    const prefixSqSum = new Float64Array(n + 1);

    for (let i = 0; i < n; i += 1) {
        const value = values[i];
        const valid = isFiniteNumber(value);
        prefixSum[i + 1] = prefixSum[i] + (valid ? value : 0);
        prefixSqSum[i + 1] = prefixSqSum[i] + (valid ? value * value : 0);
        prefixCount[i + 1] = prefixCount[i] + (valid ? 1 : 0);
    }

    return { prefixSum, prefixSqSum, prefixCount };
}

function rangeStats(prefix, start, end) {
    const sum = prefix.prefixSum[end] - prefix.prefixSum[start];
    const sqSum = prefix.prefixSqSum[end] - prefix.prefixSqSum[start];
    const count = prefix.prefixCount[end] - prefix.prefixCount[start];
    const mean = count > 0 ? sum / count : 0;
    const varianceValue = count > 0 ? Math.max(0, sqSum / count - mean * mean) : 0;
    return { sum, sqSum, count, mean, variance: varianceValue };
}

function applyMovingAverage(values, windowSize) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const prefix = buildPrefixSums(values);
    const result = new Array(n);

    for (let index = 0; index < n; index += 1) {
        const start = Math.max(0, index - radius);
        const end = Math.min(n, index + radius + 1);
        const { mean, count } = rangeStats(prefix, start, end);
        result[index] = count > 0 ? mean : (isFiniteNumber(values[index]) ? values[index] : 0);
    }

    return result;
}

function applyMedianFilter(values, windowSize) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const result = new Array(n);

    for (let index = 0; index < n; index += 1) {
        const window = [];
        const start = Math.max(0, index - radius);
        const end = Math.min(n, index + radius + 1);

        for (let i = start; i < end; i += 1) {
            const candidate = values[i];
            if (isFiniteNumber(candidate)) {
                window.push(candidate);
            }
        }

        result[index] = window.length > 0
            ? median(window)
            : (isFiniteNumber(values[index]) ? values[index] : 0);
    }

    return result;
}

function applyWienerLikeFilter(values, windowSize) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const prefix = buildPrefixSums(values);
    const means = new Array(n);
    const variances = new Array(n);

    for (let index = 0; index < n; index += 1) {
        const start = Math.max(0, index - radius);
        const end = Math.min(n, index + radius + 1);
        const stats = rangeStats(prefix, start, end);
        means[index] = stats.mean;
        variances[index] = stats.variance;
    }

    const noiseFloor = median(variances);
    const result = new Array(n);

    for (let index = 0; index < n; index += 1) {
        const value = values[index];
        const localMean = means[index];
        const localVariance = variances[index];

        if (!isFiniteNumber(value) || localVariance < 1e-12) {
            result[index] = localMean;
            continue;
        }

        const gain = Math.max(0, localVariance - noiseFloor) / Math.max(localVariance, 1e-12);
        result[index] = localMean + gain * (value - localMean);
    }

    return result;
}

function buildGaussianKernel(windowSize) {
    const radius = Math.floor(ensureOddWindow(windowSize) / 2);
    const sigma = Math.max(0.8, radius / 2);
    const kernel = new Array(radius * 2 + 1);
    let kernelSum = 0;

    for (let i = -radius; i <= radius; i += 1) {
        const value = Math.exp(-(i * i) / (2 * sigma * sigma));
        kernel[i + radius] = value;
        kernelSum += value;
    }

    for (let i = 0; i < kernel.length; i += 1) {
        kernel[i] /= kernelSum;
    }

    return { kernel, radius };
}

function applyGaussianFilter(values, windowSize) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const { kernel, radius } = buildGaussianKernel(windowSize);
    const result = new Array(n);

    for (let index = 0; index < n; index += 1) {
        let weightedSum = 0;
        let weightTotal = 0;

        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidateIndex = index + offset;
            if (candidateIndex < 0 || candidateIndex >= n) {
                continue;
            }

            const candidate = values[candidateIndex];
            if (!isFiniteNumber(candidate)) {
                continue;
            }

            const weight = kernel[offset + radius];
            weightedSum += candidate * weight;
            weightTotal += weight;
        }

        result[index] = weightTotal > 1e-12
            ? weightedSum / weightTotal
            : (isFiniteNumber(values[index]) ? values[index] : 0);
    }

    return result;
}

function applyEma(values, windowSize) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const alpha = 2 / (Math.max(2, windowSize) + 1);
    const result = new Array(n);

    let level = 0;
    let initialized = false;

    for (let i = 0; i < n; i += 1) {
        const value = values[i];
        if (!initialized && isFiniteNumber(value)) {
            level = value;
            initialized = true;
        }

        if (!initialized) {
            result[i] = 0;
            continue;
        }

        if (isFiniteNumber(value)) {
            level = alpha * value + (1 - alpha) * level;
        }
        result[i] = level;
    }

    return result;
}

function tricubeWeight(distance) {
    if (distance >= 1) {
        return 0;
    }
    const value = 1 - distance ** 3;
    return value ** 3;
}

function bisquareWeight(u) {
    if (u >= 1) {
        return 0;
    }
    const value = 1 - u * u;
    return value * value;
}

function applyLowessApproximation(values, fraction) {
    const count = values.length;
    if (count < 5) {
        return values.map((value) => (isFiniteNumber(value) ? value : 0));
    }

    const span = clamp(Math.round(count * fraction), 5, count);
    const radius = Math.max(2, Math.floor(span / 2));

    function regress(robustWeights) {
        const output = new Array(count);

        for (let index = 0; index < count; index += 1) {
            let sumW = 0;
            let sumWX = 0;
            let sumWY = 0;
            let sumWXX = 0;
            let sumWXY = 0;

            for (let offset = -radius; offset <= radius; offset += 1) {
                const candidateIndex = index + offset;
                if (candidateIndex < 0 || candidateIndex >= count) {
                    continue;
                }

                const y = values[candidateIndex];
                if (!isFiniteNumber(y)) {
                    continue;
                }

                const x = offset;
                const spanWeight = tricubeWeight(Math.abs(offset) / Math.max(radius, 1));
                const robustWeight = robustWeights ? robustWeights[candidateIndex] : 1;
                const w = spanWeight * robustWeight;

                if (w <= 0) {
                    continue;
                }

                sumW += w;
                sumWX += w * x;
                sumWY += w * y;
                sumWXX += w * x * x;
                sumWXY += w * x * y;
            }

            if (sumW < 1e-12) {
                output[index] = isFiniteNumber(values[index]) ? values[index] : 0;
                continue;
            }

            const denom = sumW * sumWXX - sumWX * sumWX;
            const intercept = Math.abs(denom) > 1e-12
                ? (sumWY * sumWXX - sumWX * sumWXY) / denom
                : sumWY / sumW;

            output[index] = intercept;
        }

        return output;
    }

    const firstPass = regress(null);
    const residuals = new Array(count);
    for (let i = 0; i < count; i += 1) {
        const value = values[i];
        residuals[i] = isFiniteNumber(value) && isFiniteNumber(firstPass[i])
            ? Math.abs(value - firstPass[i])
            : 0;
    }

    const scale = Math.max(medianAbsoluteDeviation(residuals) * 6, 1e-8);
    const robustWeights = residuals.map((residual) => bisquareWeight(residual / scale));

    return regress(robustWeights);
}

function invertMatrix(matrix) {
    const n = matrix.length;
    const augmented = matrix.map((row, i) => [
        ...row,
        ...Array.from({ length: n }, (_v, j) => (i === j ? 1 : 0)),
    ]);

    for (let pivot = 0; pivot < n; pivot += 1) {
        let pivotRow = pivot;
        for (let row = pivot + 1; row < n; row += 1) {
            if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[pivotRow][pivot])) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][pivot]) < 1e-12) {
            return null;
        }

        if (pivotRow !== pivot) {
            [augmented[pivot], augmented[pivotRow]] = [augmented[pivotRow], augmented[pivot]];
        }

        const pivotValue = augmented[pivot][pivot];
        for (let col = 0; col < 2 * n; col += 1) {
            augmented[pivot][col] /= pivotValue;
        }

        for (let row = 0; row < n; row += 1) {
            if (row === pivot) {
                continue;
            }
            const factor = augmented[row][pivot];
            for (let col = 0; col < 2 * n; col += 1) {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }

    return augmented.map((row) => row.slice(n));
}

function computeSavGolKernel(windowSize, polyorder) {
    const effectiveWindow = ensureOddWindow(windowSize, 5);
    const radius = Math.floor(effectiveWindow / 2);
    const degree = Math.min(polyorder, effectiveWindow - 1);
    const cols = degree + 1;

    const xtx = Array.from({ length: cols }, () => Array.from({ length: cols }, () => 0));
    const xt = Array.from({ length: cols }, () => Array.from({ length: effectiveWindow }, () => 0));

    for (let i = -radius; i <= radius; i += 1) {
        const row = i + radius;
        let power = 1;
        for (let col = 0; col < cols; col += 1) {
            xt[col][row] = power;
            power *= i;
        }
    }

    for (let row = 0; row < cols; row += 1) {
        for (let col = 0; col < cols; col += 1) {
            let sum = 0;
            for (let k = 0; k < effectiveWindow; k += 1) {
                sum += xt[row][k] * xt[col][k];
            }
            xtx[row][col] = sum;
        }
    }

    const xtxInv = invertMatrix(xtx);
    if (!xtxInv) {
        return null;
    }

    const coeffs = new Array(effectiveWindow).fill(0);
    for (let k = 0; k < effectiveWindow; k += 1) {
        let sum = 0;
        for (let j = 0; j < cols; j += 1) {
            sum += xtxInv[0][j] * xt[j][k];
        }
        coeffs[k] = sum;
    }

    return { coeffs, radius, effectiveWindow };
}

function applySavGol(values, windowSize, polyorder) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const kernelInfo = computeSavGolKernel(windowSize, polyorder);
    if (!kernelInfo) {
        return applyMovingAverage(values, windowSize);
    }

    const { coeffs, radius, effectiveWindow } = kernelInfo;
    const result = new Array(n);

    for (let index = 0; index < n; index += 1) {
        if (index >= radius && index < n - radius) {
            let sum = 0;
            let weightSum = 0;

            for (let offset = -radius; offset <= radius; offset += 1) {
                const value = values[index + offset];
                if (!isFiniteNumber(value)) {
                    continue;
                }
                const weight = coeffs[offset + radius];
                sum += weight * value;
                weightSum += weight;
            }

            result[index] = Math.abs(weightSum) > 1e-12
                ? sum / weightSum
                : (isFiniteNumber(values[index]) ? values[index] : 0);
            continue;
        }

        const xs = [];
        const ys = [];
        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidateIndex = index + offset;
            if (candidateIndex < 0 || candidateIndex >= n) {
                continue;
            }
            const sample = values[candidateIndex];
            if (isFiniteNumber(sample)) {
                xs.push(offset);
                ys.push(sample);
            }
        }

        if (xs.length <= 2) {
            result[index] = isFiniteNumber(values[index]) ? values[index] : 0;
            continue;
        }

        const fitted = fitPolynomial(xs, ys, Math.min(polyorder, xs.length - 1));
        result[index] = fitted ? evaluatePolynomial(fitted, 0) : (isFiniteNumber(values[index]) ? values[index] : 0);
    }

    return result;
}

export function buildFilterPreview(data, settings) {
    const points = (data ?? []).filter((point) => isFiniteNumber(point?.value));

    if (points.length === 0) {
        return {
            available: false,
            chartData: [],
            rawStd: 0,
            outputStd: 0,
            filterWindow: ensureOddWindow(settings?.filterWindow ?? 9),
            filterPolyorder: Math.min(
                settings?.filterPolyorder ?? 2,
                Math.max(1, ensureOddWindow(settings?.filterWindow ?? 9) - 2),
            ),
        };
    }

    const values = points.map((point) => point.value);
    const filterWindow = ensureOddWindow(settings?.filterWindow ?? 9, 5);
    const filterPolyorder = clamp(settings?.filterPolyorder ?? 2, 1, Math.max(1, filterWindow - 2));
    const lowessFraction = clamp(settings?.lowessFraction ?? 0.12, 0.04, 0.45);

    const filterEnabled = settings?.filterMethod !== "none";
    const detrendEnabled = settings?.detrenderMethod === "polynomial";

    let filtered;
    if (!filterEnabled) {
        filtered = values.slice();
    } else if (settings.filterMethod === "moving_average") {
        filtered = applyMovingAverage(values, filterWindow);
    } else if (settings.filterMethod === "median") {
        filtered = applyMedianFilter(values, filterWindow);
    } else if (settings.filterMethod === "wiener") {
        filtered = applyWienerLikeFilter(values, filterWindow);
    } else if (settings.filterMethod === "gaussian") {
        filtered = applyGaussianFilter(values, filterWindow);
    } else if (settings.filterMethod === "ema") {
        filtered = applyEma(values, filterWindow);
    } else if (settings.filterMethod === "lowess") {
        filtered = applyLowessApproximation(values, lowessFraction);
    } else {
        filtered = applySavGol(values, filterWindow, filterPolyorder);
    }

    const trend = detrendEnabled
        ? buildPolynomialTrend(filtered, clamp(settings?.detrenderOrder ?? 2, 1, 6))
        : filtered.map(() => 0);

    const output = filtered.map((value, index) => value - trend[index]);
    const rawStd = standardDeviation(values);
    const outputStd = standardDeviation(output);

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
            raw: round4(values[index]),
            filtered: round4(filtered[index]),
            trend: detrendEnabled ? round4(trend[index]) : null,
            output: round4(output[index]),
        })),
    };
}
