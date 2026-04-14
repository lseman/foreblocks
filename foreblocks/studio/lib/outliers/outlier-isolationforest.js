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
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length < 2) {
        return 0;
    }

    const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
    return Math.sqrt(variance);
}

function averagePathLength(sampleSize) {
    const n = Math.floor(sampleSize);
    if (n <= 1) {
        return 0;
    }

    let harmonic = 0;
    for (let i = 1; i < n; i += 1) {
        harmonic += 1 / i;
    }

    return 2 * harmonic - 2 * (n - 1) / n;
}

function sampleIndices(n, sampleSize) {
    const indices = Array.from({ length: n }, (_value, index) => index);
    const limit = Math.min(sampleSize, n);

    for (let i = 0; i < limit; i += 1) {
        const swapIndex = i + Math.floor(Math.random() * (n - i));
        [indices[i], indices[swapIndex]] = [indices[swapIndex], indices[i]];
    }

    return indices.slice(0, limit);
}

function maxIsolationTreeDepth(sampleSize) {
    return sampleSize > 1 ? Math.ceil(Math.log2(sampleSize)) : 0;
}

function getFeatureValue(point, axis) {
    return point[axis];
}

function computeFeatureRange(points, indices, start, end, axis) {
    let minValue = Infinity;
    let maxValue = -Infinity;

    for (let i = start; i < end; i += 1) {
        const value = getFeatureValue(points[indices[i]], axis);
        if (value < minValue) {
            minValue = value;
        }
        if (value > maxValue) {
            maxValue = value;
        }
    }

    return { minValue, maxValue };
}

function chooseSplitAxis(points, indices, start, end) {
    const dimension = points[0]?.length ?? 1;
    const candidateAxes = [];

    for (let axis = 0; axis < dimension; axis += 1) {
        const { minValue, maxValue } = computeFeatureRange(points, indices, start, end, axis);
        if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {
            candidateAxes.push({ axis, span: maxValue - minValue, minValue, maxValue });
        }
    }

    if (candidateAxes.length === 0) {
        return null;
    }

    const totalSpan = candidateAxes.reduce((sum, item) => sum + item.span, 0);
    if (totalSpan <= 0) {
        return candidateAxes[Math.floor(Math.random() * candidateAxes.length)];
    }

    let draw = Math.random() * totalSpan;
    for (const item of candidateAxes) {
        draw -= item.span;
        if (draw <= 0) {
            return item;
        }
    }

    return candidateAxes[candidateAxes.length - 1];
}

function partitionAroundSplit(points, indices, start, end, axis, split) {
    let left = start;
    let right = end - 1;

    while (left <= right) {
        while (left <= right && getFeatureValue(points[indices[left]], axis) < split) {
            left += 1;
        }

        while (left <= right && getFeatureValue(points[indices[right]], axis) >= split) {
            right -= 1;
        }

        if (left < right) {
            [indices[left], indices[right]] = [indices[right], indices[left]];
            left += 1;
            right -= 1;
        }
    }

    return left;
}

function buildIsolationTree(points, indices, start, end, depth, depthLimit) {
    const size = end - start;
    if (size <= 1 || depth >= depthLimit) {
        return {
            external: true,
            size,
            depth,
        };
    }

    const axisInfo = chooseSplitAxis(points, indices, start, end);
    if (!axisInfo) {
        return {
            external: true,
            size,
            depth,
        };
    }

    const { axis, minValue, maxValue } = axisInfo;
    let split = minValue + Math.random() * (maxValue - minValue);
    let mid = partitionAroundSplit(points, indices, start, end, axis, split);

    let attempts = 0;
    while ((mid === start || mid === end) && attempts < 8) {
        split = minValue + Math.random() * (maxValue - minValue);
        mid = partitionAroundSplit(points, indices, start, end, axis, split);
        attempts += 1;
    }

    if (mid === start || mid === end) {
        return {
            external: true,
            size,
            depth,
        };
    }

    return {
        external: false,
        axis,
        split,
        left: buildIsolationTree(points, indices, start, mid, depth + 1, depthLimit),
        right: buildIsolationTree(points, indices, mid, end, depth + 1, depthLimit),
    };
}

function isolationTreePathLength(point, node) {
    let current = node;

    while (!current.external) {
        current = point[current.axis] < current.split ? current.left : current.right;
    }

    return current.depth + averagePathLength(current.size);
}

function rollingWindowBounds(index, n, radius) {
    return {
        start: Math.max(0, index - radius),
        end: Math.min(n, index + radius + 1),
    };
}

function computeRollingMedian(values, radius = 3) {
    const n = values.length;
    const result = new Array(n);

    for (let i = 0; i < n; i += 1) {
        const { start, end } = rollingWindowBounds(i, n, radius);
        const window = [];
        for (let j = start; j < end; j += 1) {
            const value = values[j];
            if (Number.isFinite(value)) {
                window.push(value);
            }
        }
        result[i] = window.length > 0 ? median(window) : values[i];
    }

    return result;
}

function computeRollingMeanStd(values, radius = 5) {
    const n = values.length;
    const means = new Array(n).fill(0);
    const stds = new Array(n).fill(0);

    for (let i = 0; i < n; i += 1) {
        const { start, end } = rollingWindowBounds(i, n, radius);
        const window = [];
        for (let j = start; j < end; j += 1) {
            const value = values[j];
            if (Number.isFinite(value)) {
                window.push(value);
            }
        }

        if (window.length === 0) {
            means[i] = 0;
            stds[i] = 0;
            continue;
        }

        const mean = average(window);
        means[i] = mean;
        stds[i] = standardDeviation(window, mean);
    }

    return { means, stds };
}

function robustScaleVector(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length === 0) {
        return values.map(() => 0);
    }

    const med = median(finite);
    const mad = Math.max(medianAbsoluteDeviation(finite), 1e-8);

    return values.map((value) => (Number.isFinite(value) ? (value - med) / mad : 0));
}

function buildIsolationFeatures(values) {
    const n = values.length;
    const rollingMedian = computeRollingMedian(values, 3);
    const { means: rollingMeans, stds: rollingStds } = computeRollingMeanStd(values, 5);

    const raw = new Array(n).fill(0);
    const medianDeviation = new Array(n).fill(0);
    const diff1 = new Array(n).fill(0);
    const diff2 = new Array(n).fill(0);
    const localZ = new Array(n).fill(0);

    for (let i = 0; i < n; i += 1) {
        const value = values[i];
        const prev = i > 0 ? values[i - 1] : value;
        const prev2 = i > 1 ? values[i - 2] : prev;
        const d1 = value - prev;
        const prevD1 = prev - prev2;
        const d2 = d1 - prevD1;

        raw[i] = value;
        medianDeviation[i] = value - rollingMedian[i];
        diff1[i] = Number.isFinite(d1) ? d1 : 0;
        diff2[i] = Number.isFinite(d2) ? d2 : 0;

        const std = Math.max(rollingStds[i], 1e-8);
        localZ[i] = Number.isFinite(value) ? (value - rollingMeans[i]) / std : 0;
    }

    const scaledRaw = robustScaleVector(raw);
    const scaledMedianDeviation = robustScaleVector(medianDeviation);
    const scaledDiff1 = robustScaleVector(diff1);
    const scaledDiff2 = robustScaleVector(diff2);
    const scaledLocalZ = robustScaleVector(localZ);

    return Array.from({ length: n }, (_unused, index) => ([
        scaledRaw[index],
        scaledMedianDeviation[index],
        scaledDiff1[index],
        scaledDiff2[index],
        scaledLocalZ[index],
    ]));
}

export function computeIsolationForestScores(values, treeCount = 100, sampleSize = 64) {
    const finiteValues = [];
    const finitePositions = [];

    for (let i = 0; i < values.length; i += 1) {
        if (Number.isFinite(values[i])) {
            finiteValues.push(values[i]);
            finitePositions.push(i);
        }
    }

    const result = values.map(() => 0);
    const n = finiteValues.length;

    if (n === 0) {
        return result;
    }

    if (n === 1) {
        result[finitePositions[0]] = 0;
        return result;
    }

    const features = buildIsolationFeatures(finiteValues);
    const effectiveTreeCount = Math.max(1, Math.floor(treeCount));
    const effectiveSampleSize = Math.min(Math.max(2, Math.floor(sampleSize)), n);
    const normalizer = averagePathLength(effectiveSampleSize) || 1;
    const depthLimit = maxIsolationTreeDepth(effectiveSampleSize);
    const depthSums = new Array(n).fill(0);

    for (let tree = 0; tree < effectiveTreeCount; tree += 1) {
        const sampledIndices = sampleIndices(n, effectiveSampleSize);
        const working = sampledIndices.slice();
        const root = buildIsolationTree(
            features,
            working,
            0,
            working.length,
            0,
            depthLimit,
        );

        for (let i = 0; i < n; i += 1) {
            depthSums[i] += isolationTreePathLength(features[i], root);
        }
    }

    for (let i = 0; i < n; i += 1) {
        const avgDepth = depthSums[i] / effectiveTreeCount;
        result[finitePositions[i]] = 2 ** (-avgDepth / normalizer);
    }

    return result;
}
