function average(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
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

    const position = (values.length - 1) * ratio;
    const lowerIndex = Math.floor(position);
    const upperIndex = Math.ceil(position);
    if (lowerIndex === upperIndex) {
        return values[lowerIndex];
    }

    const weight = position - lowerIndex;
    return values[lowerIndex] * (1 - weight) + values[upperIndex] * weight;
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

export function computeModifiedZScores(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length < 2) {
        return values.map(() => 0);
    }

    const med = median(finite);
    const mad = Math.max(medianAbsoluteDeviation(finite), 1e-8);
    return values.map((value) =>
        Number.isFinite(value) ? (0.6745 * Math.abs(value - med)) / mad : 0,
    );
}

function formatMetric(value, digits = 2) {
    return Number(value).toFixed(digits);
}

function standardDeviation(values, mean) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length < 2) {
        return 0;
    }
    const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
    return Math.sqrt(variance);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function chooseContamination(count) {
    if (count >= 200) {
        return 0.05;
    }
    if (count >= 80) {
        return 0.08;
    }
    return 0.12;
}

function thresholdForContamination(scores, contamination, minThreshold = 0) {
    const finite = scores.filter((value) => Number.isFinite(value));
    if (finite.length === 0) {
        return minThreshold;
    }
    const sorted = [...finite].sort((a, b) => a - b);
    const index = Math.max(0, Math.min(sorted.length - 1, Math.floor((1 - contamination) * (sorted.length - 1))));
    return Math.max(sorted[index], minThreshold);
}

function chooseLofNeighborCount(count) {
    return Math.max(2, Math.min(8, Math.floor(Math.sqrt(count))));
}

function labelOutlierResults(points, rows) {
    const flaggedIndices = new Set(rows.map((row) => row.index));
    return {
        flaggedCount: rows.length,
        retainedCount: points.length - rows.length,
        flaggedRate: points.length > 0 ? rows.length / points.length : 0,
        chartData: points.map((point) => ({
            ...point,
            outlierValue: flaggedIndices.has(point.index) ? point.value : null,
        })),
        rows: rows.slice(0, 12),
    };
}

export function buildOutlierPreview(data, method) {
    const points = (data ?? [])
        .filter((point) => Number.isFinite(point?.value))
        .map((point, index) => ({
            index: point.t ?? index,
            timestamp: point.timestamp ?? `T${index + 1}`,
            label: point.month ?? point.timestamp ?? `T${index + 1}`,
            value: point.value,
            outlierValue: null,
        }));

    const basePreview = {
        method,
        total: points.length,
        flaggedCount: 0,
        retainedCount: points.length,
        flaggedRate: 0,
        thresholdLabel: "Detection disabled.",
        chartData: points,
        rows: [],
    };

    if (points.length === 0 || method === "none") {
        return basePreview;
    }

    if (method === "zscore") {
        const values = points.map((point) => point.value);
        const mean = average(values);
        const std = standardDeviation(values, mean);

        if (!Number.isFinite(std) || std === 0) {
            return {
                ...basePreview,
                thresholdLabel: "Z-score needs non-constant data.",
            };
        }

        const threshold = 3.0;
        const rows = points
            .map((point) => {
                const zScore = (point.value - mean) / std;
                return {
                    ...point,
                    score: Math.abs(zScore),
                    reason: `|z| = ${formatMetric(Math.abs(zScore), 2)}`,
                };
            })
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `Flagging values with |z| >= ${threshold.toFixed(2)}.`,
        };
    }

    if (method === "mad") {
        const values = points.map((point) => point.value);
        const scores = computeModifiedZScores(values);
        const threshold = 3.5;
        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `MAD score = ${formatMetric(scores[index], 2)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `MAD detection (modified z-score >= ${threshold.toFixed(2)}).`,
        };
    }

    if (method === "isolation_forest") {
        const values = points.map((point) => point.value);
        const forestTreeCount = Math.max(25, Math.min(80, Math.floor(points.length / 4)));
        const forestSampleSize = Math.min(64, Math.max(16, Math.floor(points.length / 4)));
        const scores = computeIsolationForestScores(values, forestTreeCount, forestSampleSize);
        const contamination = chooseContamination(points.length);
        const threshold = thresholdForContamination(scores, contamination, 0.6);
        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `Isolation score = ${formatMetric(scores[index], 2)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `Isolation Forest threshold ${formatMetric(threshold, 2)} (${Math.round(contamination * 100)}% contamination).`,
        };
    }

    if (method === "lof") {
        const values = points.map((point) => point.value);
        const scores = computeLofScores(values, chooseLofNeighborCount(points.length));
        const contamination = chooseContamination(points.length);
        const threshold = Math.max(thresholdForContamination(scores, contamination, 1.2), 1.2);
        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `LOF score = ${formatMetric(scores[index], 2)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `LOF threshold ${formatMetric(threshold, 2)} (${Math.round(contamination * 100)}% contamination).`,
        };
    }

    const values = points.map((point) => point.value);
    const sortedValues = [...values].sort((left, right) => left - right);
    const q1 = quantile(sortedValues, 0.25);
    const q3 = quantile(sortedValues, 0.75);
    const iqr = q3 - q1;
    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;
    const rows = points
        .map((point) => {
            if (point.value < lowerFence) {
                return {
                    ...point,
                    score: lowerFence - point.value,
                    reason: `< lower fence (${formatMetric(lowerFence, 2)})`,
                };
            }

            if (point.value > upperFence) {
                return {
                    ...point,
                    score: point.value - upperFence,
                    reason: `> upper fence (${formatMetric(upperFence, 2)})`,
                };
            }

            return null;
        })
        .filter(Boolean)
        .sort((left, right) => right.score - left.score);

    return {
        ...basePreview,
        ...labelOutlierResults(points, rows),
        thresholdLabel: `IQR fences: [${formatMetric(lowerFence, 2)}, ${formatMetric(upperFence, 2)}].`,
    };
}

export function averagePathLength(sampleSize) {
    if (sampleSize <= 1) {
        return 0;
    }
    let harmonic = 0;
    for (let i = 1; i < sampleSize; i += 1) {
        harmonic += 1 / i;
    }
    return 2 * harmonic - 2 * (sampleSize - 1) / sampleSize;
}

export function sampleIndices(n, sampleSize) {
    const indices = Array.from({ length: n }, (_value, index) => index);
    const limit = Math.min(sampleSize, n);
    for (let i = 0; i < limit; i += 1) {
        const swapIndex = i + Math.floor(Math.random() * (n - i));
        [indices[i], indices[swapIndex]] = [indices[swapIndex], indices[i]];
    }
    return indices.slice(0, limit);
}

function buildIsolationTree(sample, currentDepth = 0) {
    if (sample.length <= 1) {
        return {
            external: true,
            size: sample.length,
            depth: currentDepth,
        };
    }

    const minValue = Math.min(...sample);
    const maxValue = Math.max(...sample);
    if (minValue === maxValue) {
        return {
            external: true,
            size: sample.length,
            depth: currentDepth,
        };
    }

    let split;
    let left;
    let right;
    let attempts = 0;
    do {
        split = minValue + Math.random() * (maxValue - minValue);
        left = sample.filter((candidate) => candidate < split);
        right = sample.filter((candidate) => candidate >= split);
        attempts += 1;
    } while ((left.length === 0 || right.length === 0) && attempts < 10);

    if (left.length === 0 || right.length === 0) {
        return {
            external: true,
            size: sample.length,
            depth: currentDepth,
        };
    }

    return {
        external: false,
        split,
        left: buildIsolationTree(left, currentDepth + 1),
        right: buildIsolationTree(right, currentDepth + 1),
    };
}

function isolationTreePathLength(value, node) {
    if (node.external) {
        return node.depth + averagePathLength(node.size);
    }
    return value < node.split
        ? isolationTreePathLength(value, node.left)
        : isolationTreePathLength(value, node.right);
}

export function computeIsolationForestScores(values, treeCount = 50, sampleSize = 64) {
    const n = values.length;
    if (n === 0) {
        return [];
    }

    const depthSums = Array.from({ length: n }, () => 0);
    const sampleCount = Math.min(sampleSize, n);
    const c = averagePathLength(sampleCount) || 1;

    for (let tree = 0; tree < treeCount; tree += 1) {
        const sample = sampleIndices(n, sampleCount).map((index) => values[index]);
        const treeRoot = buildIsolationTree(sample);
        for (let index = 0; index < n; index += 1) {
            depthSums[index] += isolationTreePathLength(values[index], treeRoot);
        }
    }

    return depthSums.map((sum) => 2 ** (-(sum / treeCount) / c));
}

export function computeLofScores(values, neighborCount = 8) {
    const n = values.length;
    if (n < 3) {
        return values.map(() => 0);
    }

    const k = Math.min(neighborCount, n - 1);
    const distances = values.map((value, index) =>
        values.map((other, otherIndex) =>
            index === otherIndex ? Infinity : Math.abs(other - value),
        ),
    );

    const kDistances = distances.map((row) => [...row].sort((a, b) => a - b)[k - 1]);
    const lrd = distances.map((row, index) => {
        const nearest = row
            .map((distance, otherIndex) => ({ distance, otherIndex }))
            .sort((left, right) => left.distance - right.distance)
            .slice(0, k);
        const reachSum = nearest.reduce(
            (sum, neighbor) => sum + Math.max(kDistances[neighbor.otherIndex], neighbor.distance),
            0,
        );
        return reachSum > 0 ? k / reachSum : 0;
    });

    return distances.map((row, index) => {
        const ownLrd = Math.max(lrd[index], 1e-8);
        const nearest = row
            .map((distance, otherIndex) => ({ distance, otherIndex }))
            .sort((left, right) => left.distance - right.distance)
            .slice(0, k);
        const score = nearest.reduce(
            (sum, neighbor) => sum + lrd[neighbor.otherIndex] / ownLrd,
            0,
        );
        return score / k;
    });
}
