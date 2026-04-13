import { computeIsolationForestScores } from "./outlier-isolationforest.js";
import { computeEcodScores } from "./outlier-ecod.js";
import { computeHbosScores } from "./outlier-hbos.js";
export { computeIsolationForestScores } from "./outlier-isolationforest.js";
export { computeEcodScores } from "./outlier-ecod.js";
export { computeHbosScores } from "./outlier-hbos.js";

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
    if (count >= 500) {
        return 0.03;
    }
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
    const index = Math.max(
        0,
        Math.min(sorted.length - 1, Math.floor((1 - contamination) * (sorted.length - 1))),
    );

    return Math.max(sorted[index], minThreshold);
}

function chooseLofNeighborCount(count) {
    return Math.max(2, Math.min(12, Math.floor(Math.sqrt(count))));
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

function chooseIsolationForestConfig(count) {
    return {
        treeCount: clamp(Math.round(Math.sqrt(Math.max(count, 1)) * 12), 64, 256),
        sampleSize: clamp(Math.round(Math.min(256, Math.max(32, count * 0.5))), 16, 256),
    };
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

    if (method === "ecod") {
        const values = points.map((point) => point.value);
        const scores = computeEcodScores(values);
        const contamination = chooseContamination(points.length);
        const threshold = Math.max(thresholdForContamination(scores, contamination, 0.55), 0.55);

        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `ECOD score = ${formatMetric(scores[index], 3)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `ECOD threshold ${formatMetric(threshold, 3)} (${Math.round(contamination * 100)}% contamination).`,
        };
    }

    if (method === "hbos") {
        const values = points.map((point) => point.value);
        const scores = computeHbosScores(values);
        const contamination = chooseContamination(points.length);
        const threshold = thresholdForContamination(scores, contamination, 0);

        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `HBOS score = ${formatMetric(scores[index], 3)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `HBOS histogram threshold ${formatMetric(threshold, 3)} (${Math.round(contamination * 100)}% contamination).`,
        };
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
        const { treeCount, sampleSize } = chooseIsolationForestConfig(points.length);
        const scores = computeIsolationForestScores(values, treeCount, sampleSize);
        const contamination = chooseContamination(points.length);
        const threshold = Math.max(
            thresholdForContamination(scores, contamination, 0),
            0.55,
        );

        const rows = points
            .map((point, index) => ({
                ...point,
                score: scores[index],
                reason: `Isolation score = ${formatMetric(scores[index], 3)}`,
            }))
            .filter((point) => point.score >= threshold)
            .sort((left, right) => right.score - left.score);

        return {
            ...basePreview,
            ...labelOutlierResults(points, rows),
            thresholdLabel: `Isolation Forest threshold ${formatMetric(threshold, 3)} (${Math.round(contamination * 100)}% contamination, ${treeCount} trees, sample ${sampleSize}).`,
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

    const sortedNeighbors = distances.map((row) =>
        row
            .map((distance, otherIndex) => ({ distance, otherIndex }))
            .sort((left, right) => left.distance - right.distance),
    );

    const kDistances = sortedNeighbors.map((neighbors) => neighbors[k - 1].distance);

    const lrd = sortedNeighbors.map((neighbors) => {
        const nearest = neighbors.slice(0, k);
        const reachSum = nearest.reduce(
            (sum, neighbor) => sum + Math.max(kDistances[neighbor.otherIndex], neighbor.distance),
            0,
        );
        return reachSum > 0 ? k / reachSum : 0;
    });

    return sortedNeighbors.map((neighbors, index) => {
        const ownLrd = Math.max(lrd[index], 1e-8);
        const nearest = neighbors.slice(0, k);
        const score = nearest.reduce(
            (sum, neighbor) => sum + lrd[neighbor.otherIndex] / ownLrd,
            0,
        );
        return score / k;
    });
}
