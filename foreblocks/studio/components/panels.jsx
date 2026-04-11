import { useEffect, useMemo, useState } from "react";

import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { NumberField, Panel, PipelineNode, SelectField, SkeletonBlock, StatPill, ToggleField } from "./base.jsx";
import { buildOutlierPreview } from "../lib/outlier-algorithms.js";
import { buildFilterPreview } from "../lib/filter-algorithms.js";
import { buildDecomposition, buildEmdOverviewData, buildEmdComponentSeries, shiftValues, buildRollingHistory } from "../lib/signal-algorithms.js";

function uniqueSuggestions(items) {
    return Array.from(new Map(items.map((item) => [item.window, item])).values());
}

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

function standardDeviation(values, mean) {
    return Math.sqrt(values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length);
}

function formatMetric(value, digits = 2) {
    return Number(value).toFixed(digits);
}

function grangerMatrixTone(result) {
    if (!result) {
        return "neutral";
    }
    if (result.significance === "significant") {
        return "strong";
    }
    if (result.significance === "borderline") {
        return "medium";
    }
    return "weak";
}

const EXPLORE_TAB_GROUPS = [
    {
        key: "signal",
        label: "Signal",
        description: "Raw behavior, decomposition, lag structure, and repeating cycles.",
        tabs: ["series", "decomposition", "autocorrelation", "spectral"],
    },
    {
        key: "relationships",
        label: "Relationships",
        description: "Companion drivers, directional evidence, and timestamp-driven effects.",
        tabs: ["exogenous", "granger", "calendar"],
    },
];

const EXPLORE_TAB_META = {
    series: { label: "Time Series" },
    decomposition: { label: "Decomposition" },
    autocorrelation: { label: "Autocorrelation" },
    spectral: { label: "Spectral" },
    exogenous: { label: "Exogenous" },
    granger: { label: "Granger" },
    calendar: { label: "Calendar" },
};

const EXPLORE_GROUP_BY_TAB = EXPLORE_TAB_GROUPS.reduce((lookup, group) => {
    group.tabs.forEach((tab) => {
        lookup[tab] = group.key;
    });
    return lookup;
}, {});

function buildExploreStats(data, datasetSummary) {
    if (data.length === 0) {
        return [];
    }

    const values = data.map((point) => point.value);
    const mean = average(values);
    const targetGapCount = datasetSummary.missingTargetCount + datasetSummary.invalidTargetCount;
    const missingRate = datasetSummary.rowCount > 0 ? targetGapCount / datasetSummary.rowCount : 0;

    return [
        { label: "Count", value: values.length, tone: "accent" },
        { label: "Mean", value: formatMetric(mean), tone: "neutral" },
        { label: "Std Dev", value: formatMetric(standardDeviation(values, mean)), tone: "neutral" },
        { label: "Median", value: formatMetric(median(values)), tone: "neutral" },
        { label: "Min", value: formatMetric(Math.min(...values)), tone: "neutral" },
        { label: "Max", value: formatMetric(Math.max(...values)), tone: "neutral" },
        { label: "Missing", value: `${targetGapCount} (${formatMetric(missingRate * 100, 1)}%)`, tone: "warm" },
    ];
}



function correlation(valuesA, valuesB) {
    const pairs = valuesA
        .map((value, index) => [value, valuesB[index]])
        .filter(([left, right]) => Number.isFinite(left) && Number.isFinite(right));

    if (pairs.length < 4) {
        return 0;
    }

    const leftValues = pairs.map(([left]) => left);
    const rightValues = pairs.map(([, right]) => right);
    const leftMean = average(leftValues);
    const rightMean = average(rightValues);
    const numerator = pairs.reduce((total, [left, right]) => total + (left - leftMean) * (right - rightMean), 0);
    const leftStd = standardDeviation(leftValues, leftMean);
    const rightStd = standardDeviation(rightValues, rightMean);
    if (leftStd < 1e-10 || rightStd < 1e-10) {
        return 0;
    }
    return numerator / (pairs.length * leftStd * rightStd);
}

function zscoreSeries(values) {
    const finite = values.filter((value) => Number.isFinite(value));
    if (finite.length < 3) {
        return values.map(() => null);
    }
    const mean = average(finite);
    const std = standardDeviation(finite, mean);
    if (std < 1e-10) {
        return values.map(() => null);
    }
    return values.map((value) => (Number.isFinite(value) ? (value - mean) / std : null));
}

function fitLinearRegression(designMatrix, targets) {
    if (designMatrix.length === 0 || targets.length === 0) {
        return null;
    }

    const columnCount = designMatrix[0].length;
    const normalMatrix = Array.from({ length: columnCount }, () => Array.from({ length: columnCount }, () => 0));
    const rhs = Array.from({ length: columnCount }, () => 0);

    for (let rowIndex = 0; rowIndex < designMatrix.length; rowIndex += 1) {
        const row = designMatrix[rowIndex];
        const target = targets[rowIndex];
        for (let left = 0; left < columnCount; left += 1) {
            rhs[left] += row[left] * target;
            for (let right = 0; right < columnCount; right += 1) {
                normalMatrix[left][right] += row[left] * row[right];
            }
        }
    }

    return solveLinearSystem(normalMatrix, rhs);
}

function predictLinear(coefficients, row) {
    if (!coefficients) {
        return null;
    }
    return row.reduce((total, value, index) => total + value * coefficients[index], 0);
}

function evaluateFeatureAblation(targetValues, candidateValues, seasonalLag) {
    const rows = [];

    for (let index = Math.max(1, seasonalLag); index < targetValues.length; index += 1) {
        const target = targetValues[index];
        const lag1 = targetValues[index - 1];
        const seasonal = seasonalLag >= 2 ? targetValues[index - seasonalLag] : null;
        const candidate = candidateValues[index];

        if (!Number.isFinite(target) || !Number.isFinite(lag1) || !Number.isFinite(candidate)) {
            continue;
        }

        const base = [1, lag1];
        if (seasonalLag >= 2 && Number.isFinite(seasonal)) {
            base.push(seasonal);
        }

        rows.push({
            index,
            target,
            base,
            extended: [...base, candidate],
        });
    }

    if (rows.length < 28) {
        return null;
    }

    const testCount = Math.max(8, Math.floor(rows.length * 0.2));
    const trainRows = rows.slice(0, -testCount);
    const testRows = rows.slice(-testCount);

    if (trainRows.length < 12 || testRows.length < 8) {
        return null;
    }

    const baselineCoefficients = fitLinearRegression(trainRows.map((row) => row.base), trainRows.map((row) => row.target));
    const extendedCoefficients = fitLinearRegression(trainRows.map((row) => row.extended), trainRows.map((row) => row.target));

    if (!baselineCoefficients || !extendedCoefficients) {
        return null;
    }

    const baselineErrors = [];
    const extendedErrors = [];

    testRows.forEach((row) => {
        const baselinePrediction = predictLinear(baselineCoefficients, row.base);
        const extendedPrediction = predictLinear(extendedCoefficients, row.extended);
        if (!Number.isFinite(baselinePrediction) || !Number.isFinite(extendedPrediction)) {
            return;
        }
        baselineErrors.push(row.target - baselinePrediction);
        extendedErrors.push(row.target - extendedPrediction);
    });

    if (baselineErrors.length < 4 || extendedErrors.length < 4) {
        return null;
    }

    const baselineMae = average(baselineErrors.map((error) => Math.abs(error)));
    const baselineRmse = Math.sqrt(average(baselineErrors.map((error) => error ** 2)));
    const featureMae = average(extendedErrors.map((error) => Math.abs(error)));
    const featureRmse = Math.sqrt(average(extendedErrors.map((error) => error ** 2)));

    return {
        baselineMae,
        baselineRmse,
        featureMae,
        featureRmse,
        maeGain: baselineMae - featureMae,
        rmseGain: baselineRmse - featureRmse,
        improvementRate: baselineRmse > 1e-10 ? (baselineRmse - featureRmse) / baselineRmse : 0,
        sampleCount: rows.length,
        testCount: baselineErrors.length,
    };
}

function buildFeatureLabCandidates({ data, covariates, selectedCovariateName, lagDepth, rollingWindow, includeCalendar, includeInteractions, includeDecomposition, analysis, decomposition }) {
    const targetValues = data.map((point) => point.value);
    const candidates = [];
    const selectedCovariate = covariates.find((item) => item.name === selectedCovariateName) ?? covariates[0] ?? null;
    const selectedValues = selectedCovariate?.points?.map((point) => point.value ?? null) ?? [];
    const safeLag = clamp(Math.round(lagDepth || 6), 1, 48);
    const safeRollingWindow = clamp(Math.round(rollingWindow || 6), 2, 48);

    if (selectedCovariate && selectedValues.length === targetValues.length) {
        const covariateLag1 = shiftValues(selectedValues, 1);
        const covariateLagK = shiftValues(selectedValues, safeLag);
        const covariateDelta = selectedValues.map((value, index) => (
            Number.isFinite(value) && index >= 1 && Number.isFinite(selectedValues[index - 1])
                ? value - selectedValues[index - 1]
                : null
        ));
        const rollingMean = buildRollingHistory(selectedValues, safeRollingWindow, average);
        const rollingStd = buildRollingHistory(selectedValues, safeRollingWindow, (window) => standardDeviation(window, average(window)));

        candidates.push(
            {
                key: `${selectedCovariate.name}-lag1`,
                label: `${selectedCovariate.name} lag-1`,
                group: "lagged covariates",
                description: "One-step lag from the selected numeric companion series.",
                values: covariateLag1,
            },
            {
                key: `${selectedCovariate.name}-lagk`,
                label: `${selectedCovariate.name} lag-${safeLag}`,
                group: "lagged covariates",
                description: `Seasonal-style lag aligned to the selected feature-lag depth (${safeLag}).`,
                values: covariateLagK,
            },
            {
                key: `${selectedCovariate.name}-delta`,
                label: `${selectedCovariate.name} delta-1`,
                group: "lagged covariates",
                description: "First difference from the selected numeric companion series.",
                values: covariateDelta,
            },
            {
                key: `${selectedCovariate.name}-rollmean`,
                label: `${selectedCovariate.name} rolling mean`,
                group: "rolling stats",
                description: `Backward-looking rolling mean over ${safeRollingWindow} steps.`,
                values: rollingMean,
            },
            {
                key: `${selectedCovariate.name}-rollstd`,
                label: `${selectedCovariate.name} rolling std`,
                group: "rolling stats",
                description: `Backward-looking rolling dispersion over ${safeRollingWindow} steps.`,
                values: rollingStd,
            },
        );

        if (includeInteractions) {
            const targetLag1 = shiftValues(targetValues, 1);
            candidates.push({
                key: `${selectedCovariate.name}-interaction`,
                label: `${selectedCovariate.name} x target lag-1`,
                group: "interaction features",
                description: "Interaction between the selected covariate history and the most recent target value.",
                values: targetLag1.map((targetLag, index) => (
                    Number.isFinite(targetLag) && Number.isFinite(covariateLag1[index])
                        ? targetLag * covariateLag1[index]
                        : null
                )),
            });
        }
    }

    if (includeCalendar) {
        const parsedDates = data.map((point) => {
            const parsed = Date.parse(point.timestamp);
            return Number.isNaN(parsed) ? null : new Date(parsed);
        });
        const validDateCount = parsedDates.filter(Boolean).length;
        if (validDateCount >= Math.max(18, Math.floor(data.length * 0.5))) {
            candidates.push(
                {
                    key: "weekday-sin",
                    label: "Weekday sine",
                    group: "calendar encodings",
                    description: "Cyclic day-of-week encoding from parsed timestamps.",
                    values: parsedDates.map((date) => (date ? Math.sin((2 * Math.PI * date.getDay()) / 7) : null)),
                },
                {
                    key: "weekday-cos",
                    label: "Weekday cosine",
                    group: "calendar encodings",
                    description: "Complementary cyclic day-of-week encoding.",
                    values: parsedDates.map((date) => (date ? Math.cos((2 * Math.PI * date.getDay()) / 7) : null)),
                },
                {
                    key: "month-sin",
                    label: "Month sine",
                    group: "calendar encodings",
                    description: "Cyclic month-of-year encoding from timestamps.",
                    values: parsedDates.map((date) => (date ? Math.sin((2 * Math.PI * date.getMonth()) / 12) : null)),
                },
                {
                    key: "month-cos",
                    label: "Month cosine",
                    group: "calendar encodings",
                    description: "Complementary cyclic month-of-year encoding.",
                    values: parsedDates.map((date) => (date ? Math.cos((2 * Math.PI * date.getMonth()) / 12) : null)),
                },
            );
        }
    }

    if (includeDecomposition) {
        const decompositionView = decomposition ?? buildDecomposition(data, analysis);
        if ((decompositionView?.data ?? []).length === data.length) {
            const trendValues = decompositionView.data.map((point) => point.trend ?? null);
            const seasonalValues = decompositionView.data.map((point) => point.seasonal ?? null);
            candidates.push(
                {
                    key: "trend-lag1",
                    label: "Trend lag-1",
                    group: "decomposition-derived drivers",
                    description: "Lagged low-frequency trend component from the target decomposition.",
                    values: shiftValues(trendValues, 1),
                },
                {
                    key: "seasonal-lag1",
                    label: "Seasonal lag-1",
                    group: "decomposition-derived drivers",
                    description: "Lagged aggregate seasonal component from the target decomposition.",
                    values: shiftValues(seasonalValues, 1),
                },
            );
        }
    }

    return candidates
        .map((candidate) => ({
            ...candidate,
            coverage: candidate.values.filter((value) => Number.isFinite(value)).length / Math.max(candidate.values.length, 1),
            correlation: correlation(targetValues, candidate.values),
        }))
        .filter((candidate) => candidate.coverage >= 0.25);
}

function buildFeatureLabResults({ data, covariates, selectedCovariateName, lagDepth, rollingWindow, includeCalendar, includeInteractions, includeDecomposition, analysis, decomposition }) {
    const targetValues = data.map((point) => point.value);
    const seasonalLag = clamp(Math.round(analysis.dominantAcfLag || analysis.dominantPacfLag || 6), 2, 24);
    const candidates = buildFeatureLabCandidates({
        data,
        covariates,
        selectedCovariateName,
        lagDepth,
        rollingWindow,
        includeCalendar,
        includeInteractions,
        includeDecomposition,
        analysis,
        decomposition,
    });

    const results = candidates
        .map((candidate) => {
            const score = evaluateFeatureAblation(targetValues, candidate.values, seasonalLag);
            if (!score) {
                return null;
            }
            return {
                ...candidate,
                ...score,
                rmseGainLabel: `${score.rmseGain >= 0 ? "+" : ""}${formatMetric(score.rmseGain, 4)}`,
                improvementPercentLabel: `${score.improvementRate >= 0 ? "+" : ""}${formatMetric(score.improvementRate * 100, 1)}%`,
            };
        })
        .filter(Boolean)
        .sort((left, right) => right.rmseGain - left.rmseGain);

    return {
        candidates,
        results,
        seasonalLag,
    };
}

function buildFeaturePreviewData(data, candidateValues) {
    const targetNormalized = zscoreSeries(data.map((point) => point.value));
    const featureNormalized = zscoreSeries(candidateValues);

    return data
        .map((point, index) => ({
            label: point.month ?? point.timestamp ?? `T${index + 1}`,
            timestamp: point.timestamp ?? `T${index + 1}`,
            target: targetNormalized[index],
            feature: featureNormalized[index],
        }))
        .filter((point) => Number.isFinite(point.target) || Number.isFinite(point.feature));
}

function normalizeSeries(values) {
    const valid = values.filter((value) => value != null);
    if (valid.length < 2) {
        return values.map((value) => (value == null ? null : 0));
    }

    const mean = average(valid);
    const std = standardDeviation(valid, mean);
    return values.map((value) => (value == null ? null : (value - mean) / Math.max(std, 1e-6)));
}

function buildComparisonData(targetData, covariatePoints) {
    if (!covariatePoints || covariatePoints.length === 0 || targetData.length === 0) {
        return [];
    }

    const alignedCovariates = covariatePoints.slice(-targetData.length);
    const normalizedTarget = normalizeSeries(targetData.map((point) => point.value));
    const normalizedCovariate = normalizeSeries(alignedCovariates.map((point) => point.value));

    return targetData.map((point, index) => ({
        month: point.month,
        timestamp: point.timestamp,
        target: normalizedTarget[index],
        covariate: normalizedCovariate[index],
    }));
}

function buildSpectralComparisonData(spectralDiagnostics, acfCurve) {
    if (!spectralDiagnostics?.spectrum?.length) {
        return [];
    }

    const acfMagnitude = new Map(
        (acfCurve ?? []).map((point) => [point.lag, Math.abs(point.acf)]),
    );
    const maxAcfMagnitude = Math.max(...(acfCurve ?? []).map((point) => Math.abs(point.acf)), 1e-6);

    return [...spectralDiagnostics.spectrum]
        .filter((point) => point.period <= 48)
        .sort((left, right) => left.period - right.period)
        .map((point) => ({
            period: Number(point.period.toFixed(1)),
            spectralPower: point.normalizedPower,
            acfStrength: (acfMagnitude.get(Math.round(point.period)) ?? 0) / maxAcfMagnitude,
        }));
}

function buildSpectralAgreement(spectralDiagnostics, acfPeaks) {
    const spectralPeaks = spectralDiagnostics?.peaks ?? [];
    const lagPeaks = acfPeaks ?? [];
    if (spectralPeaks.length === 0) {
        return {
            label: "insufficient",
            strongestSharedPeriod: 0,
            sharedCount: 0,
            dominantDelta: 0,
        };
    }

    const matches = spectralPeaks
        .map((peak) => {
            const nearestLag = lagPeaks.reduce((best, candidate) => {
                if (!best) {
                    return candidate;
                }
                return Math.abs(candidate.lag - peak.period) < Math.abs(best.lag - peak.period) ? candidate : best;
            }, null);

            if (!nearestLag || Math.abs(nearestLag.lag - peak.period) > 2) {
                return null;
            }

            return {
                spectralPeriod: peak.period,
                lag: nearestLag.lag,
                delta: Math.abs(nearestLag.lag - peak.period),
                power: peak.normalizedPower,
            };
        })
        .filter(Boolean);

    const dominantDelta = Math.min(...lagPeaks.map((peak) => Math.abs(peak.lag - spectralPeaks[0].period)), Infinity);
    const strongestShared = matches.sort((left, right) => right.power - left.power)[0] ?? null;
    const label = dominantDelta <= 1
        ? "strong agreement"
        : matches.length >= 2
            ? "partial agreement"
            : matches.length >= 1
                ? "weak agreement"
                : "divergent";

    return {
        label,
        strongestSharedPeriod: strongestShared ? Number(strongestShared.spectralPeriod.toFixed(1)) : 0,
        sharedCount: matches.length,
        dominantDelta: Number((Number.isFinite(dominantDelta) ? dominantDelta : 0).toFixed(1)),
    };
}

function buildSpectralSuggestions(spectralDiagnostics, suggestedLagWindow) {
    const peakSuggestions = (spectralDiagnostics?.peaks ?? []).slice(0, 3).map((peak, index) => {
        const period = Math.max(2, Math.round(peak.period));
        return {
            label: `Spec P${period}`,
            detail: `window ${clamp(period * 2, 24, 96)}`,
            window: clamp(period * 2, 24, 96),
            rank: index,
        };
    });

    return uniqueSuggestions([
        ...peakSuggestions,
        {
            label: "Recommended",
            detail: `window ${suggestedLagWindow}`,
            window: suggestedLagWindow,
            rank: 99,
        },
    ]);
}

function buildCalendarProfiles(calendarDiagnostics) {
    return [
        {
            key: "weekday",
            title: "Day-of-week effect",
            profile: calendarDiagnostics?.weekdayProfile,
        },
        {
            key: "month",
            title: "Month-of-year effect",
            profile: calendarDiagnostics?.monthProfile,
        },
        {
            key: "hour",
            title: "Hour-of-day effect",
            profile: calendarDiagnostics?.hourProfile,
        },
    ].filter((item) => item.profile?.available && item.profile?.data?.length);
}



const KEYWORDS = new Set([
    "and",
    "as",
    "class",
    "def",
    "else",
    "False",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "None",
    "not",
    "or",
    "print",
    "return",
    "True",
]);

const BENCHMARK_COLORS = {
    naive: "var(--warm)",
    seasonal_naive: "var(--accent)",
    moving_average: "var(--secondary)",
    mean: "#f59e0b",
    median: "#10b981",
    exponential_smoothing: "#8b5cf6",
    drift: "var(--cool)",
    linear_trend: "#1f6feb",
};

function tokenType(value) {
    if (/^\s+$/.test(value)) {
        return "plain";
    }
    if (/^#.*$/.test(value)) {
        return "comment";
    }
    if (/^("([^"\\]|\\.)*"|'([^'\\]|\\.)*')$/.test(value)) {
        return "string";
    }
    if (/^\d+(?:\.\d+)?$/.test(value)) {
        return "number";
    }
    if (/^[(){}\[\].,:=+\-*/%<>!]+$/.test(value)) {
        return "operator";
    }
    if (KEYWORDS.has(value)) {
        return "keyword";
    }
    if (/^[A-Z][A-Za-z0-9_]*$/.test(value)) {
        return "class";
    }
    return "plain";
}

function highlightLine(line) {
    const commentIndex = line.indexOf("#");
    const content = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
    const comment = commentIndex >= 0 ? line.slice(commentIndex) : "";
    const tokens = content.match(/\s+|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|[(){}\[\].,:=+\-*/%<>!]+/g) ?? [content];

    const parts = tokens.map((token, index) => {
        const nextToken = tokens[index + 1];
        let type = tokenType(token);

        if (type === "plain" && /^[a-z_][A-Za-z0-9_]*$/.test(token) && nextToken === "(") {
            type = "function";
        }

        return {
            value: token,
            type,
        };
    });

    if (comment) {
        parts.push({ value: comment, type: "comment" });
    }

    if (parts.length === 0) {
        return [{ value: line || " ", type: "plain" }];
    }

    return parts;
}

function HighlightedCode({ code }) {
    return code.split("\n").map((line, index) => (
        <div className="code-line" key={`${index + 1}-${line}`}>
            <span className="code-gutter">{index + 1}</span>
            <span className="code-content">
                {highlightLine(line).map((part, partIndex) => (
                    <span key={`${index + 1}-${partIndex}`} className={`code-token code-token-${part.type}`}>
                        {part.value}
                    </span>
                ))}
            </span>
        </div>
    ));
}

function DecompositionSubplot({ title, data, dataKey, color, showXAxis = false }) {
    return (
        <div className="chart-shell short-chart decomposition-subplot">
            <div className="mini-chart-title">{title}</div>
            <ResponsiveContainer width="100%" height={180}>
                <LineChart data={data}>
                    <CartesianGrid stroke="var(--grid)" vertical={false} />
                    <XAxis
                        dataKey="month"
                        tick={{ fill: "var(--muted)" }}
                        tickLine={false}
                        axisLine={false}
                        minTickGap={20}
                        hide={!showXAxis}
                    />
                    <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} width={44} />
                    <Tooltip
                        labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                        contentStyle={{
                            backgroundColor: "var(--panel)",
                            borderColor: "var(--border)",
                            borderRadius: 16,
                            color: "var(--text)",
                        }}
                    />
                    <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

export function OverviewPanel({
    analysis,
    config,
    familyMeta,
    blueprintSummary,
    datasetSummary,
    automationDiagnostics,
    benchmark,
    spectralDiagnostics,
    forecastabilityDiagnostics,
    experimentNarrative,
    onRunAction,
}) {
    const completeness = datasetSummary.rowCount > 0
        ? datasetSummary.validObservationCount / datasetSummary.rowCount
        : 0;
    const targetGapCount = datasetSummary.missingTargetCount + datasetSummary.invalidTargetCount;
    const readinessTone = automationDiagnostics?.readinessScore >= 80
        ? "cool"
        : automationDiagnostics?.readinessScore >= 60
            ? "neutral"
            : "warm";
    const topActions = automationDiagnostics?.topActions?.slice(0, 3) ?? [];

    return (
        <Panel title="Executive Summary" kicker="Workspace overview" accent>
            <div className="stat-grid">
                <StatPill label="Observations" value={analysis.count || datasetSummary.validObservationCount || "-"} />
                <StatPill label="Completeness" value={datasetSummary.rowCount > 0 ? formatPercent(completeness) : "-"} tone="neutral" />
                <StatPill label="Missing cells" value={datasetSummary.missingCellCount || 0} tone="neutral" />
                <StatPill label="Target gaps" value={targetGapCount || 0} tone="neutral" />
                <StatPill label="Timestamp gaps" value={datasetSummary.missingTimestampCount || 0} tone="neutral" />
                <StatPill label="Dataset source" value={datasetSummary.rowCount > 0 ? `${datasetSummary.validObservationCount} points` : "pending"} tone="neutral" />
                <StatPill label="Model family" value={familyMeta.label} />
            </div>
            <p className="lead-copy">{blueprintSummary}</p>
            {experimentNarrative?.tags?.length ? (
                <div className="header-chip-row overview-tag-row">
                    {experimentNarrative.tags.map((tag) => (
                        <span key={tag} className="header-chip">{tag}</span>
                    ))}
                </div>
            ) : null}
            <div className="pipeline-map">
                <PipelineNode title="Source" detail={config.data.filename} tone="source" />
                <PipelineNode title="Preprocess" detail={`${config.prep.scalingMethod} · w=${config.prep.windowSize}`} tone="preprocess" />
                <PipelineNode title="Backbone" detail={familyMeta.label} tone="model" />
                <PipelineNode title="Train" detail={`${config.train.epochs} epochs`} tone="train" />
            </div>

            <div className="summary-card-grid overview-detail-grid">
                <div className="summary-card summary-card-highlight">
                    <div className="summary-card-title">Current blueprint</div>
                    <div className="architecture-name">{familyMeta.label}</div>
                    <div className="summary-card-copy">
                        Context {config.prep.windowSize}, horizon {config.prep.horizon}, scaling {config.prep.scalingMethod}, and trainer budget {config.train.epochs} epochs.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Dataset summary</div>
                    <div className="architecture-name">{datasetSummary.rowCount} rows</div>
                    <div className="summary-card-copy">
                        {datasetSummary.exogenousCount} exogenous columns, {datasetSummary.missingTargetCount} missing target values, {datasetSummary.missingTimestampCount} missing timestamps.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Pipeline readiness</div>
                    <div className="architecture-name">{analysis.suggestedLagWindow} step context</div>
                    <div className="summary-card-copy">
                        Current preprocessing window and timeframe recommendations are driven by live signal diagnostics.
                    </div>
                </div>
            </div>

            <div className="guidance-grid overview-guidance-grid">
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Why this blueprint</div>
                    <div className="guidance-list">
                        {(experimentNarrative?.reasons ?? []).slice(0, 4).map((reason) => (
                            <div key={reason.title} className="guidance-item">
                                <div className="summary-card-title">{reason.title}</div>
                                <div className="summary-card-copy">{reason.body}</div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Immediate next actions</div>
                    <div className="guidance-list">
                        {topActions.length > 0 ? topActions.map((action) => (
                            <div key={`${action.key}-${action.window ?? "na"}`} className="guidance-item">
                                <div className="summary-card-title">{action.title}</div>
                                <div className="summary-card-copy">{action.detail}</div>
                                <button className="mini-button overview-action-button" type="button" onClick={() => onRunAction?.(action)}>
                                    Open
                                </button>
                            </div>
                        )) : (
                            <div className="guidance-item">
                                <div className="summary-card-title">No queued actions</div>
                                <div className="summary-card-copy">Automation has not surfaced any immediate follow-up action yet.</div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </Panel>
    );
}

export function DiagnosticsPanel({
    status,
    errorMessage,
    analysis,
    spectralDiagnostics,
    forecastabilityDiagnostics,
    automationDiagnostics,
    residualDiagnostics,
    benchmark,
    datasetSummary,
}) {
    if (status === "loading") {
        return (
            <Panel title="Diagnostics" kicker="Signal & forecast health">
                <p className="lead-copy">
                    Diagnostics are computing for the current series and configuration. This panel will summarize signal quality, forecast readiness, and residual behavior.
                </p>
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Diagnostics" kicker="Signal & forecast health">
                <p className="lead-copy">Diagnostics unavailable: {errorMessage}</p>
            </Panel>
        );
    }

    const readinessTone = automationDiagnostics?.readinessScore >= 80
        ? "cool"
        : automationDiagnostics?.readinessScore >= 60
            ? "neutral"
            : "warm";

    const residualTone = residualDiagnostics?.label === "white noise" ? "cool" : residualDiagnostics?.label ? "warm" : "neutral";

    const signalBackdrop = datasetSummary.rowCount > 0
        ? `${datasetSummary.validObservationCount}/${datasetSummary.rowCount} valid observations`
        : "No series loaded";

    return (
        <Panel title="Diagnostics" kicker="Signal & forecast health" accent>
            <div className="stat-grid">
                <StatPill label="Stationarity" value={analysis.stationarityLabel} tone={analysis.stationarityLabel === "stationary" ? "cool" : analysis.stationarityLabel === "inconclusive" ? "neutral" : "warm"} />
                <StatPill label="Forecastability" value={analysis.forecastabilityLabel} tone={forecastabilityDiagnostics?.label === "high" ? "cool" : forecastabilityDiagnostics?.label === "low" ? "warm" : "neutral"} />
                <StatPill label="Seasonality" value={spectralDiagnostics?.available ? `P ${spectralDiagnostics.dominantPeriod}` : "pending"} tone="neutral" />
                <StatPill label="Readiness" value={automationDiagnostics ? `${automationDiagnostics.readinessScore}/100` : "-"} tone={readinessTone} />
                <StatPill label="Residuals" value={residualDiagnostics?.label ?? "pending"} tone={residualTone} />
                <StatPill label="Patch fit" value={analysis.patchingLabel} tone={analysis.patchingLabel === "good" ? "cool" : analysis.patchingLabel === "moderate" ? "neutral" : "warm"} />
            </div>

            <div className="summary-card-grid overview-detail-grid">
                <div className="summary-card summary-card-highlight">
                    <div className="summary-card-title">Signal posture</div>
                    <div className="architecture-name">{analysis.stationarityLabel}</div>
                    <div className="summary-card-copy">
                        {analysis.stationarityLabel === "nonstationary"
                            ? "The current signal shows nonstationary behavior; detrending or differencing may improve stability."
                            : "Signal diagnostics indicate a stable enough series for the current forecasting pipeline."}
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Recommended window</div>
                    <div className="architecture-name">{analysis.suggestedLagWindow}</div>
                    <div className="summary-card-copy">
                        Recommended context based on ACF/PACF peaks and spectrum structure.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Model posture</div>
                    <div className="architecture-name">{analysis.forecastabilityLabel}</div>
                    <div className="summary-card-copy">
                        {forecastabilityDiagnostics?.label === "high"
                            ? "The series is well suited to richer backbones."
                            : "A simpler configuration may be preferable when forecastability is low."}
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Spectral signal</div>
                    <div className="architecture-name">{spectralDiagnostics?.available ? `${formatMetric((spectralDiagnostics?.normalizedPeakPower ?? 0) * 100, 1)}%` : "pending"}</div>
                    <div className="summary-card-copy">
                        {spectralDiagnostics?.available
                            ? "Dominant cycle strength is expressed as normalized spectral power."
                            : "Spectral diagnostics are still computing or insufficient history is available."}
                    </div>
                </div>
            </div>

            {spectralDiagnostics?.available ? (
                <div className="chart-shell">
                    <div className="mini-chart-title">Spectral energy distribution</div>
                    <ResponsiveContainer width="100%" height={240}>
                        <AreaChart data={spectralDiagnostics.spectrum.slice(0, 30)}>
                            <defs>
                                <linearGradient id="diagnosticsSpectrum" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.48} />
                                    <stop offset="95%" stopColor="var(--accent-soft)" stopOpacity={0.03} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="period" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value) => [Number(value).toFixed(4), "power"]}
                                labelFormatter={(label) => `Period ${label}`}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Area type="monotone" dataKey="power" stroke="var(--accent)" fill="url(#diagnosticsSpectrum)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            ) : null}

            {benchmark?.winner ? (
                <div className="summary-card-grid benchmark-grid">
                    <div className="summary-card">
                        <div className="summary-card-title">Benchmark leader</div>
                        <div className="architecture-name">{benchmark.winner.label}</div>
                        <div className="summary-card-copy">Current top baseline across rolling-origin backtests.</div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Horizon range</div>
                        <div className="architecture-name">{benchmark.horizon}</div>
                        <div className="summary-card-copy">The validation horizon used for the current benchmark ladder.</div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Folds</div>
                        <div className="architecture-name">{benchmark.splits}</div>
                        <div className="summary-card-copy">Number of backtest folds used for stable comparison.</div>
                    </div>
                </div>
            ) : null}

            {residualDiagnostics?.available ? (
                <div className="summary-card-grid benchmark-grid">
                    <div className="summary-card">
                        <div className="summary-card-title">Residual whiteness</div>
                        <div className="architecture-name">{residualDiagnostics.label}</div>
                        <div className="summary-card-copy">
                            {residualDiagnostics.rejectWhiteNoise
                                ? "Residual correlation remains after the baseline." : "Residuals resemble white noise."}
                        </div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Ljung-Box Q</div>
                        <div className="architecture-name">{residualDiagnostics.statistic}</div>
                        <div className="summary-card-copy">p-value {residualDiagnostics.pValue}</div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Max residual ACF</div>
                        <div className="architecture-name">{residualDiagnostics.maxResidualAcf}</div>
                        <div className="summary-card-copy">Largest autocorrelation observed in the residual test window.</div>
                    </div>
                </div>
            ) : null}
        </Panel>
    );
}

export function AutomationPanel({ status, errorMessage, automationDiagnostics, analysis, onRunAction }) {
    if (status === "loading") {
        return (
            <Panel title="Action Queue" kicker="Decision engine">
                <SkeletonBlock pills={4} lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Action Queue" kicker="Decision engine">
                <p className="lead-copy">
                    Automation diagnostics unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!automationDiagnostics) {
        return null;
    }

    const tone = automationDiagnostics.readinessScore >= 80
        ? "cool"
        : automationDiagnostics.readinessScore >= 60
            ? "neutral"
            : "warm";

    return (
        <Panel title="Action Queue" kicker="Decision engine" accent>
            <AutomationPanelContent
                automationDiagnostics={automationDiagnostics}
                analysis={analysis}
                tone={tone}
                onRunAction={onRunAction}
            />
        </Panel>
    );
}

function AutomationPanelContent({ automationDiagnostics, analysis, tone, onRunAction }) {
    const [dismissedActions, setDismissedActions] = useState(() => new Set());
    const [dismissedItems, setDismissedItems] = useState(() => new Set());

    const visibleActions = automationDiagnostics.topActions.filter(
        (action) => !dismissedActions.has(`${action.key}-${action.window ?? "na"}`),
    );

    const handleRunAction = (action) => {
        const key = `${action.key}-${action.window ?? "na"}`;
        setDismissedActions((prev) => new Set([...prev, key]));
        onRunAction(action);
    };

    const handleRunItem = (item) => {
        setDismissedItems((prev) => new Set([...prev, item.text]));
        onRunAction(item);
    };

    return (
        <>
            <div className="stat-grid compact-grid automation-score-grid">
                <StatPill label="Readiness" value={`${automationDiagnostics.readinessScore}/100`} tone={tone} />
                <StatPill label="Confidence" value={`${automationDiagnostics.automationConfidence}/100`} tone="neutral" />
                <StatPill label="Data quality" value={`${automationDiagnostics.dataQualityScore}/100`} tone="neutral" />
                <StatPill label="Consensus" value={`${automationDiagnostics.consensusScore}/100`} tone="neutral" />
            </div>

            {visibleActions.length > 0 ? (
                <div className="summary-card-grid automation-grid automation-action-grid">
                    {visibleActions.map((action) => {
                        const runnable = ["apply_blueprint", "auto_prep", "apply_window", "apply_spectral_window"].includes(action.key);
                        return (
                            <div key={`${action.key}-${action.window ?? "na"}`} className="summary-card automation-action-card">
                                <div className="summary-card-title">{action.priority} priority</div>
                                <div className="architecture-name">{action.title}</div>
                                <div className="summary-card-copy">{action.detail}</div>
                                <button className="mini-button automation-action-button" type="button" onClick={() => handleRunAction(action)}>
                                    {runnable ? "Run action" : "Open guide"}
                                </button>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <p className="lead-copy" style={{ marginBottom: 14 }}>All recommended actions have been applied.</p>
            )}

            <div className="automation-info-grid">
                <div className="callout automation-callout">
                    <div className="callout-title">Highlights</div>
                    <div className="callout-copy automation-note-list">
                        {automationDiagnostics.highlights.filter((item) => !dismissedItems.has(item.text)).length > 0
                            ? automationDiagnostics.highlights
                                .filter((item) => !dismissedItems.has(item.text))
                                .map((item) => (
                                    <button key={item.text} className="automation-note-button" type="button" onClick={() => handleRunItem(item)}>
                                        {item.text}
                                    </button>
                                ))
                            : <div>No strong positive signals registered yet.</div>}
                    </div>
                </div>
                <div className="callout automation-callout">
                    <div className="callout-title">Risks</div>
                    <div className="callout-copy automation-note-list">
                        {automationDiagnostics.risks.filter((item) => !dismissedItems.has(item.text)).length > 0
                            ? automationDiagnostics.risks
                                .filter((item) => !dismissedItems.has(item.text))
                                .map((item) => (
                                    <button key={item.text} className="automation-note-button" type="button" onClick={() => handleRunItem(item)}>
                                        {item.text}
                                    </button>
                                ))
                            : <div>No dominant execution risk flagged by the current diagnostics.</div>}
                    </div>
                </div>
            </div>

            <div className="automation-flag-row">
                <span className="header-chip">readiness {automationDiagnostics.readinessLabel}</span>
                <span className="header-chip">family {analysis.recommendedFamily}</span>
                <span className="header-chip">window {analysis.suggestedLagWindow}</span>
                <span className="header-chip">horizon {analysis.recommendedHorizon}</span>
                <span className="header-chip">scaling {analysis.scalingMethod}</span>
                <span className="header-chip">forecastability {analysis.forecastabilityLabel}</span>
            </div>
        </>
    );
}

export function StationarityPanel({ analysis, status, errorMessage, onApplyPreprocessing, hasMissingValues }) {
    if (status === "loading") {
        return (
            <Panel title="Signal Checks" kicker="Stationarity & readiness">
                <SkeletonBlock pills={3} lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Signal Checks" kicker="Stationarity & readiness">
                <p className="lead-copy">
                    Stationarity diagnostics unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    return (
        <Panel
            title="Signal Checks"
            kicker="Stationarity & readiness"
            actions={
                <button className="ghost-button" type="button" onClick={onApplyPreprocessing}>
                    Auto-configure prep
                </button>
            }
        >
            <div className="summary-card-grid automation-grid">
                <div className="summary-card">
                    <div className="summary-card-title">ADF unit root</div>
                    <div className="architecture-name">{analysis.adfStatistic}</div>
                    <div className="summary-card-copy">
                        5% critical value {analysis.adfCriticalValue}. {analysis.adfRejectUnitRoot ? "Rejects unit root." : "Does not reject unit root."} Lag order {analysis.adfLagOrder}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">KPSS level stationarity</div>
                    <div className="architecture-name">{analysis.kpssStatistic}</div>
                    <div className="summary-card-copy">
                        5% critical value {analysis.kpssCriticalValue}. {analysis.kpssRejectStationarity ? "Rejects level stationarity." : "Does not reject level stationarity."} Bandwidth {analysis.kpssBandwidth}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Automated verdict</div>
                    <div className="architecture-name">{analysis.stationarityLabel}</div>
                    <div className="summary-card-copy">
                        {analysis.automationSummary}
                    </div>
                </div>
            </div>
            <div className="automation-flag-row">
                {analysis.automationFlags.map((flag) => (
                    <span key={flag} className="header-chip">
                        {flag}
                    </span>
                ))}
                <span className="header-chip">Cadence hint: {analysis.frequencyHint}</span>
                {hasMissingValues ? <span className="header-chip">Enable imputation</span> : null}
            </div>
        </Panel>
    );
}

export function RegimePanel({ status, errorMessage, changePoints, stabilityDiagnostics, seriesData = [], changePointMethod = "segmentation", onChangePointMethod }) {
    if (status === "loading") {
        return (
            <Panel title="Regime & Stability" kicker="Structural diagnostics">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Regime & Stability" kicker="Structural diagnostics">
                <p className="lead-copy">
                    Regime diagnostics unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    const strongestChangePoint = changePoints?.points?.reduce((best, point) => (
        !best || point.score > best.score ? point : best
    ), null);

    return (
        <Panel title="Regime & Stability" kicker="Structural diagnostics">
            <div className="field-grid regime-selector-grid">
                <SelectField
                    label="Change-point algorithm"
                    value={changePointMethod}
                    onChange={(value) => onChangePointMethod?.(value)}
                    options={[
                        { value: "segmentation", label: "Penalized segmentation" },
                        { value: "mean_shift", label: "Windowed mean-shift" },
                    ]}
                    hint="Recomputes structural breaks for the loaded series using the selected detector."
                />
            </div>

            <div className="summary-card-grid automation-grid">
                <div className="summary-card">
                    <div className="summary-card-title">Change points</div>
                    <div className="architecture-name">{changePoints?.points?.length ?? 0}</div>
                    <div className="summary-card-copy">
                        {changePoints?.label ?? "No regime estimate yet"}. {changePoints?.methodLabel ?? "Change-point detector"} score {changePoints?.threshold ?? 0}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Largest shift</div>
                    <div className="architecture-name">
                        {strongestChangePoint ? formatMetric(strongestChangePoint.magnitude) : "0.00"}
                    </div>
                    <div className="summary-card-copy">
                        {strongestChangePoint ? `${strongestChangePoint.direction} near ${strongestChangePoint.timestamp}` : "No material structural break detected."}
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Seasonality stability</div>
                    <div className="architecture-name">{stabilityDiagnostics?.seasonalityLabel ?? "insufficient"}</div>
                    <div className="summary-card-copy">
                        Dominant period {stabilityDiagnostics?.dominantPeriod ?? "-"} with spectral drift median {stabilityDiagnostics?.spectralDriftMedian ?? 0} and strength range {stabilityDiagnostics?.seasonalityRange ?? 0}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Exogenous stability</div>
                    <div className="architecture-name">{stabilityDiagnostics?.exogenousLabel ?? "unavailable"}</div>
                    <div className="summary-card-copy">
                        {stabilityDiagnostics?.topExogenousName ? `${stabilityDiagnostics.topExogenousName} leads in ${stabilityDiagnostics.exogenousShare}% of windows with lag std ${stabilityDiagnostics.exogenousLagStd}.` : "No stable exogenous driver detected yet."}
                    </div>
                </div>
            </div>

            <div className="automation-flag-row">
                <span className="header-chip">Change-point method: {changePoints?.methodLabel ?? "pending"}</span>
                <span className="header-chip">Stability method: spectral drift</span>
                <span className="header-chip">Regime consistency: {stabilityDiagnostics?.regimeConsistencyScore ?? 0}/100</span>
                {stabilityDiagnostics?.dominantPeriodCv != null ? <span className="header-chip">Period CV: {formatMetric(stabilityDiagnostics.dominantPeriodCv, 3)}</span> : null}
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Rolling spectral stability</div>
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={stabilityDiagnostics?.windows ?? []}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="window" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[0, 1]} />
                            <Tooltip
                                formatter={(value, name) => [
                                    Number(value).toFixed(3),
                                    name === "seasonalityStrength"
                                        ? "Seasonality strength"
                                        : name === "spectralDrift"
                                            ? "Spectral drift"
                                            : "Top exogenous score",
                                ]}
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="seasonalityStrength" stroke="var(--accent)" strokeWidth={2.4} dot={{ r: 2.8 }} />
                            <Line type="monotone" dataKey="spectralDrift" stroke="var(--secondary)" strokeWidth={2.2} dot={{ r: 2.6 }} />
                            <Line type="monotone" dataKey="exogenousScore" stroke="var(--warm)" strokeWidth={2.2} dot={{ r: 2.8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Series With Change Points</div>
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={seriesData}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis
                                dataKey="t"
                                type="number"
                                domain={["dataMin", "dataMax"]}
                                tick={{ fill: "var(--muted)" }}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => seriesData[Math.round(value)]?.month ?? value}
                            />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value, name) => {
                                    if (name === "value") {
                                        return [Number(value).toFixed(3), "Series value"];
                                    }
                                    return [Number(value).toFixed(3), name];
                                }}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="value" stroke="var(--secondary)" strokeWidth={2.2} dot={false} />
                            {(changePoints?.points ?? []).map((point) => (
                                <ReferenceLine
                                    key={`${point.timestamp}-${point.index}`}
                                    x={point.index}
                                    stroke="var(--warm)"
                                    strokeWidth={1.8}
                                    strokeDasharray="6 4"
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="automation-flag-row">
                {(changePoints?.points ?? []).map((point) => (
                    <span key={`${point.timestamp}-${point.index}`} className="header-chip">
                        {point.timestamp}: {point.direction} ({formatMetric(point.score)})
                    </span>
                ))}
                {(changePoints?.segments ?? []).length > 0 ? (
                    <span className="header-chip">
                        Segments: {(changePoints?.segments ?? []).length}
                    </span>
                ) : null}
                {stabilityDiagnostics?.topExogenousName ? (
                    <span className="header-chip">
                        Stable driver candidate: {stabilityDiagnostics.topExogenousName}
                    </span>
                ) : null}
            </div>
        </Panel>
    );
}

export function OutlierPanel({
    status,
    errorMessage,
    data,
    targetColumn,
    previewMethod,
    onPreviewMethodChange,
    pipelineMethod,
    pipelineEnabled,
    onSyncFromPreprocessing,
}) {
    const outlierPreview = useMemo(
        () => buildOutlierPreview(data, previewMethod),
        [data, previewMethod],
    );

    if (status === "loading") {
        return (
            <Panel title="Outlier Detection" kicker="Diagnostics sandbox">
                <SkeletonBlock pills={3} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Outlier Detection" kicker="Diagnostics sandbox">
                <p className="lead-copy">
                    Outlier preview unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Outlier Detection" kicker="Diagnostics sandbox">
            <div className="diagnostics-control-row">
                <div className="field-grid regime-selector-grid">
                    <SelectField
                        label="Preview algorithm"
                        value={previewMethod}
                        onChange={(value) => onPreviewMethodChange?.(value)}
                        options={[
                            { value: "iqr", label: "IQR fences" },
                            { value: "zscore", label: "Z-score" },
                            { value: "mad", label: "MAD" },
                            { value: "isolation_forest", label: "Isolation Forest" },
                            { value: "lof", label: "Local Outlier Factor" },
                            { value: "none", label: "No detection" },
                        ]}
                        hint="This selector only affects the diagnostics preview, not the preprocessing pipeline."
                    />
                </div>
                <div className="diagnostics-inline-actions">
                    <button
                        type="button"
                        className="ghost-button"
                        onClick={() => onSyncFromPreprocessing?.()}
                        disabled={previewMethod === pipelineMethod}
                    >
                        Sync from preprocessing
                    </button>
                </div>
            </div>

            <div className="callout dataset-callout outlier-preview">
                <div className="callout-title">Outlier Detection Preview</div>
                <div className="callout-copy dataset-status-copy">
                    <div>
                        Screening {targetColumn || "target"} with <strong>{previewMethod}</strong>. Pipeline preprocessing still uses <strong>{pipelineMethod}</strong> with cleanup {pipelineEnabled ? "enabled" : "disabled"}.
                    </div>
                    <div>{outlierPreview.thresholdLabel}</div>
                </div>

                <div className="outlier-stat-grid">
                    <StatPill label="Flagged" value={outlierPreview.flaggedCount} tone="warm" />
                    <StatPill label="Rate" value={formatPercent(outlierPreview.flaggedRate)} tone="cool" />
                    <StatPill label="Retained" value={outlierPreview.retainedCount} tone="accent" />
                </div>

                <div className="chart-shell outlier-chart-shell">
                    <div className="mini-chart-title">Series With Detected Outliers</div>
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={outlierPreview.chartData}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value, name) => {
                                    if (name === "value") {
                                        return [formatMetric(value, 3), targetColumn || "target"];
                                    }

                                    if (name === "outlierValue") {
                                        return [formatMetric(value, 3), "Detected outlier"];
                                    }

                                    return [value, name];
                                }}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="value" stroke="var(--warm)" strokeWidth={2.1} dot={false} name="value" />
                            <Line
                                type="monotone"
                                dataKey="outlierValue"
                                stroke="transparent"
                                connectNulls={false}
                                name="outlierValue"
                                dot={{ r: 5, fill: "var(--accent)", stroke: "var(--warm)", strokeWidth: 2 }}
                                activeDot={{ r: 6, fill: "var(--accent-strong)", stroke: "var(--warm)", strokeWidth: 2 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {previewMethod === "none" ? (
                    <div className="callout-copy">
                        Preview detection is off, so no rows are being flagged.
                    </div>
                ) : outlierPreview.rows.length > 0 ? (
                    <div className="outlier-table-shell">
                        <div className="outlier-table-head">
                            Showing the most extreme flagged rows from the loaded series.
                        </div>
                        <div className="outlier-table-scroll">
                            <table className="outlier-table">
                                <thead>
                                    <tr>
                                        <th>Index</th>
                                        <th>Timestamp</th>
                                        <th>Value</th>
                                        <th>Reason</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {outlierPreview.rows.map((row) => (
                                        <tr key={`${row.timestamp}-${row.index}-${row.value}`}>
                                            <td>{row.index}</td>
                                            <td>{row.timestamp}</td>
                                            <td>{formatMetric(row.value, 3)}</td>
                                            <td>{row.reason}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ) : (
                    <div className="callout-copy">
                        No observations were flagged by the current preview algorithm on the loaded dataset.
                    </div>
                )}
            </div>
        </Panel>
    );
}

export function FilterPanel({ status, errorMessage, data, settings, onSettingsChange }) {
    const preview = useMemo(() => buildFilterPreview(data, settings), [data, settings]);
    const filterEnabled = settings.filterMethod !== "none";
    const detrendEnabled = settings.detrenderMethod !== "none";

    const updateSettings = (key, value) => {
        onSettingsChange?.((current) => ({
            ...current,
            [key]: value,
            ...(key === "filterMethod" ? { applyFilter: value !== "none" } : null),
            ...(key === "detrenderMethod" ? { detrend: value !== "none" } : null),
        }));
    };

    if (status === "loading") {
        return (
            <Panel title="Filter & Detrend" kicker="Diagnostics sandbox">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Filter & Detrend" kicker="Diagnostics sandbox">
                <p className="lead-copy">
                    Filtering preview unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Filter & Detrend" kicker="Diagnostics sandbox">
            <div className="preprocess-control-split">
                <div className="preprocess-control-card preprocess-control-filter">
                    <div className="preprocess-control-heading">Filter</div>
                    <div className="field-grid">
                        <SelectField
                            label="Filtering"
                            value={settings.filterMethod}
                            onChange={(value) => updateSettings("filterMethod", value)}
                            options={[
                                { value: "none", label: "No filtering" },
                                { value: "savgol", label: "Savitzky-Golay" },
                                { value: "moving_average", label: "Rolling average" },
                                { value: "median", label: "Median filter" },
                                { value: "wiener", label: "Wiener-like" },
                                { value: "gaussian", label: "Gaussian" },
                                { value: "ema", label: "Exponential moving average" },
                                { value: "lowess", label: "LOWESS" },
                            ]}
                            hint="Preview-only smoothing. Extra controls appear only for the selected filter."
                        />
                        {filterEnabled && settings.filterMethod !== "lowess" ? (
                            <NumberField
                                label="Filter window"
                                value={settings.filterWindow}
                                onChange={(value) => updateSettings("filterWindow", value)}
                                min={3}
                                max={61}
                                step={2}
                                hint="Auto-adjusted to an odd centered window."
                            />
                        ) : null}
                        {filterEnabled && settings.filterMethod === "savgol" ? (
                            <NumberField
                                label="SavGol polyorder"
                                value={settings.filterPolyorder}
                                onChange={(value) => updateSettings("filterPolyorder", value)}
                                min={1}
                                max={6}
                                hint="Higher orders preserve more curvature inside the same smoothing window."
                            />
                        ) : null}
                        {filterEnabled && settings.filterMethod === "lowess" ? (
                            <NumberField
                                label="LOWESS fraction"
                                value={settings.lowessFraction}
                                onChange={(value) => updateSettings("lowessFraction", value)}
                                min={0.04}
                                max={0.45}
                                step={0.01}
                                hint="Fraction of the visible series used in each local regression."
                            />
                        ) : null}
                    </div>
                </div>
                <div className="preprocess-control-card preprocess-control-detrend">
                    <div className="preprocess-control-heading">Detrend</div>
                    <div className="field-grid">
                        <SelectField
                            label="Detrending"
                            value={settings.detrenderMethod}
                            onChange={(value) => updateSettings("detrenderMethod", value)}
                            options={[
                                { value: "none", label: "No detrending" },
                                { value: "polynomial", label: "Polynomial" },
                            ]}
                            hint="Preview-only trend removal. This does not change the generated preprocessing pipeline."
                        />
                        {detrendEnabled && settings.detrenderMethod === "polynomial" ? (
                            <NumberField
                                label="Polynomial order"
                                value={settings.detrenderOrder}
                                onChange={(value) => updateSettings("detrenderOrder", value)}
                                min={1}
                                max={6}
                                hint="Higher orders fit more curvature into the removed trend."
                            />
                        ) : null}
                    </div>
                </div>
            </div>

            {preview.available ? (
                <>
                    <div className="summary-card-grid benchmark-grid preprocess-preview-grid">
                        <div className="summary-card summary-card-highlight">
                            <div className="summary-card-title">Preview output</div>
                            <div className="architecture-name">
                                {filterEnabled ? settings.filterMethod.replace("_", " ") : "raw"}
                            </div>
                            <div className="summary-card-copy">
                                {detrendEnabled
                                    ? `Detrended with polynomial order ${settings.detrenderOrder}.`
                                    : "No trend removed."}
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Filter window</div>
                            <div className="architecture-name">
                                {filterEnabled && settings.filterMethod !== "lowess" ? preview.filterWindow : "-"}
                            </div>
                            <div className="summary-card-copy">
                                {filterEnabled && settings.filterMethod !== "lowess"
                                    ? "Effective centered window used for the live smoothing preview."
                                    : "The selected setup does not use a centered smoothing window."}
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Method parameter</div>
                            <div className="architecture-name">
                                {!filterEnabled
                                    ? "-"
                                    : settings.filterMethod === "savgol"
                                        ? preview.filterPolyorder
                                        : settings.filterMethod === "lowess"
                                            ? preview.lowessFraction.toFixed(2)
                                            : preview.filterWindow}
                            </div>
                            <div className="summary-card-copy">
                                {!filterEnabled
                                    ? "Filtering is disabled."
                                    : settings.filterMethod === "savgol"
                                        ? "Savitzky-Golay polynomial order."
                                        : settings.filterMethod === "lowess"
                                            ? "Local regression span as a fraction of the series."
                                            : "Effective local neighborhood used by the selected smoother."}
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Std before / after</div>
                            <div className="architecture-name">{preview.rawStd.toFixed(3)} / {preview.outputStd.toFixed(3)}</div>
                            <div className="summary-card-copy">
                                A quick read on how much variance the sandbox removes.
                            </div>
                        </div>
                    </div>

                    <div className="chart-shell preprocess-preview-chart-shell">
                        <div className="mini-chart-title">Live filtering and detrending preview</div>
                        <ResponsiveContainer width="100%" height={320}>
                            <LineChart data={preview.chartData}>
                                <CartesianGrid stroke="var(--grid)" vertical={false} />
                                <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={18} />
                                <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                <Tooltip
                                    labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                    formatter={(value, name) => [Number(value).toFixed(4), name]}
                                    contentStyle={{
                                        backgroundColor: "var(--panel)",
                                        borderColor: "var(--border)",
                                        borderRadius: 16,
                                        color: "var(--text)",
                                    }}
                                />
                                <Line type="monotone" dataKey="raw" stroke="var(--warm)" strokeWidth={2.1} dot={false} name="Raw" />
                                {filterEnabled ? (
                                    <Line type="monotone" dataKey="filtered" stroke="var(--accent)" strokeWidth={2.2} dot={false} name="Filtered" />
                                ) : null}
                                {detrendEnabled ? (
                                    <Line type="monotone" dataKey="trend" stroke="var(--cool)" strokeWidth={1.9} strokeDasharray="5 4" dot={false} name="Trend" />
                                ) : null}
                                <Line type="monotone" dataKey="output" stroke="var(--secondary)" strokeWidth={2.3} dot={false} name="Preview output" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <p className="lead-copy preprocess-preview-copy">
                        This tab is a diagnostics sandbox only. It lets you inspect smoothing windows and polynomial trend order in real time without changing the pipeline or emitted code.
                    </p>
                </>
            ) : (
                <p className="lead-copy preprocess-preview-copy">
                    Load a dataset to see the filtering and polynomial detrender update in real time.
                </p>
            )}
        </Panel>
    );
}

export function EmdPanel({ status, errorMessage, emdDiagnostics, seriesData = [] }) {
    const overviewData = useMemo(
        () => buildEmdOverviewData(seriesData, emdDiagnostics),
        [seriesData, emdDiagnostics],
    );
    const dominantComponent = useMemo(() => {
        return (emdDiagnostics?.components ?? []).reduce((best, component) => (
            !best || component.energyShare > best.energyShare ? component : best
        ), null);
    }, [emdDiagnostics]);
    const energyLadder = useMemo(() => {
        if (!emdDiagnostics?.available) {
            return [];
        }

        return [
            ...(emdDiagnostics.components ?? []).map((component) => ({
                name: component.name,
                energyShare: component.energyShare,
            })),
            {
                name: emdDiagnostics.residual?.name ?? "Residual",
                energyShare: emdDiagnostics.residual?.energyShare ?? 0,
            },
        ];
    }, [emdDiagnostics]);
    const visibleComponents = useMemo(() => {
        if (!emdDiagnostics?.available) {
            return [];
        }

        const palette = ["var(--accent)", "var(--secondary)", "var(--warm)", "var(--cool)", "#1f6feb"];
        const components = (emdDiagnostics.components ?? []).slice(0, 4).map((component, index) => ({
            ...component,
            color: palette[index % palette.length],
            chartData: buildEmdComponentSeries(seriesData, component),
        }));

        if (emdDiagnostics.residual) {
            components.push({
                ...emdDiagnostics.residual,
                name: emdDiagnostics.residual.name ?? "Residual",
                color: palette[components.length % palette.length],
                chartData: buildEmdComponentSeries(seriesData, emdDiagnostics.residual),
            });
        }

        return components;
    }, [emdDiagnostics, seriesData]);

    if (status === "loading") {
        return (
            <Panel title="EMD" kicker="Empirical mode decomposition">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="EMD" kicker="Empirical mode decomposition">
                <p className="lead-copy">
                    EMD unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!emdDiagnostics?.available) {
        return (
            <Panel title="EMD" kicker="Empirical mode decomposition">
                <p className="lead-copy">
                    {emdDiagnostics?.reason ?? "EMD needs a longer, non-constant series before intrinsic mode functions can be extracted."}
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="EMD" kicker="Empirical mode decomposition">
            <div className="summary-card-grid emd-summary-grid">
                <div className="summary-card">
                    <div className="summary-card-title">IMFs extracted</div>
                    <div className="architecture-name">{emdDiagnostics.imfCount}</div>
                    <div className="summary-card-copy">
                        Real EMD sifting pass over the loaded series.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Total sifts</div>
                    <div className="architecture-name">{emdDiagnostics.totalSifts}</div>
                    <div className="summary-card-copy">
                        Stop threshold {formatMetric(emdDiagnostics.siftThreshold ?? 0.03, 3)} with {emdDiagnostics.interpolationLabel.toLowerCase()}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Reconstruction RMSE</div>
                    <div className="architecture-name">{formatMetric(emdDiagnostics.reconstructionError, 6)}</div>
                    <div className="summary-card-copy">
                        Observed signal vs sum of extracted IMFs plus residual.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Dominant mode</div>
                    <div className="architecture-name">{dominantComponent?.name ?? "-"}</div>
                    <div className="summary-card-copy">
                        Energy share {formatMetric((dominantComponent?.energyShare ?? 0) * 100, 1)}% with {dominantComponent?.zeroCrossings ?? 0} zero crossings.
                    </div>
                </div>
            </div>

            <div className="automation-flag-row">
                <span className="header-chip">Method: {emdDiagnostics.methodLabel}</span>
                <span className="header-chip">Interpolation: {emdDiagnostics.interpolationLabel}</span>
                <span className="header-chip">Residual energy: {formatMetric(emdDiagnostics.residualEnergyShare * 100, 1)}%</span>
                {(emdDiagnostics.components?.length ?? 0) > 4 ? <span className="header-chip">Showing first 4 IMFs plus residual</span> : null}
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell emd-energy-shell">
                    <div className="mini-chart-title">Observed vs reconstructed</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={overviewData}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="month" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value, name) => [
                                    formatMetric(Number(value), 4),
                                    name === "observed" ? "Observed" : name === "reconstruction" ? "Reconstruction" : "Residual",
                                ]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="observed" stroke="var(--warm)" strokeWidth={2.3} dot={false} />
                            <Line type="monotone" dataKey="reconstruction" stroke="var(--secondary)" strokeWidth={2.1} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell benchmark-chart-shell emd-energy-shell">
                    <div className="mini-chart-title">Energy ladder</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={energyLadder}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="name" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value) => [`${formatMetric(Number(value) * 100, 2)}%`, "Energy share"]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Bar dataKey="energyShare" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="emd-component-grid">
                {visibleComponents.map((component, index) => (
                    <div key={`${component.name}-${index}`} className="emd-component-card">
                        <DecompositionSubplot
                            title={component.name}
                            data={component.chartData}
                            dataKey="value"
                            color={component.color}
                            showXAxis={index === visibleComponents.length - 1}
                        />
                        <div className="automation-flag-row emd-component-meta">
                            <span className="header-chip">energy {formatMetric((component.energyShare ?? 0) * 100, 1)}%</span>
                            <span className="header-chip">zero crossings {component.zeroCrossings ?? 0}</span>
                            <span className="header-chip">extrema {component.extremaCount ?? 0}</span>
                            {component.sifts != null ? <span className="header-chip">sifts {component.sifts}</span> : null}
                        </div>
                    </div>
                ))}
            </div>

            <p className="lead-copy">
                This panel runs a real empirical mode decomposition on the loaded series inside the diagnostics worker. IMF 1 is the highest-frequency mode, later IMFs become progressively slower, and the residual captures the remaining trend-like structure after iterative mean-envelope removal.
            </p>
        </Panel>
    );
}

export function IntermittencyPanel({ status, errorMessage, intermittencyDiagnostics }) {
    const intermittencyHistogram = intermittencyDiagnostics?.idleRunHistogram ?? [];
    const intermittencyIntervals = intermittencyDiagnostics?.eventIntervals ?? [];
    const intermittencyBaselineHints = intermittencyDiagnostics?.baselineHints ?? [];
    const intermittencyPreprocessingSuggestions = intermittencyDiagnostics?.preprocessingSuggestions ?? [];

    if (status === "loading") {
        return (
            <Panel title="Intermittency" kicker="Preparation lab">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Intermittency" kicker="Preparation lab">
                <p className="lead-copy">
                    Intermittency analysis unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!intermittencyDiagnostics?.available) {
        return (
            <Panel title="Intermittency" kicker="Preparation lab">
                <p className="lead-copy">
                    Intermittency analysis needs a longer target history before sparse-demand and burstiness metrics become reliable.
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Intermittency" kicker="Preparation lab">
            <div className="summary-card-grid benchmark-grid spectral-grid">
                <div className="summary-card summary-card-highlight">
                    <div className="summary-card-title">Intermittency class</div>
                    <div className="architecture-name">{intermittencyDiagnostics.classification}</div>
                    <div className="summary-card-copy">
                        Overall target activity looks {intermittencyDiagnostics.label} with zero share {formatMetric(intermittencyDiagnostics.zeroShare * 100, 1)}%.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">ADI</div>
                    <div className="architecture-name">{formatMetric(intermittencyDiagnostics.adi, 3)}</div>
                    <div className="summary-card-copy">
                        Average interval between active observations.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">CV²</div>
                    <div className="architecture-name">{formatMetric(intermittencyDiagnostics.cv2, 3)}</div>
                    <div className="summary-card-copy">
                        Squared coefficient of variation for active magnitudes.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Longest idle run</div>
                    <div className="architecture-name">{intermittencyDiagnostics.longestIdleRun}</div>
                    <div className="summary-card-copy">
                        Maximum consecutive zero-like stretch under the detected threshold.
                    </div>
                </div>
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Activity timeline</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={intermittencyDiagnostics.activityTimeline}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value, name) => [
                                    formatMetric(Number(value), 4),
                                    name === "value" ? "Series value" : "Active event",
                                ]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="value" stroke="var(--warm)" strokeWidth={2.1} dot={false} />
                            <Line
                                type="monotone"
                                dataKey="activeValue"
                                stroke="transparent"
                                connectNulls={false}
                                dot={{ r: 4, fill: "var(--panel-solid)", stroke: "var(--accent)", strokeWidth: 2 }}
                                activeDot={{ r: 5, fill: "var(--panel-solid)", stroke: "var(--accent-strong)", strokeWidth: 2 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Idle run histogram</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={intermittencyHistogram}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="runLength" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value) => [value, "Run count"]}
                                labelFormatter={(label) => `Idle run ${label}`}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Bar dataKey="count" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {intermittencyIntervals.length > 0 ? (
                <div className="chart-shell benchmark-chart-shell intermittency-interval-shell">
                    <div className="mini-chart-title">Event spacing</div>
                    <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={intermittencyIntervals.slice(0, 24)}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="event" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value, name) => [
                                    Number(value).toFixed(name === "magnitude" ? 4 : 2),
                                    name === "gap" ? "Gap length" : "Event magnitude",
                                ]}
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Bar dataKey="gap" fill="var(--secondary)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : null}

            <div className="guidance-grid">
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Baseline hints</div>
                    <div className="guidance-list">
                        {intermittencyBaselineHints.map((hint) => (
                            <div key={hint.key} className="guidance-item">
                                <div className="summary-card-title">{hint.label}</div>
                                <div className="automation-flag-row">
                                    <span className="header-chip">{hint.suitability}</span>
                                </div>
                                <div className="summary-card-copy">{hint.detail}</div>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Preprocessing suggestions</div>
                    <div className="guidance-list">
                        {intermittencyPreprocessingSuggestions.map((hint) => (
                            <div key={hint.key} className="guidance-item">
                                <div className="summary-card-title">{hint.label}</div>
                                <div className="automation-flag-row">
                                    <span className="header-chip">{hint.suitability}</span>
                                </div>
                                <div className="summary-card-copy">{hint.detail}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <p className="lead-copy">
                Intermittency belongs in the preparation workspace because it directly changes baseline choice, zero-aware modeling assumptions, and preprocessing posture before training starts.
            </p>
        </Panel>
    );
}

export function VolatilityPanel({ status, errorMessage, volatilityDiagnostics }) {
    const volatilityRolling = volatilityDiagnostics?.rollingVolatility ?? [];
    const volatilityShocks = volatilityDiagnostics?.shockTimeline ?? [];
    const volatilityTransformSuggestions = volatilityDiagnostics?.varianceTransformGuidance?.suggestions ?? [];

    if (status === "loading") {
        return (
            <Panel title="Volatility" kicker="Preparation lab">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Volatility" kicker="Preparation lab">
                <p className="lead-copy">
                    Volatility analysis unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!volatilityDiagnostics?.available) {
        return (
            <Panel title="Volatility" kicker="Preparation lab">
                <p className="lead-copy">
                    Volatility clustering diagnostics need a longer sequence of innovations before heteroskedasticity can be tested reliably.
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Volatility" kicker="Preparation lab">
            <div className="summary-card-grid benchmark-grid spectral-grid">
                <div className={`summary-card ${volatilityDiagnostics.label !== "stable" ? "summary-card-highlight" : ""}`}>
                    <div className="summary-card-title">Volatility regime</div>
                    <div className="architecture-name">{volatilityDiagnostics.label}</div>
                    <div className="summary-card-copy">
                        ARCH-style test and rolling variance both point to {volatilityDiagnostics.label} volatility behavior.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">ARCH p-value</div>
                    <div className="architecture-name">{formatMetric(volatilityDiagnostics.archPValue, 4)}</div>
                    <div className="summary-card-copy">
                        Lag order {volatilityDiagnostics.archLag} over squared standardized innovations.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Regime ratio</div>
                    <div className="architecture-name">{formatMetric(volatilityDiagnostics.regimeRatio, 3)}</div>
                    <div className="summary-card-copy">
                        90th vs 10th percentile rolling volatility.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Clustering score</div>
                    <div className="architecture-name">{formatMetric(volatilityDiagnostics.clusteringScore, 3)}</div>
                    <div className="summary-card-copy">
                        Average absolute autocorrelation of absolute innovations.
                    </div>
                </div>
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Rolling volatility</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={volatilityRolling}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value, name) => [
                                    formatMetric(Number(value), 4),
                                    name === "volatility" ? "Rolling volatility" : "Mean absolute change",
                                ]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Line type="monotone" dataKey="volatility" stroke="var(--secondary)" strokeWidth={2.3} dot={{ r: 2.8 }} />
                            <Line type="monotone" dataKey="meanAbsChange" stroke="var(--warm)" strokeWidth={2.1} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Absolute shocks</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={volatilityShocks}>
                            <defs>
                                <linearGradient id="shockFill" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="var(--warm)" stopOpacity={0.35} />
                                    <stop offset="95%" stopColor="var(--warm-soft)" stopOpacity={0.08} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                formatter={(value) => [formatMetric(Number(value), 4), "Absolute change"]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Area type="monotone" dataKey="absChange" stroke="var(--warm)" fill="url(#shockFill)" strokeWidth={2.2} dot={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="guidance-grid">
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Variance-stabilizing transforms</div>
                    <div className="guidance-list">
                        {volatilityTransformSuggestions.map((hint) => (
                            <div key={hint.key} className="guidance-item">
                                <div className="summary-card-title">{hint.label}</div>
                                <div className="automation-flag-row">
                                    <span className="header-chip">{hint.suitability}</span>
                                </div>
                                <div className="summary-card-copy">{hint.detail}</div>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Transform context</div>
                    <div className="guidance-list">
                        <div className="guidance-item">
                            <div className="summary-card-title">Preferred transform</div>
                            <div className="summary-card-copy">
                                {volatilityDiagnostics.varianceTransformGuidance?.preferredTransform ?? "none"} with skewness {formatMetric(volatilityDiagnostics.varianceTransformGuidance?.skewness ?? 0, 3)} and minimum value {formatMetric(volatilityDiagnostics.varianceTransformGuidance?.minValue ?? 0, 4)}.
                            </div>
                        </div>
                        <div className="guidance-item">
                            <div className="summary-card-title">Zero mass</div>
                            <div className="summary-card-copy">
                                Zero share {formatMetric((volatilityDiagnostics.varianceTransformGuidance?.zeroShare ?? 0) * 100, 1)}% determines whether Box-Cox is feasible or whether log1p is safer.
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <p className="lead-copy">
                Volatility belongs in the preparation workspace because it directly affects transform choice, scaling strategy, and how defensible it is to stabilize the target before modeling.
            </p>
        </Panel>
    );
}

export function PatchingPanel({ status, errorMessage, patchingDiagnostics, activeWindow, onApplyPatchGeometry }) {
    const patchScaleData = patchingDiagnostics?.scaleBandData ?? [];
    const effectivePatchCount = patchingDiagnostics?.available
        ? Math.max(1, Math.floor(Math.max(activeWindow - patchingDiagnostics.recommendedPatchLen, 0) / Math.max(1, patchingDiagnostics.recommendedStride)) + 1)
        : 0;

    if (status === "loading") {
        return (
            <Panel title="Patching" kicker="Preparation lab">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Patching" kicker="Preparation lab">
                <p className="lead-copy">
                    Patching analysis unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!patchingDiagnostics?.available) {
        return (
            <Panel title="Patching" kicker="Preparation lab">
                <p className="lead-copy">
                    Patching analysis needs a longer signal before patch geometry and multi-scale usefulness can be estimated reliably.
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Patching" kicker="Preparation lab">
            <div className="automation-flag-row">
                <button
                    type="button"
                    className="solid-button"
                    onClick={() => onApplyPatchGeometry?.(patchingDiagnostics)}
                >
                    Apply patch geometry
                </button>
                <span className="header-chip">
                    Copies L {patchingDiagnostics.recommendedPatchLen} and S {patchingDiagnostics.recommendedStride} into Step 03.
                </span>
            </div>

            <div className="summary-card-grid benchmark-grid spectral-grid">
                <div className={`summary-card ${patchingDiagnostics.patchingLabel === "good" ? "summary-card-highlight" : ""}`}>
                    <div className="summary-card-title">Patch suitability</div>
                    <div className="architecture-name">{patchingDiagnostics.patchingLabel}</div>
                    <div className="summary-card-copy">
                        Score {patchingDiagnostics.patchingScore}/100 with repeatability {formatMetric(patchingDiagnostics.repeatability, 3)} and roughness {formatMetric(patchingDiagnostics.roughness, 3)}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Suggested patch length</div>
                    <div className="architecture-name">L {patchingDiagnostics.recommendedPatchLen}</div>
                    <div className="summary-card-copy">
                        Anchored to the dominant cycle around period {formatMetric(patchingDiagnostics.dominantPeriod, 1)}.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Suggested stride</div>
                    <div className="architecture-name">S {patchingDiagnostics.recommendedStride}</div>
                    <div className="summary-card-copy">
                        Conservative overlap to preserve local transitions between tokens.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Patches per window</div>
                    <div className="architecture-name">{effectivePatchCount}</div>
                    <div className="summary-card-copy">
                        For the active context window of {activeWindow} steps.
                    </div>
                </div>
                <div className="summary-card">
                    <div className="summary-card-title">Multi-scale value</div>
                    <div className="architecture-name">{patchingDiagnostics.multiscaleLabel}</div>
                    <div className="summary-card-copy">
                        Score {patchingDiagnostics.multiscaleScore}/100 based on energy spread across short, medium, and long scales.
                    </div>
                </div>
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell stable-chart-shell">
                    <div className="mini-chart-title">Scale energy allocation</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart
                            data={patchScaleData}
                            margin={{ top: 8, right: 10, left: 0, bottom: 8 }}
                        >
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="band" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[0, 1]} />
                            <Tooltip
                                formatter={(value, _name, payload) => [
                                    `${formatMetric(Number(value) * 100, 1)}%`,
                                    payload?.payload?.detail ?? "Energy share",
                                ]}
                                labelFormatter={(label) => `${label} scale`}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                                wrapperStyle={{ pointerEvents: "none" }}
                            />
                            <Bar
                                dataKey="share"
                                fill="var(--accent)"
                                radius={[6, 6, 0, 0]}
                                isAnimationActive={false}
                                activeBar={false}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Patch candidates</div>
                    <div className="guidance-list">
                        <div className="guidance-item">
                            <div className="summary-card-title">Recommended set</div>
                            <div className="automation-flag-row">
                                {patchingDiagnostics.recommendedPatchSet.map((candidate) => (
                                    <span key={candidate} className="header-chip">L {candidate}</span>
                                ))}
                            </div>
                            <div className="summary-card-copy">
                                Start with the middle candidate, then compare shorter and longer tokens before widening the transformer stack.
                            </div>
                        </div>
                        <div className="guidance-item">
                            <div className="summary-card-title">Architecture hint</div>
                            <div className="summary-card-copy">
                                {patchingDiagnostics.multiscaleLabel === "high"
                                    ? "A multi-scale encoder is likely worthwhile because the signal carries material structure at several separated periods."
                                    : patchingDiagnostics.multiscaleLabel === "moderate"
                                        ? "A two-scale setup is worth benchmarking if a single patch size underfits either short bursts or slower cycles."
                                        : "One dominant patch size should be enough for the first transformer benchmark; multi-scale complexity is likely optional."}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="guidance-grid">
                <div className="chart-shell guidance-shell">
                    <div className="mini-chart-title">Patching notes</div>
                    <div className="guidance-list">
                        {patchingDiagnostics.notes.map((note) => (
                            <div key={note.key} className="guidance-item">
                                <div className="summary-card-title">{note.label}</div>
                                <div className="automation-flag-row">
                                    <span className="header-chip">{note.suitability}</span>
                                </div>
                                <div className="summary-card-copy">{note.detail}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <p className="lead-copy">
                Patching belongs in the preparation workspace because it is not just descriptive. It directly sets transformer token geometry and feeds back into the Step 03 architecture choices.
            </p>
        </Panel>
    );
}

export function FeatureLabPanel({
    status,
    errorMessage,
    data,
    covariates,
    targetColumn,
    analysis,
    decomposition,
}) {
    const [selectedCovariateName, setSelectedCovariateName] = useState(covariates?.[0]?.name ?? "");
    const [lagDepth, setLagDepth] = useState(() => clamp(Math.round((analysis?.dominantAcfLag || 6)), 1, 24));
    const [rollingWindow, setRollingWindow] = useState(6);
    const [includeCalendar, setIncludeCalendar] = useState(true);
    const [includeInteractions, setIncludeInteractions] = useState(true);
    const [includeDecomposition, setIncludeDecomposition] = useState(true);
    const [selectedFeatureKey, setSelectedFeatureKey] = useState("");

    useEffect(() => {
        if (!covariates?.length) {
            setSelectedCovariateName("");
            return;
        }
        const exists = covariates.some((item) => item.name === selectedCovariateName);
        if (!exists) {
            setSelectedCovariateName(covariates[0].name);
        }
    }, [covariates, selectedCovariateName]);

    const featureLab = useMemo(() => buildFeatureLabResults({
        data,
        covariates,
        selectedCovariateName,
        lagDepth,
        rollingWindow,
        includeCalendar,
        includeInteractions,
        includeDecomposition,
        analysis,
        decomposition,
    }), [data, covariates, selectedCovariateName, lagDepth, rollingWindow, includeCalendar, includeInteractions, includeDecomposition, analysis, decomposition]);

    useEffect(() => {
        if (!featureLab.results.length) {
            setSelectedFeatureKey("");
            return;
        }
        if (!featureLab.results.some((item) => item.key === selectedFeatureKey)) {
            setSelectedFeatureKey(featureLab.results[0].key);
        }
    }, [featureLab.results, selectedFeatureKey]);

    const activeResult = featureLab.results.find((item) => item.key === selectedFeatureKey) ?? featureLab.results[0] ?? null;
    const previewData = useMemo(
        () => (activeResult ? buildFeaturePreviewData(data, activeResult.values) : []),
        [data, activeResult],
    );

    if (status === "loading") {
        return (
            <Panel title="Feature Lab" kicker="Preparation lab">
                <SkeletonBlock pills={4} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Feature Lab" kicker="Preparation lab">
                <p className="lead-copy">
                    Feature lab unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Feature Lab" kicker="Preparation lab">
            <div className="field-grid feature-lab-control-grid">
                <SelectField
                    label="Primary covariate"
                    value={selectedCovariateName}
                    onChange={setSelectedCovariateName}
                    options={covariates?.length
                        ? covariates.map((item) => ({ value: item.name, label: item.name }))
                        : [{ value: "", label: "No numeric covariates detected" }]}
                    hint="Lagged and rolling drivers are generated from this selected numeric series."
                />
                <NumberField
                    label="Feature lag depth"
                    value={lagDepth}
                    onChange={setLagDepth}
                    min={1}
                    max={48}
                    hint="Used for seasonal-style lag candidates and quick ablation screening."
                />
                <NumberField
                    label="Rolling window"
                    value={rollingWindow}
                    onChange={setRollingWindow}
                    min={2}
                    max={48}
                    hint="Backward-looking window used for moving summary features."
                />
                <div className="callout dataset-callout">
                    <div className="callout-title">Sandbox scope</div>
                    <div className="callout-copy">
                        This lab engineers candidate drivers from covariates, timestamps, and decomposition components, then scores each one with a quick one-step baseline ablation. It is a fast screening tool, not a replacement for the full validation workspace.
                    </div>
                </div>
            </div>

            <div className="field-grid feature-lab-toggle-grid">
                <ToggleField
                    label="Calendar encodings"
                    checked={includeCalendar}
                    onChange={setIncludeCalendar}
                    hint="Weekday and month cyclic encodings when timestamps are parseable."
                />
                <ToggleField
                    label="Interaction features"
                    checked={includeInteractions}
                    onChange={setIncludeInteractions}
                    hint="Interactions between target lag history and the selected covariate."
                />
                <ToggleField
                    label="Decomposition drivers"
                    checked={includeDecomposition}
                    onChange={setIncludeDecomposition}
                    hint="Lagged trend and seasonal components derived from the target decomposition."
                />
            </div>

            {featureLab.results.length > 0 ? (
                <>
                    <div className="summary-card-grid benchmark-grid spectral-grid">
                        <div className="summary-card summary-card-highlight">
                            <div className="summary-card-title">Top feature</div>
                            <div className="architecture-name">{activeResult?.label ?? "-"}</div>
                            <div className="summary-card-copy">
                                {activeResult?.group ?? "-"} with RMSE delta {activeResult?.rmseGainLabel ?? "0.0000"} on the quick baseline screen.
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Candidate count</div>
                            <div className="architecture-name">{featureLab.results.length}</div>
                            <div className="summary-card-copy">
                                {featureLab.candidates.length} engineered candidates were generated before model-fit screening.
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Baseline RMSE</div>
                            <div className="architecture-name">{formatMetric(activeResult?.baselineRmse ?? 0, 4)}</div>
                            <div className="summary-card-copy">
                                One-step autoregressive reference using target lags only.
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-card-title">Ablation gain</div>
                            <div className="architecture-name">{activeResult?.improvementPercentLabel ?? "0.0%"}</div>
                            <div className="summary-card-copy">
                                Relative RMSE improvement after adding the selected engineered driver.
                            </div>
                        </div>
                    </div>

                    <div className="summary-card-grid exogenous-grid">
                        {featureLab.results.slice(0, 6).map((result) => (
                            <button
                                key={result.key}
                                type="button"
                                className={`summary-card summary-card-button ${activeResult?.key === result.key ? "summary-card-highlight" : ""}`}
                                onClick={() => setSelectedFeatureKey(result.key)}
                            >
                                <div className="summary-card-title">{result.group}</div>
                                <div className="architecture-name">{result.label}</div>
                                <div className="summary-card-copy">
                                    RMSE {result.rmseGainLabel} · coverage {formatMetric(result.coverage * 100, 1)}% · corr {formatMetric(result.correlation, 3)}.
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="benchmark-chart-grid">
                        <div className="chart-shell benchmark-chart-shell">
                            <div className="mini-chart-title">Ablation impact by candidate</div>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={featureLab.results.slice(0, 8)}>
                                    <CartesianGrid stroke="var(--grid)" vertical={false} />
                                    <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} interval={0} angle={-16} textAnchor="end" height={68} />
                                    <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                    <ReferenceLine y={0} stroke="var(--grid)" />
                                    <Tooltip
                                        formatter={(value, name) => [
                                            formatMetric(Number(value), name === "improvementRate" ? 3 : 4),
                                            name === "rmseGain" ? "RMSE delta" : name === "maeGain" ? "MAE delta" : "Improvement rate",
                                        ]}
                                        labelFormatter={(label) => `${label}`}
                                        contentStyle={{
                                            backgroundColor: "var(--panel)",
                                            borderColor: "var(--border)",
                                            borderRadius: 16,
                                            color: "var(--text)",
                                        }}
                                    />
                                    <Bar dataKey="rmseGain" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="chart-shell benchmark-chart-shell">
                            <div className="mini-chart-title">Target vs selected feature (normalized)</div>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={previewData}>
                                    <CartesianGrid stroke="var(--grid)" vertical={false} />
                                    <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                                    <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                        formatter={(value, name) => [formatMetric(Number(value), 3), name === "target" ? targetColumn || "target" : activeResult?.label ?? "feature"]}
                                        contentStyle={{
                                            backgroundColor: "var(--panel)",
                                            borderColor: "var(--border)",
                                            borderRadius: 16,
                                            color: "var(--text)",
                                        }}
                                    />
                                    <Line type="monotone" dataKey="target" stroke="var(--warm)" strokeWidth={2.3} dot={false} />
                                    <Line type="monotone" dataKey="feature" stroke="var(--secondary)" strokeWidth={2.2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="guidance-grid">
                        <div className="chart-shell guidance-shell">
                            <div className="mini-chart-title">Selected feature detail</div>
                            <div className="guidance-list">
                                <div className="guidance-item">
                                    <div className="summary-card-title">Description</div>
                                    <div className="summary-card-copy">{activeResult?.description ?? "No active feature selected."}</div>
                                </div>
                                <div className="guidance-item">
                                    <div className="summary-card-title">Coverage and correlation</div>
                                    <div className="summary-card-copy">
                                        Coverage {formatMetric((activeResult?.coverage ?? 0) * 100, 1)}% with raw correlation {formatMetric(activeResult?.correlation ?? 0, 3)} against the target.
                                    </div>
                                </div>
                                <div className="guidance-item">
                                    <div className="summary-card-title">Baseline comparison</div>
                                    <div className="summary-card-copy">
                                        Baseline RMSE {formatMetric(activeResult?.baselineRmse ?? 0, 4)} vs augmented RMSE {formatMetric(activeResult?.featureRmse ?? 0, 4)} over {activeResult?.testCount ?? 0} holdout rows.
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="chart-shell guidance-shell">
                            <div className="mini-chart-title">Feature groups in this run</div>
                            <div className="guidance-list">
                                {Array.from(new Map(featureLab.results.map((item) => [item.group, item])).values()).map((result) => (
                                    <div key={result.group} className="guidance-item">
                                        <div className="summary-card-title">{result.group}</div>
                                        <div className="summary-card-copy">
                                            Best candidate: {featureLab.results.find((item) => item.group === result.group)?.label ?? "-"}.
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <p className="lead-copy">
                        Feature Lab screens engineered drivers with a quick autoregressive ablation model so you can see whether lagged covariates, rolling summaries, calendar encodings, interactions, or decomposition-derived drivers are worth carrying into a more serious benchmark.
                    </p>
                </>
            ) : (
                <p className="lead-copy">
                    Feature Lab could not build enough valid candidates from the current series. Load at least one numeric covariate or keep parseable timestamps/decomposition enabled so the sandbox has usable feature families to test.
                </p>
            )}
        </Panel>
    );
}

export function ExplorePanel({
    data,
    status,
    errorMessage,
    datasetSummary,
    headers,
    covariates,
    targetColumn,
    timestampColumn,
    analysis,
    acfCurve,
    pacfCurve,
    decomposition,
    exogenousScreening,
    grangerDiagnostics,
    spectralDiagnostics,
    calendarDiagnostics,
    onApplyLagWindow,
    activeTabOverride,
    onActiveTabChange,
}) {
    const requestedActiveTab = activeTabOverride ?? "series";
    const activeTab = EXPLORE_GROUP_BY_TAB[requestedActiveTab] ? requestedActiveTab : "series";
    const setActiveTab = (tab) => onActiveTabChange?.(tab);
    const [selectedExogenous, setSelectedExogenous] = useState("");
    const [grangerDirection, setGrangerDirection] = useState("predictorToTarget");
    const [rememberedTabsByGroup, setRememberedTabsByGroup] = useState(() => Object.fromEntries(EXPLORE_TAB_GROUPS.map((group) => [group.key, group.tabs[0]])));
    const stats = useMemo(() => buildExploreStats(data, datasetSummary), [data, datasetSummary]);
    const decompositionView = useMemo(() => decomposition ?? buildDecomposition(data, analysis), [decomposition, data, analysis]);
    const exogenousColumns = headers.filter((header) => header !== targetColumn && header !== timestampColumn).slice(0, 2);
    const activeExogenousName = selectedExogenous || exogenousScreening?.[0]?.name || covariates?.[0]?.name || "";
    const activeCovariate = covariates?.find((item) => item.name === activeExogenousName) ?? null;
    const activeExogenousDiagnostics = exogenousScreening?.find((item) => item.name === activeExogenousName) ?? null;
    const activeGrangerDirection = grangerDiagnostics?.directions?.[grangerDirection] ?? grangerDiagnostics;
    const activeGrangerDiagnostics = activeGrangerDirection?.results?.find((item) => item.name === activeExogenousName)
        ?? activeGrangerDirection?.results?.[0]
        ?? null;
    const comparisonData = useMemo(() => buildComparisonData(data, activeCovariate?.points ?? []), [data, activeCovariate]);
    const spectralComparisonData = useMemo(() => buildSpectralComparisonData(spectralDiagnostics, acfCurve), [spectralDiagnostics, acfCurve]);
    const spectralAgreement = useMemo(() => buildSpectralAgreement(spectralDiagnostics, analysis.acfPeaks), [spectralDiagnostics, analysis.acfPeaks]);
    const spectralSuggestions = useMemo(() => buildSpectralSuggestions(spectralDiagnostics, analysis.suggestedLagWindow), [spectralDiagnostics, analysis.suggestedLagWindow]);
    const calendarProfiles = useMemo(() => buildCalendarProfiles(calendarDiagnostics), [calendarDiagnostics]);
    const exogenousLagWindows = activeExogenousDiagnostics?.leadLagStability?.windows ?? [];
    const [activeWindow, setActiveWindow] = useState(analysis.suggestedLagWindow);
    useEffect(() => {
        setActiveWindow(analysis.suggestedLagWindow);
    }, [analysis.suggestedLagWindow]);
    const activeExploreGroup = EXPLORE_TAB_GROUPS.find((group) => group.key === EXPLORE_GROUP_BY_TAB[activeTab]) ?? EXPLORE_TAB_GROUPS[0];

    useEffect(() => {
        const groupKey = EXPLORE_GROUP_BY_TAB[activeTab];
        if (!groupKey) {
            return;
        }
        setRememberedTabsByGroup((current) => {
            if (current[groupKey] === activeTab) {
                return current;
            }
            return {
                ...current,
                [groupKey]: activeTab,
            };
        });
    }, [activeTab]);

    if (status === "loading") {
        return (
            <Panel title="Explore" kicker="Data explorer">
                <SkeletonBlock title chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Explore" kicker="Data explorer">
                <p className="lead-copy">
                    Exploration unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    const lagSuggestions = uniqueSuggestions([
        {
            label: `PACF L${analysis.dominantPacfLag}`,
            detail: `window ${Math.max(24, Math.min(96, analysis.dominantPacfLag * 3))}`,
            window: Math.max(24, Math.min(96, analysis.dominantPacfLag * 3)),
        },
        {
            label: `ACF L${analysis.dominantAcfLag}`,
            detail: `window ${Math.max(24, Math.min(96, analysis.dominantAcfLag * 2))}`,
            window: Math.max(24, Math.min(96, analysis.dominantAcfLag * 2)),
        },
        {
            label: "Recommended",
            detail: `window ${analysis.suggestedLagWindow}`,
            window: analysis.suggestedLagWindow,
        },
    ]);

    return (
        <Panel title="Explore" kicker="Data explorer">
            <div className="explore-nav-shell">
                <div className="explore-section-row">
                    {EXPLORE_TAB_GROUPS.map((group) => (
                        <button
                            key={group.key}
                            type="button"
                            className={`explore-section-tab ${activeExploreGroup.key === group.key ? "explore-section-tab-active" : ""}`}
                            onClick={() => setActiveTab(rememberedTabsByGroup[group.key] ?? group.tabs[0])}
                        >
                            {group.label}
                        </button>
                    ))}
                </div>

                <div className="explore-section-caption">
                    <div className="explore-section-summary-row">
                        <span className="header-chip">{activeExploreGroup.label}</span>
                        <span className="header-chip">{activeExploreGroup.tabs.length} views</span>
                    </div>
                    <p className="explore-section-copy">{activeExploreGroup.description}</p>
                </div>

                <div className="explore-tab-row">
                    {activeExploreGroup.tabs.map((tab) => (
                        <button
                            key={tab}
                            type="button"
                            className={`explore-tab ${activeTab === tab ? "explore-tab-active" : ""}`}
                            onClick={() => setActiveTab(tab)}
                        >
                            {EXPLORE_TAB_META[tab]?.label ?? tab}
                        </button>
                    ))}
                </div>
            </div>

            <div className="explore-stat-grid">
                {stats.map((item) => (
                    <StatPill key={item.label} label={item.label} value={item.value} tone={item.tone} />
                ))}
            </div>

            <div className="explore-chart-shell">
                <div className="explore-toolbar">
                    <div className="header-chip-row explore-series-chips">
                        <span className="header-chip header-chip-active">target({targetColumn})</span>
                        {exogenousColumns.map((header) => (
                            <button
                                key={header}
                                type="button"
                                className={`header-chip header-chip-button ${activeExogenousName === header ? "header-chip-active" : "header-chip-muted"}`}
                                onClick={() => {
                                    setSelectedExogenous(header);
                                    setActiveTab("exogenous");
                                }}
                            >
                                {header}
                            </button>
                        ))}
                    </div>
                    <div className="header-chip-row explore-series-chips">
                        <span className="header-chip">{analysis.stationarityLabel}</span>
                        <span className="header-chip">freq {analysis.frequencyHint}</span>
                    </div>
                </div>

                {activeTab === "series" ? (
                    <>
                        <div className="chart-shell explore-chart-body">
                            <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={data}>
                                    <CartesianGrid stroke="var(--grid)" vertical={false} />
                                    <XAxis dataKey="month" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                                    <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                        contentStyle={{
                                            backgroundColor: "var(--panel)",
                                            borderColor: "var(--border)",
                                            borderRadius: 16,
                                            color: "var(--text)",
                                        }}
                                    />
                                    <Line type="monotone" dataKey="value" stroke="var(--warm)" strokeWidth={2.4} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <p className="lead-copy">
                            Raw series view across {data.length} retained observations. The current target is {targetColumn}, with stationarity classified as {analysis.stationarityLabel} and trend labeled {analysis.trendLabel}.
                        </p>
                    </>
                ) : null}

                {activeTab === "decomposition" ? (
                    <>
                        <div className="explore-chart-body decomposition-grid">
                            <DecompositionSubplot
                                title="Observed"
                                data={decompositionView.data}
                                dataKey="observed"
                                color="var(--warm)"
                            />
                            <DecompositionSubplot
                                title="Trend"
                                data={decompositionView.data}
                                dataKey="trend"
                                color="var(--secondary)"
                            />
                            <DecompositionSubplot
                                title="Seasonality"
                                data={decompositionView.data}
                                dataKey="seasonal"
                                color="var(--accent)"
                            />
                            <DecompositionSubplot
                                title="Residual"
                                data={decompositionView.data}
                                dataKey="residual"
                                color="var(--cool)"
                                showXAxis
                            />
                        </div>
                        <div className="automation-flag-row">
                            <span className="header-chip">periods {decompositionView.periods.join(", ")}</span>
                            <span className="header-chip">seasonality strength {formatMetric((decompositionView.strength ?? 0) * 100, 1)}%</span>
                            <span className="header-chip">residual std {formatMetric(decompositionView.residualStd)}</span>
                            <span className="header-chip">{analysis.needsDetrend ? "detrend recommended" : "detrend optional"}</span>
                        </div>
                        {decompositionView.components?.length ? (
                            <div className="automation-flag-row">
                                {decompositionView.components.map((component) => (
                                    <span key={component.period} className="header-chip">
                                        P{component.period}: {formatMetric(component.strength * 100, 1)}%
                                    </span>
                                ))}
                            </div>
                        ) : null}
                        <p className="lead-copy">
                            MSTL-style decomposition derived from the live series using the strongest seasonal lags found by autocorrelation. The four subplots separate observed signal, low-frequency trend, aggregated seasonal structure, and residual noise.
                        </p>
                    </>
                ) : null}

                {activeTab === "autocorrelation" ? (
                    <>
                        <div className="diagnostics-grid explore-chart-body">
                            <div className="chart-shell short-chart">
                                <div className="mini-chart-title">ACF</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <BarChart data={acfCurve}>
                                        <CartesianGrid stroke="var(--grid)" vertical={false} />
                                        <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                                        <Tooltip
                                            formatter={(value) => [Number(value).toFixed(3), "ACF"]}
                                            contentStyle={{
                                                backgroundColor: "var(--panel)",
                                                borderColor: "var(--border)",
                                                borderRadius: 16,
                                                color: "var(--text)",
                                            }}
                                        />
                                        <ReferenceLine y={Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                                        <ReferenceLine y={-Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                                        <ReferenceLine x={analysis.dominantAcfLag} stroke="var(--warm)" strokeDasharray="4 4" />
                                        <Bar dataKey="acf" fill="var(--warm)" radius={[6, 6, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="chart-shell short-chart">
                                <div className="mini-chart-title">PACF</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <BarChart data={pacfCurve}>
                                        <CartesianGrid stroke="var(--grid)" vertical={false} />
                                        <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                        <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                                        <Tooltip
                                            formatter={(value) => [Number(value).toFixed(3), "PACF"]}
                                            contentStyle={{
                                                backgroundColor: "var(--panel)",
                                                borderColor: "var(--border)",
                                                borderRadius: 16,
                                                color: "var(--text)",
                                            }}
                                        />
                                        <ReferenceLine y={Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                                        <ReferenceLine y={-Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                                        <ReferenceLine x={analysis.dominantPacfLag} stroke="var(--accent)" strokeDasharray="4 4" />
                                        <Bar dataKey="pacf" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <div className="lag-suggestion-row">
                            {lagSuggestions.map((suggestion) => (
                                <button
                                    key={`${suggestion.label}-${suggestion.window}`}
                                    type="button"
                                    className={`lag-suggestion ${activeWindow === suggestion.window ? "lag-suggestion-active" : ""}`}
                                    onClick={() => {
                                        setActiveWindow(suggestion.window);
                                        onApplyLagWindow(suggestion.window);
                                    }}
                                >
                                    <strong>{suggestion.label}</strong>
                                    <span>{suggestion.detail}</span>
                                </button>
                            ))}
                        </div>
                        <p className="lead-copy">
                            ACF highlights repeating structure near lag {analysis.dominantAcfLag}, while PACF isolates direct dependence near lag {analysis.dominantPacfLag}. Apply one of these windows directly to the preprocessing context when you want a faster exploratory loop.
                        </p>
                    </>
                ) : null}

                {activeTab === "exogenous" ? (
                    <>
                        {exogenousScreening?.length ? (
                            <>
                                <div className="summary-card-grid exogenous-grid">
                                    {exogenousScreening.slice(0, 6).map((feature) => (
                                        <button
                                            key={feature.name}
                                            type="button"
                                            className={`summary-card summary-card-button ${activeExogenousName === feature.name ? "summary-card-highlight" : ""}`}
                                            onClick={() => setSelectedExogenous(feature.name)}
                                        >
                                            <div className="summary-card-title">{feature.name}</div>
                                            <div className="architecture-name">{feature.recommendation}</div>
                                            <div className="summary-card-copy">
                                                Score {feature.score} · lag0 {feature.lag0Correlation} · best lag {feature.bestLag} · {feature.leadLagStability?.label ?? "lag pending"} lag · cointegration {feature.cointegration?.label ?? "pending"}.
                                            </div>
                                        </button>
                                    ))}
                                </div>
                                {activeExogenousDiagnostics ? (
                                    <div className="summary-card-grid benchmark-grid spectral-grid">
                                        <div className="summary-card summary-card-highlight">
                                            <div className="summary-card-title">Lead-lag direction</div>
                                            <div className="architecture-name">L{activeExogenousDiagnostics.bestLag}</div>
                                            <div className="summary-card-copy">
                                                {activeExogenousDiagnostics.direction} with best correlation {formatMetric(activeExogenousDiagnostics.bestCorrelation, 3)}.
                                            </div>
                                        </div>
                                        <div className="summary-card">
                                            <div className="summary-card-title">Lag stability</div>
                                            <div className="architecture-name">{activeExogenousDiagnostics.leadLagStability?.label ?? "unknown"}</div>
                                            <div className="summary-card-copy">
                                                Stable-share {formatMetric((activeExogenousDiagnostics.leadLagStability?.stableShare ?? 0) * 100, 1)}% with lag std {formatMetric(activeExogenousDiagnostics.leadLagStability?.lagStd ?? 0, 2)}.
                                            </div>
                                        </div>
                                        <div className="summary-card">
                                            <div className="summary-card-title">Cointegration</div>
                                            <div className="architecture-name">{activeExogenousDiagnostics.cointegration?.label ?? "unknown"}</div>
                                            <div className="summary-card-copy">
                                                Residual scale {formatMetric(activeExogenousDiagnostics.cointegration?.residualScale ?? 0, 3)} · ADF {formatMetric(activeExogenousDiagnostics.cointegration?.residualAdfStatistic ?? 0, 3)}.
                                            </div>
                                        </div>
                                        <div className="summary-card">
                                            <div className="summary-card-title">Coverage</div>
                                            <div className="architecture-name">{formatMetric(activeExogenousDiagnostics.coverage, 1)}%</div>
                                            <div className="summary-card-copy">
                                                Missing rows {activeExogenousDiagnostics.missingCount} after aligning covariate history with the target.
                                            </div>
                                        </div>
                                    </div>
                                ) : null}
                                <div className="chart-shell explore-chart-body">
                                    <div className="mini-chart-title">
                                        Target vs {activeExogenousName} (normalized)
                                    </div>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart data={comparisonData}>
                                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                                            <XAxis dataKey="month" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={20} />
                                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                            <Tooltip
                                                labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? ""}
                                                contentStyle={{
                                                    backgroundColor: "var(--panel)",
                                                    borderColor: "var(--border)",
                                                    borderRadius: 16,
                                                    color: "var(--text)",
                                                }}
                                            />
                                            <Line type="monotone" dataKey="target" stroke="var(--warm)" strokeWidth={2.2} dot={false} />
                                            <Line type="monotone" dataKey="covariate" stroke="var(--accent)" strokeWidth={2.2} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                                {exogenousLagWindows.length > 0 ? (
                                    <div className="chart-shell explore-chart-body">
                                        <div className="mini-chart-title">Rolling lead-lag stability</div>
                                        <ResponsiveContainer width="100%" height={280}>
                                            <BarChart data={exogenousLagWindows}>
                                                <CartesianGrid stroke="var(--grid)" vertical={false} />
                                                <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                <Tooltip
                                                    formatter={(value, name) => [
                                                        Number(value).toFixed(name === "bestLag" ? 1 : 3),
                                                        name === "bestLag" ? "Best lag" : name === "bestCorrelation" ? "Best correlation" : "Window score",
                                                    ]}
                                                    labelFormatter={(label) => `${activeExogenousName} · ${label}`}
                                                    contentStyle={{
                                                        backgroundColor: "var(--panel)",
                                                        borderColor: "var(--border)",
                                                        borderRadius: 16,
                                                        color: "var(--text)",
                                                    }}
                                                />
                                                <ReferenceLine y={0} stroke="var(--grid)" />
                                                <Bar dataKey="bestLag" fill="var(--secondary)" radius={[6, 6, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                ) : null}
                                <p className="lead-copy">
                                    Exogenous screening now combines global correlation, rolling lag stability, and a simple cointegration heuristic. Positive best lags indicate the covariate leads the target, while stable lag windows and stationary residual spreads are stronger evidence that the relationship is durable enough to promote.
                                </p>
                            </>
                        ) : (
                            <p className="lead-copy">
                                No usable exogenous numeric columns were found after aligning the dataset with the target series.
                            </p>
                        )}
                    </>
                ) : null}

                {activeTab === "granger" ? (
                    <>
                        {grangerDiagnostics?.available ? (
                            <>
                                <div className="metric-toggle-row granger-direction-toggle-row">
                                    <button
                                        type="button"
                                        className={`metric-toggle ${grangerDirection === "predictorToTarget" ? "metric-toggle-active" : ""}`}
                                        onClick={() => setGrangerDirection("predictorToTarget")}
                                    >
                                        Covariate -&gt; Target
                                    </button>
                                    <button
                                        type="button"
                                        className={`metric-toggle ${grangerDirection === "targetToPredictor" ? "metric-toggle-active" : ""}`}
                                        onClick={() => setGrangerDirection("targetToPredictor")}
                                    >
                                        Target -&gt; Covariate
                                    </button>
                                </div>

                                {activeGrangerDirection?.available ? (
                                    <>
                                        <div className="chart-shell guidance-shell granger-matrix-shell">
                                            <div className="mini-chart-title">Pairwise matrix</div>
                                            <div className="granger-matrix-caption">
                                                Direction shown is {grangerDirection === "predictorToTarget" ? "covariate to target" : "target to covariate"}. Click any column to pin that driver in the detailed charts below.
                                            </div>
                                            <div
                                                className="granger-matrix-grid"
                                                style={{ gridTemplateColumns: `minmax(148px, 1.15fr) repeat(${Math.min(activeGrangerDirection.results.length, 6)}, minmax(112px, 1fr))` }}
                                            >
                                                <div className="granger-matrix-corner">{grangerDirection === "predictorToTarget" ? "Predictor > target" : "Target > predictor"}</div>
                                                {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                    <button
                                                        key={`matrix-column-${feature.name}`}
                                                        type="button"
                                                        className={`granger-matrix-header ${activeGrangerDiagnostics?.name === feature.name ? "granger-matrix-header-active" : ""}`}
                                                        onClick={() => setSelectedExogenous(feature.name)}
                                                    >
                                                        {feature.name}
                                                    </button>
                                                ))}

                                                <div className="granger-matrix-row-label">Evidence score</div>
                                                {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                    <button
                                                        key={`matrix-score-${feature.name}`}
                                                        type="button"
                                                        className={`granger-matrix-cell granger-matrix-cell-${grangerMatrixTone(feature)} ${activeGrangerDiagnostics?.name === feature.name ? "granger-matrix-cell-active" : ""}`}
                                                        onClick={() => setSelectedExogenous(feature.name)}
                                                    >
                                                        <strong>{formatMetric(feature.score, 1)}</strong>
                                                        <span>{feature.significance}</span>
                                                    </button>
                                                ))}

                                                <div className="granger-matrix-row-label">Best lag</div>
                                                {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                    <button
                                                        key={`matrix-lag-${feature.name}`}
                                                        type="button"
                                                        className={`granger-matrix-cell ${activeGrangerDiagnostics?.name === feature.name ? "granger-matrix-cell-active" : ""}`}
                                                        onClick={() => setSelectedExogenous(feature.name)}
                                                    >
                                                        <strong>L{feature.bestLag}</strong>
                                                        <span>{feature.mode}</span>
                                                    </button>
                                                ))}

                                                <div className="granger-matrix-row-label">p-value</div>
                                                {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                    <button
                                                        key={`matrix-pvalue-${feature.name}`}
                                                        type="button"
                                                        className={`granger-matrix-cell granger-matrix-cell-${grangerMatrixTone(feature)} ${activeGrangerDiagnostics?.name === feature.name ? "granger-matrix-cell-active" : ""}`}
                                                        onClick={() => setSelectedExogenous(feature.name)}
                                                    >
                                                        <strong>{formatMetric(feature.pValue, 4)}</strong>
                                                        <span>F {formatMetric(feature.fStatistic, 2)}</span>
                                                    </button>
                                                ))}

                                                <div className="granger-matrix-row-label">Coverage</div>
                                                {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                    <button
                                                        key={`matrix-coverage-${feature.name}`}
                                                        type="button"
                                                        className={`granger-matrix-cell ${activeGrangerDiagnostics?.name === feature.name ? "granger-matrix-cell-active" : ""}`}
                                                        onClick={() => setSelectedExogenous(feature.name)}
                                                    >
                                                        <strong>{formatMetric(feature.coverage, 1)}%</strong>
                                                        <span>{feature.validCount} rows</span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div className="summary-card-grid exogenous-grid">
                                            {activeGrangerDirection.results.slice(0, 6).map((feature) => (
                                                <button
                                                    key={feature.name}
                                                    type="button"
                                                    className={`summary-card summary-card-button ${activeGrangerDiagnostics?.name === feature.name ? "summary-card-highlight" : ""}`}
                                                    onClick={() => setSelectedExogenous(feature.name)}
                                                >
                                                    <div className="summary-card-title">{feature.name}</div>
                                                    <div className="architecture-name">{feature.significance}</div>
                                                    <div className="summary-card-copy">
                                                        Lag {feature.bestLag} · F {formatMetric(feature.fStatistic, 3)} · p {formatMetric(feature.pValue, 4)} · score {formatMetric(feature.score, 1)}.
                                                    </div>
                                                </button>
                                            ))}
                                        </div>

                                        <div className="summary-card-grid benchmark-grid spectral-grid">
                                            <div className={`summary-card ${activeGrangerDiagnostics?.significance === "significant" ? "summary-card-highlight" : ""}`}>
                                                <div className="summary-card-title">Top driver</div>
                                                <div className="architecture-name">{activeGrangerDiagnostics?.name ?? "-"}</div>
                                                <div className="summary-card-copy">
                                                    Best lag {activeGrangerDiagnostics?.bestLag ?? "-"} in {activeGrangerDiagnostics?.mode ?? "level"} mode.
                                                </div>
                                            </div>
                                            <div className="summary-card">
                                                <div className="summary-card-title">Significant links</div>
                                                <div className="architecture-name">{activeGrangerDirection.significantCount}</div>
                                                <div className="summary-card-copy">
                                                    Out of {activeGrangerDirection.testedCount} tested companion series.
                                                </div>
                                            </div>
                                            <div className="summary-card">
                                                <div className="summary-card-title">F-statistic</div>
                                                <div className="architecture-name">{formatMetric(activeGrangerDiagnostics?.fStatistic ?? 0, 3)}</div>
                                                <div className="summary-card-copy">
                                                    Stronger values imply more incremental predictive power from lagged companion history.
                                                </div>
                                            </div>
                                            <div className="summary-card">
                                                <div className="summary-card-title">Aligned coverage</div>
                                                <div className="architecture-name">{formatMetric(activeGrangerDiagnostics?.coverage ?? 0, 1)}%</div>
                                                <div className="summary-card-copy">
                                                    Rows retained after aligning the target with the selected companion series.
                                                </div>
                                            </div>
                                        </div>

                                        {activeGrangerDiagnostics ? (
                                            <div className="benchmark-chart-grid">
                                                <div className="chart-shell benchmark-chart-shell">
                                                    <div className="mini-chart-title">Lag profile for {activeGrangerDiagnostics.name}</div>
                                                    <ResponsiveContainer width="100%" height={280}>
                                                        <LineChart data={activeGrangerDiagnostics.lagTests}>
                                                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                                                            <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                            <Tooltip
                                                                formatter={(value, name) => [
                                                                    formatMetric(Number(value), name === "pValue" ? 4 : 3),
                                                                    name === "fStatistic" ? "F-statistic" : name === "pValue" ? "p-value" : "Delta R²",
                                                                ]}
                                                                labelFormatter={(label) => `${activeGrangerDiagnostics.name} · lag ${label}`}
                                                                contentStyle={{
                                                                    backgroundColor: "var(--panel)",
                                                                    borderColor: "var(--border)",
                                                                    borderRadius: 16,
                                                                    color: "var(--text)",
                                                                }}
                                                            />
                                                            <Line type="monotone" dataKey="fStatistic" stroke="var(--secondary)" strokeWidth={2.4} dot={{ r: 3 }} />
                                                            <Line type="monotone" dataKey="deltaR2" stroke="var(--accent)" strokeWidth={2.2} dot={false} />
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                </div>

                                                <div className="chart-shell benchmark-chart-shell">
                                                    <div className="mini-chart-title">Candidate evidence</div>
                                                    <ResponsiveContainer width="100%" height={280}>
                                                        <BarChart data={activeGrangerDirection.results.slice(0, 8)}>
                                                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                                                            <XAxis dataKey="name" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} interval={0} angle={-18} textAnchor="end" height={64} />
                                                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                            <Tooltip
                                                                formatter={(value, name, payload) => [
                                                                    formatMetric(Number(value), name === "score" ? 1 : 4),
                                                                    name === "score" ? "Evidence score" : "p-value",
                                                                ]}
                                                                labelFormatter={(label) => `${label}`}
                                                                contentStyle={{
                                                                    backgroundColor: "var(--panel)",
                                                                    borderColor: "var(--border)",
                                                                    borderRadius: 16,
                                                                    color: "var(--text)",
                                                                }}
                                                            />
                                                            <Bar dataKey="score" fill="var(--warm)" radius={[6, 6, 0, 0]} />
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        ) : null}

                                        <p className="lead-copy">
                                            Granger causality tests whether lagged history from one series improves one-step prediction of another beyond the response series&apos;s own lagged values. Treat significant links as directional forecasting evidence, not structural causality: they are strongest when the aligned series are stable, sufficiently long, and not dominated by shared calendar effects.
                                        </p>
                                    </>
                                ) : (
                                    <p className="lead-copy">
                                        {activeGrangerDirection?.reason ?? "This Granger direction does not have enough aligned history to estimate stable lag effects."}
                                    </p>
                                )}
                            </>
                        ) : (
                            <p className="lead-copy">
                                {grangerDiagnostics?.reason ?? "Granger causality needs at least one companion numeric series with enough aligned history to test lagged predictive value."}
                            </p>
                        )}
                    </>
                ) : null}

                {activeTab === "spectral" ? (
                    <>
                        {spectralDiagnostics?.available ? (
                            <>
                                <div className="summary-card-grid benchmark-grid spectral-grid">
                                    <div className="summary-card summary-card-highlight">
                                        <div className="summary-card-title">Dominant period</div>
                                        <div className="architecture-name">{spectralDiagnostics.dominantPeriod}</div>
                                        <div className="summary-card-copy">
                                            Frequency {spectralDiagnostics.dominantFrequency} with normalized peak power {formatMetric(spectralDiagnostics.normalizedPeakPower * 100, 1)}%.
                                        </div>
                                    </div>
                                    <div className="summary-card">
                                        <div className="summary-card-title">ACF agreement</div>
                                        <div className="architecture-name">{spectralAgreement.label}</div>
                                        <div className="summary-card-copy">
                                            Shared cycles {spectralAgreement.sharedCount} · dominant delta {spectralAgreement.dominantDelta}.
                                        </div>
                                    </div>
                                    <div className="summary-card">
                                        <div className="summary-card-title">Strongest shared</div>
                                        <div className="architecture-name">{spectralAgreement.strongestSharedPeriod || "-"}</div>
                                        <div className="summary-card-copy">
                                            Best overlap between spectral peaks and ACF seasonal lags.
                                        </div>
                                    </div>
                                    {spectralDiagnostics.peaks.slice(0, 3).map((peak, index) => (
                                        <div key={`${peak.period}-${peak.frequency}`} className="summary-card">
                                            <div className="summary-card-title">Peak {index + 1}</div>
                                            <div className="architecture-name">P {peak.period}</div>
                                            <div className="summary-card-copy">
                                                f {peak.frequency} · power {formatMetric(peak.normalizedPower * 100, 1)}%.
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <div className="chart-shell explore-chart-body">
                                    <div className="mini-chart-title">Power spectrum</div>
                                    <ResponsiveContainer width="100%" height={320}>
                                        <AreaChart data={spectralDiagnostics.spectrum}>
                                            <defs>
                                                <linearGradient id="spectralPower" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="var(--secondary)" stopOpacity={0.35} />
                                                    <stop offset="95%" stopColor="var(--secondary-soft)" stopOpacity={0.06} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                                            <XAxis dataKey="period" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={24} />
                                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                            <Tooltip
                                                formatter={(value, name, payload) => [
                                                    name === "normalizedPower"
                                                        ? `${formatMetric(Number(value) * 100, 2)}%`
                                                        : Number(value).toFixed(4),
                                                    name === "normalizedPower" ? "Normalized power" : "Power",
                                                ]}
                                                labelFormatter={(_label, payload) => {
                                                    const point = payload?.[0]?.payload;
                                                    return point ? `Period ${point.period} · f ${point.frequency}` : "";
                                                }}
                                                contentStyle={{
                                                    backgroundColor: "var(--panel)",
                                                    borderColor: "var(--border)",
                                                    borderRadius: 16,
                                                    color: "var(--text)",
                                                }}
                                            />
                                            <Area type="monotone" dataKey="normalizedPower" stroke="var(--secondary)" fill="url(#spectralPower)" strokeWidth={2.4} dot={false} />
                                            {spectralDiagnostics.peaks.slice(0, 3).map((peak) => (
                                                <ReferenceLine key={`peak-${peak.period}`} x={peak.period} stroke="var(--warm)" strokeDasharray="4 4" />
                                            ))}
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="chart-shell explore-chart-body">
                                    <div className="mini-chart-title">Spectrum vs ACF agreement</div>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart data={spectralComparisonData}>
                                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                                            <XAxis dataKey="period" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} minTickGap={24} />
                                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[0, 1]} />
                                            <Tooltip
                                                formatter={(value, name) => [
                                                    `${formatMetric(Number(value) * 100, 1)}%`,
                                                    name === "spectralPower" ? "Spectral power" : "ACF strength",
                                                ]}
                                                labelFormatter={(label) => `Period ${label}`}
                                                contentStyle={{
                                                    backgroundColor: "var(--panel)",
                                                    borderColor: "var(--border)",
                                                    borderRadius: 16,
                                                    color: "var(--text)",
                                                }}
                                            />
                                            <Line type="monotone" dataKey="spectralPower" stroke="var(--secondary)" strokeWidth={2.4} dot={false} />
                                            <Line type="monotone" dataKey="acfStrength" stroke="var(--warm)" strokeWidth={2.2} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="lag-suggestion-row">
                                    {spectralSuggestions.map((suggestion) => (
                                        <button
                                            key={`${suggestion.label}-${suggestion.window}`}
                                            type="button"
                                            className={`lag-suggestion ${activeWindow === suggestion.window ? "lag-suggestion-active" : ""}`}
                                            onClick={() => {
                                                setActiveWindow(suggestion.window);
                                                onApplyLagWindow(suggestion.window);
                                            }}
                                        >
                                            <strong>{suggestion.label}</strong>
                                            <span>{suggestion.detail}</span>
                                        </button>
                                    ))}
                                </div>
                                <p className="lead-copy">
                                    The spectral view estimates how much variance concentrates in repeating cycles. The agreement view cross-checks those cycles against lag-domain autocorrelation, while the suggestions let you promote strong spectral periods into context-window choices directly.
                                </p>
                            </>
                        ) : (
                            <p className="lead-copy">
                                Spectral analysis needs a longer series before a stable power spectrum can be estimated.
                            </p>
                        )}
                    </>
                ) : null}

                {activeTab === "calendar" ? (
                    calendarDiagnostics?.available ? (
                        <>
                            <div className="summary-card-grid benchmark-grid spectral-grid">
                                <div className="summary-card summary-card-highlight">
                                    <div className="summary-card-title">Strongest profile</div>
                                    <div className="architecture-name">{calendarDiagnostics.strongestProfile}</div>
                                    <div className="summary-card-copy">
                                        Effect share {formatMetric(calendarDiagnostics.strongestEffectShare * 100, 1)}% with an overall {calendarDiagnostics.label} calendar signature.
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="summary-card-title">Time features</div>
                                    <div className="architecture-name">{calendarDiagnostics.recommendation}</div>
                                    <div className="summary-card-copy">
                                        Calendar covariates are {calendarDiagnostics.recommendation === "recommended" ? "worth enabling" : "optional"} for the current series.
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="summary-card-title">Weekend gap</div>
                                    <div className="architecture-name">{formatMetric(calendarDiagnostics.weekendGap, 3)}</div>
                                    <div className="summary-card-copy">
                                        Mean weekend level minus mean weekday level on timestamped observations.
                                    </div>
                                </div>
                                <div className="summary-card">
                                    <div className="summary-card-title">Timestamp coverage</div>
                                    <div className="architecture-name">{calendarDiagnostics.validTimestampCount}</div>
                                    <div className="summary-card-copy">
                                        Rows with valid timestamps available for calendar analysis.
                                    </div>
                                </div>
                            </div>

                            <div className="calendar-profile-grid">
                                {calendarProfiles.map((item) => (
                                    <div key={item.key} className="chart-shell benchmark-chart-shell">
                                        <div className="mini-chart-title">{item.title}</div>
                                        <ResponsiveContainer width="100%" height={280}>
                                            <BarChart data={item.profile.data.filter((bucket) => bucket.count > 0)}>
                                                <CartesianGrid stroke="var(--grid)" vertical={false} />
                                                <XAxis dataKey="label" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                                                <Tooltip
                                                    formatter={(value, name, payload) => {
                                                        if (name === "mean") {
                                                            return [formatMetric(Number(value), 4), "Mean value"];
                                                        }
                                                        if (name === "relativeIndex") {
                                                            return [formatMetric(Number(value), 3), "Std-normalized effect"];
                                                        }
                                                        return [value, name];
                                                    }}
                                                    labelFormatter={(label) => `${item.title}: ${label}`}
                                                    contentStyle={{
                                                        backgroundColor: "var(--panel)",
                                                        borderColor: "var(--border)",
                                                        borderRadius: 16,
                                                        color: "var(--text)",
                                                    }}
                                                />
                                                <Bar dataKey="mean" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                        <div className="automation-flag-row calendar-profile-meta">
                                            <span className="header-chip">peak {item.profile.peakLabel}</span>
                                            <span className="header-chip">trough {item.profile.troughLabel}</span>
                                            <span className="header-chip">share {formatMetric(item.profile.effectShare * 100, 1)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <p className="lead-copy">
                                Calendar analysis groups the observed values by timestamp fields and measures how much variance is explained by weekday, month, and hourly structure. Strong effects are a direct signal that calendar covariates should be part of the preprocessing view.
                            </p>
                        </>
                    ) : (
                        <p className="lead-copy">
                            Calendar analysis needs enough valid timestamps before weekday, monthly, or hourly effects can be estimated reliably.
                        </p>
                    )
                ) : null}

            </div>
        </Panel>
    );
}

export function BacktestingPanel({ benchmark, residualDiagnostics, status, errorMessage }) {
    const [activeMetric, setActiveMetric] = useState("mae");
    const activeModels = useMemo(() => benchmark?.models?.slice(0, 4) ?? [], [benchmark]);

    if (status === "loading") {
        return (
            <Panel title="Backtesting" kicker="Rolling-origin validation">
                <SkeletonBlock pills={2} chart lines={2} />
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Backtesting" kicker="Rolling-origin validation">
                <p className="lead-copy">
                    Backtesting unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    if (!benchmark?.models?.length) {
        return (
            <Panel title="Backtesting" kicker="Rolling-origin validation">
                <p className="lead-copy">
                    Not enough observations yet to run rolling-origin baseline comparisons.
                </p>
            </Panel>
        );
    }

    const metricLabel = activeMetric === "mae"
        ? "MAE"
        : activeMetric === "rmse"
            ? "RMSE"
            : "sMAPE";
    const metricSuffix = activeMetric === "smape" ? "%" : "";

    return (
        <Panel title="Backtesting" kicker="Rolling-origin validation">
            <div className="summary-card-grid benchmark-grid">
                {benchmark.models.map((model) => (
                    <div key={model.name} className={`summary-card ${benchmark.winner?.name === model.name ? "summary-card-highlight" : ""}`}>
                        <div className="summary-card-title">{model.label}</div>
                        <div className="architecture-name">MAE {model.mae}</div>
                        <div className="summary-card-copy">
                            RMSE {model.rmse} · sMAPE {model.smape}% across {benchmark.splits} folds.
                        </div>
                    </div>
                ))}
            </div>

            {residualDiagnostics?.available ? (
                <div className="summary-card-grid benchmark-grid benchmark-subgrid">
                    <div className="summary-card">
                        <div className="summary-card-title">Winning residuals</div>
                        <div className="architecture-name">{residualDiagnostics.label}</div>
                        <div className="summary-card-copy">
                            Ljung-Box on the winning baseline residual sequence.
                        </div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Ljung-Box Q</div>
                        <div className="architecture-name">{residualDiagnostics.statistic}</div>
                        <div className="summary-card-copy">
                            p-value {residualDiagnostics.pValue} using {residualDiagnostics.lagCount} residual lags.
                        </div>
                    </div>
                    <div className="summary-card">
                        <div className="summary-card-title">Residual autocorr</div>
                        <div className="architecture-name">{residualDiagnostics.maxResidualAcf}</div>
                        <div className="summary-card-copy">
                            {residualDiagnostics.rejectWhiteNoise ? "Residual structure remains after the baseline." : "Residuals are close to white noise under the baseline."}
                        </div>
                    </div>
                </div>
            ) : null}

            <div className="metric-toggle-row">
                {[
                    { key: "mae", label: "Fold MAE" },
                    { key: "rmse", label: "Fold RMSE" },
                    { key: "smape", label: "Fold sMAPE" },
                ].map((metric) => (
                    <button
                        key={metric.key}
                        type="button"
                        className={`metric-toggle ${activeMetric === metric.key ? "metric-toggle-active" : ""}`}
                        onClick={() => setActiveMetric(metric.key)}
                    >
                        {metric.label}
                    </button>
                ))}
            </div>

            <div className="benchmark-chart-grid">
                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Fold-by-fold {metricLabel}</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={benchmark.foldCurves}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="split" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value, name) => [
                                    `${Number(value).toFixed(activeMetric === "smape" ? 2 : 4)}${metricSuffix}`,
                                    activeModels.find((model) => `${model.name}_${activeMetric}` === name)?.label ?? name,
                                ]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            {activeModels.map((model) => (
                                <Line
                                    key={`${model.name}-${activeMetric}`}
                                    type="monotone"
                                    dataKey={`${model.name}_${activeMetric}`}
                                    name={`${model.name}_${activeMetric}`}
                                    stroke={BENCHMARK_COLORS[model.name] ?? "var(--accent)"}
                                    strokeWidth={benchmark.winner?.name === model.name ? 2.8 : 2}
                                    dot={{ r: benchmark.winner?.name === model.name ? 3.2 : 2.4 }}
                                    activeDot={{ r: 4.5 }}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-shell benchmark-chart-shell">
                    <div className="mini-chart-title">Horizon-wise absolute error</div>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={benchmark.horizonProfiles}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="horizon" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <Tooltip
                                formatter={(value, name) => [
                                    Number(value).toFixed(4),
                                    activeModels.find((model) => `${model.name}_mae` === name)?.label ?? name,
                                ]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            {activeModels.map((model) => (
                                <Line
                                    key={`${model.name}-horizon`}
                                    type="monotone"
                                    dataKey={`${model.name}_mae`}
                                    name={`${model.name}_mae`}
                                    stroke={BENCHMARK_COLORS[model.name] ?? "var(--accent)"}
                                    strokeWidth={benchmark.winner?.name === model.name ? 2.8 : 2}
                                    dot={false}
                                    activeDot={{ r: 4.5 }}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="benchmark-legend">
                {activeModels.map((model) => (
                    <span key={model.name} className="legend-item">
                        <span className="legend-dot" style={{ background: BENCHMARK_COLORS[model.name] ?? "var(--accent)" }} />
                        {model.label}
                    </span>
                ))}
            </div>

            {residualDiagnostics?.residualAcf?.length ? (
                <div className="chart-shell benchmark-residual-chart">
                    <div className="mini-chart-title">Winning residual autocorrelation</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={residualDiagnostics.residualAcf}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                            <Tooltip
                                formatter={(value) => [Number(value).toFixed(3), "Residual ACF"]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <Bar dataKey="value" fill="var(--cool)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : null}

            <p className="lead-copy">
                The fold chart shows stability across rolling origins, while the horizon chart shows how quickly each baseline degrades as the forecast step moves farther from the origin. The current winner is {benchmark.winner?.label ?? "pending"} over {benchmark.splits} folds and horizon {benchmark.horizon}.
            </p>
        </Panel>
    );
}

export function PacfPanel({ acfCurve, pacfCurve, analysis, activeWindow, onApplyLagWindow, status, errorMessage }) {
    if (status === "loading") {
        return (
            <Panel title="ACF / PACF Lag Scan" kicker="Lag diagnostics">
                <p className="lead-copy">
                    Running ACF and PACF diagnostics in the background worker. Lag suggestions will appear as soon as the correlation scan is ready.
                </p>
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="ACF / PACF Lag Scan" kicker="Lag diagnostics">
                <p className="lead-copy">
                    Diagnostics worker failed: {errorMessage}
                </p>
            </Panel>
        );
    }

    const lagSuggestions = uniqueSuggestions([
        {
            label: `PACF L${analysis.dominantPacfLag}`,
            detail: `window ${Math.max(24, Math.min(96, analysis.dominantPacfLag * 3))}`,
            window: Math.max(24, Math.min(96, analysis.dominantPacfLag * 3)),
        },
        {
            label: `ACF L${analysis.dominantAcfLag}`,
            detail: `window ${Math.max(24, Math.min(96, analysis.dominantAcfLag * 2))}`,
            window: Math.max(24, Math.min(96, analysis.dominantAcfLag * 2)),
        },
        {
            label: "Recommended",
            detail: `window ${analysis.suggestedLagWindow}`,
            window: analysis.suggestedLagWindow,
        },
    ]);

    return (
        <Panel title="ACF / PACF Lag Scan" kicker="Lag diagnostics">
            <div className="diagnostics-grid">
                <div className="chart-shell short-chart">
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={acfCurve}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                            <Tooltip
                                formatter={(value) => [Number(value).toFixed(3), "ACF"]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <ReferenceLine y={Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                            <ReferenceLine y={-Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                            <ReferenceLine x={analysis.dominantAcfLag} stroke="var(--accent)" strokeDasharray="4 4" />
                            <Bar dataKey="acf" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                <div className="chart-shell short-chart">
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={pacfCurve}>
                            <CartesianGrid stroke="var(--grid)" vertical={false} />
                            <XAxis dataKey="lag" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                            <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                            <Tooltip
                                formatter={(value) => [Number(value).toFixed(3), "PACF"]}
                                contentStyle={{
                                    backgroundColor: "var(--panel)",
                                    borderColor: "var(--border)",
                                    borderRadius: 16,
                                    color: "var(--text)",
                                }}
                            />
                            <ReferenceLine y={Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                            <ReferenceLine y={-Number(analysis.pacfThreshold)} stroke="var(--secondary)" strokeDasharray="4 4" />
                            <ReferenceLine x={analysis.dominantPacfLag} stroke="var(--accent)" strokeDasharray="4 4" />
                            <Bar dataKey="pacf" fill="var(--secondary)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
            <p className="lead-copy">
                ACF highlights repeating structure near lag {analysis.dominantAcfLag}, while PACF isolates direct dependence near lag {analysis.dominantPacfLag}. Use one of the lag-derived context windows below or keep the combined recommendation at {analysis.suggestedLagWindow} steps.
            </p>
            <div className="lag-suggestion-row">
                {lagSuggestions.map((suggestion) => (
                    <button
                        key={`${suggestion.label}-${suggestion.window}`}
                        type="button"
                        className={`lag-suggestion ${activeWindow === suggestion.window ? "lag-suggestion-active" : ""}`}
                        onClick={() => onApplyLagWindow(suggestion.window)}
                    >
                        <strong>{suggestion.label}</strong>
                        <span>{suggestion.detail}</span>
                    </button>
                ))}
            </div>
        </Panel>
    );
}

export function SeriesPanel({ data, status, errorMessage }) {
    if (status === "loading") {
        return (
            <Panel title="Reservoir Signal" kicker="Observed series">
                <p className="lead-copy">
                    Loading real series data and rebuilding the preview.
                </p>
            </Panel>
        );
    }

    if (status === "error") {
        return (
            <Panel title="Reservoir Signal" kicker="Observed series">
                <p className="lead-copy">
                    Series preview unavailable: {errorMessage}
                </p>
            </Panel>
        );
    }

    return (
        <Panel title="Reservoir Signal" kicker="Observed series">
            <div className="chart-shell">
                <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="seriesArea" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.35} />
                                <stop offset="95%" stopColor="var(--accent-soft)" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid stroke="var(--grid)" vertical={false} />
                        <XAxis dataKey="month" tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                        <YAxis tick={{ fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "var(--panel)",
                                borderColor: "var(--border)",
                                borderRadius: 16,
                                color: "var(--text)",
                            }}
                        />
                        <Area type="monotone" dataKey="value" stroke="var(--accent)" fill="url(#seriesArea)" strokeWidth={3} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </Panel>
    );
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function CodePanel({ code, notebook }) {
    return (
        <Panel
            title="Generated foreBlocks Script"
            kicker="Blueprint"
            actions={
                <div className="topbar-actions">
                    {notebook ? (
                        <button
                            className="ghost-button"
                            type="button"
                            onClick={() => downloadFile("forecast_pipeline.ipynb", notebook, "application/json")}
                        >
                            Download .ipynb
                        </button>
                    ) : null}
                    <button className="ghost-button" type="button" onClick={() => navigator.clipboard.writeText(code)}>
                        Copy script
                    </button>
                </div>
            }
        >
            <div className="code-shell">
                <div className="code-toolbar">
                    <div className="traffic-lights">
                        <span />
                        <span />
                        <span />
                    </div>
                    <div className="code-title">forecast_pipeline.py</div>
                </div>
                <pre className="code-body">
                    <code>
                        <HighlightedCode code={code} />
                    </code>
                </pre>
            </div>
        </Panel>
    );
}

export function NarrativePanel({ narrative }) {
    if (!narrative) {
        return null;
    }

    return (
        <Panel title="Experiment Narrative" kicker="Blueprint rationale" accent>
            <p className="lead-copy">{narrative.summary}</p>
            <div className="summary-card-grid narrative-grid">
                {narrative.reasons.map((reason) => (
                    <div key={reason.title} className="summary-card">
                        <div className="summary-card-title">{reason.title}</div>
                        <div className="summary-card-copy">{reason.body}</div>
                    </div>
                ))}
            </div>
            <div className="automation-flag-row">
                {narrative.tags.map((tag) => (
                    <span key={tag} className="header-chip">{tag}</span>
                ))}
            </div>
        </Panel>
    );
}
