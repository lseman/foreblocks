function average(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function standardDeviation(values, mean) {
    return Math.sqrt(values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function movingAverage(values, window) {
    const radius = Math.floor(window / 2);
    return values.map((value, index) => {
        const start = Math.max(0, index - radius);
        const end = Math.min(values.length, index + radius + 1);
        const slice = values.slice(start, end).filter((entry) => Number.isFinite(entry));
        return slice.length > 0 ? average(slice) : value;
    });
}

function smoothSeasonalCycle(cycle) {
    if (cycle.length < 3) {
        return cycle;
    }
    const count = cycle.length;
    return cycle.map((_, index) => {
        const prev = cycle[(index - 1 + count) % count];
        const curr = cycle[index];
        const next = cycle[(index + 1) % count];
        return average([prev, curr, next]);
    });
}

export function buildDecomposition(data, analysis) {
    if (data.length === 0) {
        return {
            periods: [],
            strength: 0,
            components: [],
            data: [],
            residualStd: 0,
        };
    }

    const values = data.map((point) => point.value);
    const period = clamp(Math.max(analysis.dominantAcfLag || 12, analysis.dominantPacfLag || 6), 4, Math.min(24, Math.max(4, values.length - 1)));
    const trendWindow = Math.min(values.length, Math.max(5, period * 2 + 1));
    let trend = movingAverage(values, trendWindow);
    let seasonalCycle = Array.from({ length: period }, () => 0);

    for (let iter = 0; iter < 2; iter += 1) {
        const seasonalBuckets = Array.from({ length: period }, () => ({ sum: 0, count: 0 }));
        values.forEach((value, index) => {
            const detrended = Number.isFinite(value) && Number.isFinite(trend[index]) ? value - trend[index] : null;
            if (Number.isFinite(detrended)) {
                const bucket = seasonalBuckets[index % period];
                bucket.sum += detrended;
                bucket.count += 1;
            }
        });

        const rawSeasonalMeans = seasonalBuckets.map((bucket) => (bucket.count > 0 ? bucket.sum / bucket.count : 0));
        const smoothedSeasonalMeans = smoothSeasonalCycle(rawSeasonalMeans);
        const seasonalMean = smoothedSeasonalMeans.length > 0 ? average(smoothedSeasonalMeans) : 0;
        seasonalCycle = smoothedSeasonalMeans.map((value) => value - seasonalMean);

        const deseasonalized = values.map((value, index) =>
            Number.isFinite(value) ? value - seasonalCycle[index % period] : 0,
        );
        trend = movingAverage(deseasonalized, trendWindow);
    }

    const seasonal = values.map((_value, index) => seasonalCycle[index % period] ?? 0);
    const deseasonalized = values.map((value, index) =>
        Number.isFinite(value) ? value - seasonal[index] : 0,
    );
    const refinedTrend = movingAverage(deseasonalized, trendWindow);

    const decompositionData = data.map((point, index) => {
        const pointTrend = Number.isFinite(refinedTrend[index]) ? refinedTrend[index] : 0;
        const pointSeasonal = seasonal[index];
        const residual = Number.isFinite(point.value) ? point.value - pointTrend - pointSeasonal : 0;

        return {
            ...point,
            trend: Number(pointTrend.toFixed(3)),
            seasonal: Number(pointSeasonal.toFixed(3)),
            residual: Number(residual.toFixed(3)),
        };
    });

    const observedValues = decompositionData.map((point) => point.value);
    const residualValues = decompositionData.map((point) => point.residual);
    const totalVariance = standardDeviation(observedValues, average(observedValues)) ** 2;
    const seasonalVariance = standardDeviation(seasonal, average(seasonal)) ** 2;
    const strength = totalVariance > 1e-12 ? Math.min(1, seasonalVariance / totalVariance) : 0;

    return {
        periods: [period],
        strength,
        components: [{ period, strength }],
        data: decompositionData.map((point) => ({
            ...point,
            observed: point.value,
        })),
        residualStd: standardDeviation(residualValues, average(residualValues)),
    };
}

export function shiftValues(values, lag) {
    return values.map((_value, index) => {
        const sourceIndex = index - lag;
        return sourceIndex >= 0 && Number.isFinite(values[sourceIndex]) ? values[sourceIndex] : null;
    });
}

export function buildRollingHistory(values, windowSize, reducer) {
    const effectiveWindow = Math.max(2, Math.round(windowSize));
    return values.map((_value, index) => {
        const start = Math.max(0, index - effectiveWindow);
        const window = values.slice(start, index).filter((item) => Number.isFinite(item));
        return window.length >= Math.max(2, Math.min(4, effectiveWindow - 1)) ? reducer(window) : null;
    });
}

export function buildEmdOverviewData(seriesData, emdDiagnostics) {
    if (!emdDiagnostics?.available) {
        return [];
    }

    const components = emdDiagnostics.components ?? [];
    const residual = emdDiagnostics.residual?.values ?? [];
    return seriesData.map((point, index) => ({
        ...point,
        observed: point.value,
        reconstruction: residual[index] + components.reduce((total, component) => total + (component.values[index] ?? 0), 0),
        residual: residual[index] ?? 0,
    }));
}

export function buildEmdComponentSeries(seriesData, component) {
    return seriesData.map((point, index) => ({
        ...point,
        value: component.values[index] ?? 0,
    }));
}
