// Helper utilities
function average(values) {
    if (!values.length) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function variance(values, mean = average(values)) {
    if (values.length < 2) return 0;
    return values.reduce((total, v) => total + (v - mean) ** 2, 0) / values.length;
}

function standardDeviation(values, mean) {
    return Math.sqrt(variance(values, mean));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

// More efficient moving average (avoids slice every time)
function movingAverage(values, windowSize) {
    if (windowSize < 1) windowSize = 1;
    const radius = Math.floor(windowSize / 2);
    const result = new Array(values.length);
    let sum = 0;
    let count = 0;

    for (let i = 0; i < values.length; i++) {
        // Add new element
        sum += values[i];
        count++;

        // Remove old element if window is full
        if (i >= windowSize) {
            sum -= values[i - windowSize];
            count--;
        }

        const start = Math.max(0, i - radius);
        const end = Math.min(values.length, i + radius + 1);
        // For true centered MA we still need to adjust, but this is faster approximation
        result[i] = sum / (end - start); // or use fixed window where possible
    }
    return result;
}

// Better centered moving average (more accurate for decomposition)
function centeredMovingAverage(values, windowSize) {
    const ma = new Array(values.length).fill(0);
    const half = Math.floor(windowSize / 2);

    for (let i = 0; i < values.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(values.length, i + half + 1);
        let sum = 0;
        for (let j = start; j < end; j++) sum += values[j];
        ma[i] = sum / (end - start);
    }
    return ma;
}

function centerSeasonalPattern(pattern) {
    const mean = average(pattern);
    return pattern.map(v => v - mean);
}

function buildSimpleSeasonalComponent(detrended, period) {
    const buckets = Array.from({ length: period }, () => []);
    detrended.forEach((value, i) => {
        if (value != null) buckets[i % period].push(value);
    });

    const pattern = centerSeasonalPattern(
        buckets.map(bucket => bucket.length ? average(bucket) : 0)
    );

    return detrended.map((_, i) => pattern[i % period]);
}

// Main decomposition - closer to MSTL spirit
export function buildMstlStyleDecomposition(series, seasonalPeriods = []) {
    const observed = series.map(p => p.value);
    const n = observed.length;
    if (n === 0) throw new Error("Empty series");

    let periods = seasonalPeriods.length > 0
        ? [...seasonalPeriods].sort((a, b) => a - b)  // ascending order (important!)
        : [12];

    // Remove unrealistic periods
    periods = periods.filter(p => p >= 2 && p <= Math.floor(n / 3));

    if (periods.length === 0) periods = [12];

    // Initialize seasonal components
    const seasonalComponents = periods.map(period => ({
        period,
        series: new Array(n).fill(0),
        strength: 0
    }));

    // Initial trend (simple but decent starting point)
    let trend = centeredMovingAverage(observed, Math.max(7, periods[0] * 2 + 1));

    const iterations = 2; // outer loops - usually enough

    for (let iter = 0; iter < iterations; iter++) {
        for (let compIdx = 0; compIdx < seasonalComponents.length; compIdx++) {
            const component = seasonalComponents[compIdx];

            // Compute "other seasonal" sum (exclude current)
            const otherSeasonal = new Array(n).fill(0);
            seasonalComponents.forEach((other, idx) => {
                if (idx !== compIdx) {
                    for (let i = 0; i < n; i++) otherSeasonal[i] += other.series[i];
                }
            });

            // Detrend + remove other seasonals
            const detrended = observed.map((v, i) => v - trend[i] - otherSeasonal[i]);

            // Extract this seasonal
            component.series = buildSimpleSeasonalComponent(detrended, component.period);
        }

        // Update trend on fully deseasonalized series
        const combinedSeasonal = new Array(n).fill(0);
        seasonalComponents.forEach(comp => {
            for (let i = 0; i < n; i++) combinedSeasonal[i] += comp.series[i];
        });

        const deseasonalized = observed.map((v, i) => v - combinedSeasonal[i]);
        trend = centeredMovingAverage(deseasonalized, Math.max(7, Math.min(n - 1, periods[0] * 2 + 1)));
    }

    // Final calculations
    const combinedSeasonal = new Array(n).fill(0);
    seasonalComponents.forEach(comp => {
        for (let i = 0; i < n; i++) combinedSeasonal[i] += comp.series[i];
    });

    const residual = observed.map((v, i) => v - trend[i] - combinedSeasonal[i]);
    const deseasonalizedFinal = observed.map((v, i) => v - combinedSeasonal[i]); // or trend? depends

    // Seasonal strengths (standard formula)
    const resVar = variance(residual);
    seasonalComponents.forEach(comp => {
        const seasonalPlusRes = comp.series.map((v, i) => v + residual[i]);
        const sVar = variance(seasonalPlusRes);
        comp.strength = clamp(1 - resVar / Math.max(1e-10, sVar), 0, 1);
    });

    const combinedStrength = clamp(1 - resVar / Math.max(1e-10, variance(deseasonalizedFinal)), 0, 1);

    return {
        periods,
        strength: Number(combinedStrength.toFixed(3)),
        components: seasonalComponents.map(c => ({
            period: c.period,
            strength: Number(c.strength.toFixed(3))
        })),
        data: series.map((point, i) => ({
            ...point,
            observed: Number(observed[i].toFixed(4)),
            trend: Number(trend[i].toFixed(4)),
            seasonal: Number(combinedSeasonal[i].toFixed(4)),
            residual: Number(residual[i].toFixed(4))
        })),
        residualStd: Number(standardDeviation(residual).toFixed(4)),
    };
}
export function inferSeasonalPeriods(peaks = [], seriesLength = 0) {
    const maxPeriod = Math.max(2, Math.floor(seriesLength / 3));
    const periods = Array.from(new Set(
        (peaks || [])
            .map((point) => Math.round(point?.lag ?? 0))
            .filter((lag) => Number.isFinite(lag) && lag >= 2 && lag <= maxPeriod),
    )).sort((a, b) => a - b);

    if (periods.length > 0) {
        return periods.slice(0, 3);
    }

    return [Math.min(12, Math.max(2, Math.floor(seriesLength / 4)))];
}
