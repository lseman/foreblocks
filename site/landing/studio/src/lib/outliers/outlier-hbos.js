export function computeHbosScores(values, binCount = null) {
    const finiteValues = values.filter((value) => Number.isFinite(value));
    if (finiteValues.length === 0) {
        return values.map(() => 0);
    }

    const n = finiteValues.length;
    const bins = binCount === null
        ? Math.min(Math.max(Math.floor(Math.sqrt(n)), 10), 50)
        : Math.max(2, Math.min(binCount, n));

    const minValue = Math.min(...finiteValues);
    const maxValue = Math.max(...finiteValues);
    const span = Math.max(maxValue - minValue, Number.EPSILON);

    const counts = new Array(bins).fill(0);
    const indices = values.map((value) => {
        if (!Number.isFinite(value)) {
            return -1;
        }

        let index = Math.floor(((value - minValue) / span) * bins);
        if (index < 0) index = 0;
        if (index >= bins) index = bins - 1;
        counts[index] += 1;
        return index;
    });

    const total = counts.reduce((sum, count) => sum + count, 0);
    const densities = counts.map((count) => (count + 1e-8) / Math.max(total, 1));

    return indices.map((index) => {
        if (index < 0) {
            return 0;
        }
        return -Math.log(densities[index]);
    });
}
