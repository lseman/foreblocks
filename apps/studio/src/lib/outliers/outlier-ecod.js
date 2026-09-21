function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function computeEmpiricalRanks(values) {
    const finite = values
        .map((value, index) => ({ value, index }))
        .filter((item) => Number.isFinite(item.value));

    const sorted = [...finite].sort((a, b) => a.value - b.value);
    const n = sorted.length;
    const ranks = new Array(values.length).fill(null);

    for (let rank = 0; rank < n; rank += 1) {
        ranks[sorted[rank].index] = rank;
    }

    return { ranks, n };
}

export function computeEcodScores(values) {
    const { ranks, n } = computeEmpiricalRanks(values);
    const scores = new Array(values.length).fill(0);
    if (n === 0) {
        return scores;
    }

    const denominator = n + 1;
    for (let index = 0; index < values.length; index += 1) {
        const rank = ranks[index];
        if (rank === null || rank === undefined) {
            scores[index] = 0;
            continue;
        }

        const cdf = (rank + 1) / denominator;
        const tail = Math.min(cdf, 1 - cdf);
        const score = 1 - 2 * tail;
        scores[index] = clamp(score, 0, 1);
    }

    return scores;
}
