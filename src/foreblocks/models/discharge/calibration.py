"""Native scalar temperature calibration of fixed classifier logits."""

import numpy as np


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Minimize held-out NLL over a positive temperature in [0.05, 20].

    Bounded golden-section search uses convexity of NLL in inverse temperature.
    Include the uncalibrated value explicitly so fitted NLL cannot be worse.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if (
        logits.ndim != 2
        or not len(logits)
        or logits.shape[1] < 2
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected nonempty finite [samples, classes>=2] logits.")
    if (
        targets.shape != (len(logits),)
        or targets.dtype.kind not in "iu"
        or np.any((targets < 0) | (targets >= logits.shape[1]))
    ):
        raise ValueError("Targets must be integer class indices matching logits.")

    def nll(inverse_temperature):
        scaled = logits * inverse_temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        return float(
            np.mean(
                np.log(np.exp(shifted).sum(axis=1))
                - shifted[np.arange(len(targets)), targets]
            )
        )

    low, high = 0.05, 20.0
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(80):
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        if nll(left) < nll(right):
            high = right
        else:
            low = left
    inverse = min([1.0, 0.05, 20.0, (low + high) / 2], key=nll)
    return float(1 / inverse)
