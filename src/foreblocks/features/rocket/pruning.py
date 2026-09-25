"""Kernel pruning for Rocket-family transforms.

Both selectors take standardized features `[N, H]`, labels, and the kernel id
of every feature column (a transform's `kernel_groups_`), and return the ids of
the kernels to keep. Feed them to the transform's `select_kernels` so pruned
kernels are no longer computed, then refit the linear head on the kept columns
(`RocketClassifier(pruning=...)` does all of this).

- `pocket_select`: POCKET (Chen et al., 2024, https://arxiv.org/abs/2309.08499).
  Stage 1 solves the group-lasso (l2,1) least-squares classifier on +-1
  class indicators by proximal gradient, re-picking the soft threshold at every
  iteration so exactly the requested number of kernel groups stays non-zero.
  Stage 2 (the ridge refit) is the caller's classifier fit.
- `srocket_select`: S-ROCKET (Salehinejad et al., 2022,
  https://arxiv.org/abs/2203.03445). A population of binary kernel masks is
  evolved against the validation accuracy of a ridge classifier trained once
  on all kernels, with pruned kernels' contributions removed (no refit per
  candidate). This implementation fixes the number of active kernels to a
  user-defined rate instead of trading it off in the fitness.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer


def _n_keep(n_groups: int, keep: float | int) -> int:
    n = int(round(keep * n_groups)) if isinstance(keep, float) and keep <= 1 else int(keep)
    return int(np.clip(n, 1, n_groups))


def _group_index(groups) -> tuple[np.ndarray, np.ndarray]:
    ids, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return ids, inverse


def _plus_minus_targets(y) -> np.ndarray:
    targets = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y).astype(np.float64)
    if targets.shape[1] == 1:  # binary: one column per class, as in the paper
        targets = np.hstack([-targets, targets])
    return targets


def _folds(y: np.ndarray, n_folds: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    seed = int(rng.integers(2**31 - 1))
    _, counts = np.unique(y, return_counts=True)
    n_folds = max(2, min(int(n_folds), len(y)))
    if counts.min() >= n_folds:
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def pocket_select(
    features,
    y,
    groups,
    keep: float | int = 0.4,
    n_iter: int = 100,
) -> np.ndarray:
    """POCKET stage 1: kernel ids of the `keep` (fraction or count) kernel
    groups that survive group-lasso proximal gradient with a per-iteration
    threshold chosen to prune the rest.
    """
    x = np.asarray(features, dtype=np.float64)
    ids, inverse = _group_index(groups)
    n_groups = len(ids)
    n_keep = _n_keep(n_groups, keep)
    targets = _plus_minus_targets(y)
    targets -= targets.mean(axis=0)  # absorbs the intercept
    x = x - x.mean(axis=0)
    # Step 1 / L with L the largest eigenvalue of X^T X (via the smaller Gram).
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    lipschitz = float(np.linalg.eigvalsh(gram)[-1]) if gram.size else 1.0
    step = 1.0 / max(lipschitz, 1e-12)
    weights = np.zeros((x.shape[1], targets.shape[1]))
    momentum = weights.copy()
    t = 1.0
    for _ in range(max(1, int(n_iter))):
        # FISTA step on 0.5 * ||X W - Y||^2, then the l2,1 proximal operator.
        z = momentum - step * (x.T @ (x @ momentum - targets))
        norms = np.sqrt(np.bincount(inverse, weights=(z * z).sum(axis=1), minlength=n_groups))
        if n_keep < n_groups:
            threshold = np.partition(norms, n_groups - n_keep - 1)[n_groups - n_keep - 1]
        else:
            threshold = 0.0
        shrink = np.where(norms > threshold, 1.0 - threshold / np.maximum(norms, 1e-300), 0.0)
        new_weights = z * shrink[inverse][:, None]
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = new_weights + ((t - 1.0) / t_next) * (new_weights - weights)
        weights, t = new_weights, t_next
    norms = np.sqrt(np.bincount(inverse, weights=(weights**2).sum(axis=1), minlength=n_groups))
    return np.sort(ids[np.argsort(-norms, kind="stable")[:n_keep]])


def srocket_select(
    features,
    y,
    groups,
    keep: float | int = 0.4,
    population: int = 32,
    generations: int = 60,
    mutation_rate: float = 0.02,
    n_folds: int = 3,
    alphas=None,
    seed: int | None = None,
) -> np.ndarray:
    """S-ROCKET-style evolutionary selection of `keep` (fraction or count)
    kernels maximizing held-out accuracy of pretrained ridge heads.

    Fitness is the mean accuracy over `n_folds` stratified folds (one head per
    fold, trained on all kernels) rather than one validation split: a single
    small split lets the search overfit it with chance-level kernels.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(y)
    ids, inverse = _group_index(groups)
    n_groups = len(ids)
    n_keep = _n_keep(n_groups, keep)
    if n_keep == n_groups:
        return ids

    indicator = sparse.csr_matrix(
        (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
        shape=(n_groups, len(inverse)),
    )
    alphas = np.logspace(-3, 3, 13) if alphas is None else alphas
    # One ridge head per fold; each fold contributes, per kernel, its share of
    # every held-out decision score: [G, n_val * C'].
    folds = []
    importance = np.zeros(n_groups)
    for train, val in _folds(y, n_folds, rng):
        head = RidgeClassifierCV(alphas=alphas).fit(x[train], y[train])
        coef = np.atleast_2d(head.coef_)  # [C', H]
        contributions = np.stack(
            [indicator @ (x[val] * c).T for c in coef], axis=2
        ).reshape(n_groups, -1)
        truth = np.searchsorted(head.classes_, y[val])
        folds.append((contributions, np.atleast_1d(head.intercept_), truth))
        importance += np.sqrt(
            np.bincount(inverse, weights=(coef**2).sum(axis=0), minlength=n_groups)
        )

    def fold_fitness(masks, contributions, intercept, truth) -> np.ndarray:
        scores = (masks @ contributions).reshape(len(masks), len(truth), -1) + intercept
        if scores.shape[2] == 1:  # binary ridge: one decision score
            predicted = (scores[..., 0] > 0).astype(int)
            margin = np.where(truth == 1, scores[..., 0], -scores[..., 0])
        else:
            predicted = scores.argmax(axis=2)
            true_score = np.take_along_axis(scores, truth[None, :, None], axis=2)[..., 0]
            other = scores.copy()
            np.put_along_axis(other, truth[None, :, None], -np.inf, axis=2)
            margin = true_score - other.max(axis=2)
        accuracy = (predicted == truth).mean(axis=1)
        # Mean margin breaks accuracy ties (bounded well below 1 / n_val).
        return accuracy + 1e-3 / max(len(truth), 1) * np.tanh(margin.mean(axis=1))

    def fitness(masks: np.ndarray) -> np.ndarray:
        m = masks.astype(np.float64)
        return np.mean([fold_fitness(m, *fold) for fold in folds], axis=0)

    probs = importance / importance.sum() if importance.sum() > 0 else None

    def random_mask(weighted: bool) -> np.ndarray:
        mask = np.zeros(n_groups, dtype=bool)
        mask[rng.choice(n_groups, n_keep, replace=False, p=probs if weighted else None)] = True
        return mask

    def repair(mask: np.ndarray) -> np.ndarray:
        on = np.flatnonzero(mask)
        if len(on) > n_keep:
            mask[rng.choice(on, len(on) - n_keep, replace=False)] = False
        elif len(on) < n_keep:
            off = np.flatnonzero(~mask)
            mask[rng.choice(off, n_keep - len(on), replace=False)] = True
        return mask

    top = np.zeros(n_groups, dtype=bool)
    top[np.argsort(-importance, kind="stable")[:n_keep]] = True
    pop = np.stack([top] + [random_mask(i % 2 == 0) for i in range(max(population, 4) - 1)])
    scores = fitness(pop)
    n_swap = max(1, int(round(mutation_rate * n_keep)))
    for _ in range(max(0, int(generations))):
        children = [pop[np.argmax(scores)].copy()]  # elitism
        while len(children) < len(pop):
            a, b = (
                max(rng.choice(len(pop), 3, replace=False), key=lambda i: scores[i])
                for _ in range(2)
            )
            child = repair(np.where(rng.random(n_groups) < 0.5, pop[a], pop[b]))
            on, off = np.flatnonzero(child), np.flatnonzero(~child)
            if len(off):
                swap = min(n_swap, len(off))
                child[rng.choice(on, swap, replace=False)] = False
                child[rng.choice(off, swap, replace=False)] = True
            children.append(child)
        pop = np.stack(children)
        scores = fitness(pop)
    return np.sort(ids[pop[np.argmax(scores)]])
