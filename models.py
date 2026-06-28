"""Model fitting helpers shared by behavioural annotators."""

from __future__ import annotations

import numpy as np


def converged_cleanly(model) -> bool:
    """Treat tiny likelihood decreases as numerical noise."""
    precision = np.finfo(float).eps**0.5
    history = np.asarray(model.monitor_.history)
    return len(history) > 1 and bool(np.all(np.diff(history) >= -precision))


def fit_gaussian_hmm(
    arrays: list[np.ndarray],
    n_components: int = 3,
    n_iter: int = 500,
    random_seeds=range(10),
):
    """Fit several GaussianHMM initialisations and return the best model."""
    from hmmlearn.hmm import GaussianHMM

    if not arrays:
        raise ValueError("No valid trajectory sequences are available for HMM fitting.")
    stacked = np.vstack(arrays)
    if len(stacked) < n_components:
        raise ValueError("n_components cannot exceed the number of prepared observations.")
    lengths = [len(array) for array in arrays]
    best_model = None
    best_score = -np.inf
    any_converged = False
    for seed in random_seeds:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-3,
            min_covar=1e-2,
            random_state=seed,
        )
        model.fit(stacked, lengths)
        score = model.score(stacked, lengths)
        if score > best_score:
            best_score = score
            best_model = model
        any_converged = any_converged or converged_cleanly(model)
    return best_model, {"score": float(best_score), "converged": any_converged}
