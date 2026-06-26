"""Model fitting helpers shared by the HMM annotator and legacy callers."""

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


def apply_hmm(arrays, seq_dfs, n_components=3, columns=None):
    """Backward-compatible low-level HMM application.

    Prefer ``state_annotation.HMM(...).annotate(trajectory_collection)`` for
    the common pipeline and an annotated ``TrajectoryCollection`` result.
    """
    model, metadata = fit_gaussian_hmm(arrays, n_components=n_components)
    speed_index = list(columns.feature_cols).index("speed") if columns and "speed" in columns.feature_cols else 0
    stacked = np.vstack(arrays)
    raw_states = model.predict(stacked)
    state_speeds = [float(np.mean(stacked[raw_states == state, speed_index])) for state in range(n_components)]
    order = np.argsort(state_speeds)
    mapping = {int(old): int(new) for new, old in enumerate(order)}

    for frame, array in zip(seq_dfs, arrays):
        frame["state"] = [mapping[int(state)] for state in model.predict(array)]
    state_mappings = {
        "model_state_mapping": mapping,
        "state_speeds": state_speeds,
        "order": order,
        "state_names": {index: f"state_{index}" for index in range(n_components)},
        **metadata,
    }
    return seq_dfs, None, state_mappings
