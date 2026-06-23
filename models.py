import numpy as np

def converged_cleanly(model) -> bool:
    precision = np.finfo(float).eps ** 0.5
    history = np.array(model.monitor_.history)
    return bool(np.all(np.diff(history) >= -precision))


def apply_hmm(arrays, seq_dfs, n_components=3, columns=None):
    import numpy as np
    from hmmlearn.hmm import GaussianHMM

    from .preprocessing import process_trajectories

    lengths = [arr.shape[0] for arr in arrays]
    stacked = np.vstack(arrays)
    best_model = None
    best_score = -np.inf
    has_converged = False
    for seed in range(10):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=500,
            tol=1e-3,
            min_covar=1e-2,
            random_state=seed,
        )
        model.fit(stacked, lengths)
        score = model.score(stacked, lengths)
        if converged_cleanly(model) and not has_converged:
            has_converged = True
        if score > best_score:
            best_score = score
            best_model = model

    if not has_converged:
        print("Model did not converge.")
    else:
        print(f"Model converged after {model.n_iter} iterations.")

    stacked_state_seq = best_model.predict(stacked)
    speed_column_index = list(columns.feature_cols).index("speed")

    state_speeds = []
    for k in range(n_components):
        idx = stacked_state_seq == k
        state_speeds.append(np.mean(stacked[idx, speed_column_index]))

    order = np.argsort(state_speeds)
    model_state_mapping = {old: new for new, old in enumerate(order)}

    state_mappings = {
        "model_state_mapping": model_state_mapping,
        "state_speeds": state_speeds,
        "order": order,
        "state_names": {0: "resting", 1: "foraging", 2: "traveling"},
    }

    state_seqs = []
    for arr in arrays:
        raw_states = best_model.predict(arr)
        mapped_states = np.array([model_state_mapping[s] for s in raw_states])
        state_seqs.append(mapped_states)

    for df, states in zip(seq_dfs, state_seqs):
        df["state"] = states

    animal_trajectories, dt_threshold = process_trajectories(seq_dfs, columns)
    return animal_trajectories, dt_threshold, state_mappings
