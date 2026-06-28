"""HMM behavioural annotation using the shared UTM preprocessing pipeline."""

from __future__ import annotations

from typing import Iterable

import movingpandas as mpd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:  # Support both package and repository-root imports.
    from .annotation import AnnotationResult, BaseAnnotator
    from .models import fit_gaussian_hmm
    from .preprocessing import Feature, feature_matrix, normalise_features
except ImportError:  # pragma: no cover - exercised by direct script usage
    from annotation import AnnotationResult, BaseAnnotator
    from models import fit_gaussian_hmm
    from preprocessing import Feature, feature_matrix, normalise_features


class HMM(BaseAnnotator):
    """Annotate trajectories with a Gaussian HMM.

    It receives a ``TrajectoryCollection`` and a list of :class:`Feature`
    values, runs the shared UTM feature calculation and gap splitting, and
    returns an :class:`annotation.AnnotationResult`.
    """

    def __init__(
        self,
        features: Iterable[Feature | str] | None = None,
        num_states: int = 3,
        scale: bool = True,
        max_gap_factor: float = 5.0,
        n_iter: int = 500,
        random_seeds=range(10),
    ):
        super().__init__(features=features, max_gap_factor=max_gap_factor)
        if num_states < 1:
            raise ValueError("num_states must be at least one.")
        self.num_states = num_states
        self.scale = scale
        self.n_iter = n_iter
        self.random_seeds = tuple(random_seeds)

    def evaluation_segment_basis(self) -> str:
        return "state_run"

    def evaluation_metadata(self, annotation) -> dict:
        return {"configured_num_states": self.num_states}

    def annotate(self, trajectory_collection: mpd.TrajectoryCollection) -> AnnotationResult:
        prepared = self.prepare(trajectory_collection)
        self.initialise_annotations(prepared)
        feature_list = normalise_features(self.features)
        arrays = [feature_matrix(sequence, feature_list) for sequence in prepared.sequences]
        if not arrays:
            raise ValueError("No sequences with at least two valid UTM observations are available.")

        scaler = StandardScaler().fit(np.vstack(arrays)) if self.scale else None
        model_arrays = [scaler.transform(array) for array in arrays] if scaler is not None else arrays
        model, fit_metadata = fit_gaussian_hmm(
            model_arrays,
            n_components=self.num_states,
            n_iter=self.n_iter,
            random_seeds=self.random_seeds,
        )

        # Stable ordering is especially helpful for interpretation.  If speed
        # was not selected, fall back to the first requested feature.
        ordering_index = feature_list.index(Feature.SPEED) if Feature.SPEED in feature_list else 0
        raw_stacked = np.vstack(model_arrays)
        raw_states = model.predict(raw_stacked, [len(array) for array in model_arrays])
        state_values = []
        for state in range(self.num_states):
            values = raw_stacked[raw_states == state, ordering_index]
            # A fitted HMM can leave a state unvisited on short data.  Keep its
            # label deterministic by sorting such states last.
            state_values.append(float(np.mean(values)) if len(values) else float("inf"))
        order = np.argsort(state_values)
        mapping = {int(old): int(new) for new, old in enumerate(order)}

        labels_by_row: dict[int, int] = {}
        segment_by_row: dict[int, int] = {}
        for segment_id, (sequence, values) in enumerate(zip(prepared.sequences, model_arrays)):
            states = model.predict(values)
            for row_id, state in zip(sequence.row_ids, states):
                labels_by_row[int(row_id)] = mapping[int(state)]
                segment_by_row[int(row_id)] = segment_id

        row_ids = prepared.points["_hmmcma_row_id"]
        labels = row_ids.map(labels_by_row).fillna(-1).astype(int)
        prepared.points["state"] = labels
        prepared.points["cluster"] = labels
        prepared.points["segment_id"] = row_ids.map(segment_by_row).fillna(-1).astype(int)
        # HMM has no change-point estimation, but the common column remains so
        # HMM and BCPA output can be consumed interchangeably.
        prepared.points["change_point"] = False

        state_names = {state: f"state_{state}" for state in range(self.num_states)}
        if Feature.SPEED in feature_list and self.num_states == 3:
            state_names = {0: "resting", 1: "foraging", 2: "traveling"}
        result = AnnotationResult(
            trajectory_collection=prepared.to_trajectory_collection(),
            labels=labels.to_numpy(),
            change_points={},
            metadata={
                "features": tuple(feature.value for feature in feature_list),
                "model_state_mapping": mapping,
                "state_values": state_values,
                "state_names": state_names,
                "scaler": scaler,
                **fit_metadata,
            },
        )
        return self.finalise_result(result)
