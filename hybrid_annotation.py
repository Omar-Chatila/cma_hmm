"""BCPA-smoothed HMM annotation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import movingpandas as mpd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:  # Support both package and repository-root imports.
    from .annotation import AnnotationResult, BaseAnnotator
    from .BCPA import BCPA
    from .models import fit_gaussian_hmm
    from .preprocessing import Feature, PreparedSequence, feature_matrix, normalise_features
except ImportError:  # pragma: no cover - exercised by direct script usage
    from annotation import AnnotationResult, BaseAnnotator
    from BCPA import BCPA
    from models import fit_gaussian_hmm
    from preprocessing import Feature, PreparedSequence, feature_matrix, normalise_features


@dataclass
class _HybridSegment:
    sequence: PreparedSequence
    sequence_index: int
    start: int
    end: int
    segment_id: int

    @property
    def row_ids(self) -> np.ndarray:
        return self.sequence.row_ids[self.start : self.end]


class BCPAHMM(BaseAnnotator):
    """Fit an HMM normally, then smooth its states over BCPA segments.

    BCPA supplies temporal episode boundaries.  The HMM supplies behavioural
    state semantics.  Each BCPA segment receives one final state: by default,
    the HMM state with the largest summed posterior probability across that
    segment.
    """

    def __init__(
        self,
        features: Iterable[Feature | str] | None = None,
        penalty: float = 5.0,
        min_segment_size: int = 3,
        num_states: int = 3,
        scale: bool = True,
        segment_state_method: str = "posterior_sum",
        max_gap_factor: float = 5.0,
        n_iter: int = 500,
        random_seeds=range(10),
    ):
        super().__init__(features=features, max_gap_factor=max_gap_factor)
        if num_states < 1:
            raise ValueError("num_states must be at least one.")
        if penalty <= 0:
            raise ValueError("penalty must be positive.")
        if min_segment_size < 1:
            raise ValueError("min_segment_size must be at least one.")
        valid_methods = {"posterior_sum", "majority_vote"}
        if segment_state_method not in valid_methods:
            choices = ", ".join(sorted(valid_methods))
            raise ValueError(f"segment_state_method must be one of: {choices}")
        self.penalty = penalty
        self.min_segment_size = min_segment_size
        self.num_states = num_states
        self.scale = scale
        self.segment_state_method = segment_state_method
        self.n_iter = n_iter
        self.random_seeds = tuple(random_seeds)

    def _bcpa(self) -> BCPA:
        return BCPA(
            features=self.features,
            penalty=self.penalty,
            num_clusters=self.num_states,
            min_segment_size=self.min_segment_size,
            max_gap_factor=self.max_gap_factor,
        )

    def evaluation_metadata(self, annotation) -> dict:
        return {
            "configured_num_states": self.num_states,
            "bcpa_penalty": self.penalty,
            "bcpa_min_segment_size": self.min_segment_size,
            "segment_state_method": self.segment_state_method,
        }

    def _bcpa_segments(self, sequences: list[PreparedSequence], time_col: str) -> tuple[list[_HybridSegment], dict]:
        segmenter = self._bcpa()
        change_points: dict[object, list] = defaultdict(list)
        segments: list[_HybridSegment] = []
        next_segment_id = 0

        for sequence_index, sequence in enumerate(sequences):
            points = segmenter.change_points(sequence)
            for point in points:
                change_points[sequence.trajectory_id].append(sequence.frame.iloc[point][time_col])

            start = 0
            for end in [*points, len(sequence.frame)]:
                if end > start:
                    segments.append(
                        _HybridSegment(
                            sequence=sequence,
                            sequence_index=sequence_index,
                            start=start,
                            end=end,
                            segment_id=next_segment_id,
                        )
                    )
                    next_segment_id += 1
                start = end

        return segments, dict(change_points)

    def annotate(self, trajectory_collection: mpd.TrajectoryCollection) -> AnnotationResult:
        prepared = self.prepare(trajectory_collection)
        self.initialise_annotations(prepared)
        feature_list = normalise_features(self.features)

        arrays = [feature_matrix(sequence, feature_list) for sequence in prepared.sequences]
        if not arrays:
            raise ValueError("No sequences with at least two valid UTM observations are available.")
        segments, change_points = self._bcpa_segments(prepared.sequences, prepared.time_col)
        if not segments:
            raise ValueError("No BCPA segments with valid UTM observations are available.")

        scaler = StandardScaler().fit(np.vstack(arrays)) if self.scale else None
        model_arrays = [scaler.transform(array) for array in arrays] if scaler is not None else arrays
        model, fit_metadata = fit_gaussian_hmm(
            model_arrays,
            n_components=self.num_states,
            n_iter=self.n_iter,
            random_seeds=self.random_seeds,
        )

        ordering_index = feature_list.index(Feature.SPEED) if Feature.SPEED in feature_list else 0
        stacked = np.vstack(model_arrays)
        raw_states = model.predict(stacked, [len(array) for array in model_arrays])
        state_values = []
        for state in range(self.num_states):
            values = stacked[raw_states == state, ordering_index]
            state_values.append(float(np.mean(values)) if len(values) else float("inf"))
        order = np.argsort(state_values)
        mapping = {int(old): int(new) for new, old in enumerate(order)}
        inverse_mapping = {new: old for old, new in mapping.items()}

        point_states = [model.predict(values) for values in model_arrays]
        point_posteriors = [model.predict_proba(values) for values in model_arrays]

        labels_by_row: dict[int, int] = {}
        segment_by_row: dict[int, int] = {}
        change_by_row: set[int] = set()
        segment_states: dict[int, int] = {}
        for segment in segments:
            segment_state = self._segment_state(
                point_states[segment.sequence_index][segment.start : segment.end],
                point_posteriors[segment.sequence_index][segment.start : segment.end],
            )
            label = mapping[segment_state]
            segment_states[segment.segment_id] = label
            for row_id in segment.row_ids:
                labels_by_row[int(row_id)] = label
                segment_by_row[int(row_id)] = segment.segment_id
            if segment.end < len(segment.sequence.frame):
                change_by_row.add(int(segment.sequence.row_ids[segment.end]))

        row_ids = prepared.points["_hmmcma_row_id"]
        labels = row_ids.map(labels_by_row).fillna(-1).astype(int)
        prepared.points["state"] = labels
        prepared.points["cluster"] = labels
        prepared.points["segment_id"] = row_ids.map(segment_by_row).fillna(-1).astype(int)
        prepared.points["change_point"] = row_ids.isin(change_by_row)

        state_names = {state: f"state_{state}" for state in range(self.num_states)}
        if Feature.SPEED in feature_list and self.num_states == 3:
            state_names = {0: "resting", 1: "foraging", 2: "traveling"}
        result = AnnotationResult(
            trajectory_collection=prepared.to_trajectory_collection(),
            labels=labels.to_numpy(),
            change_points=change_points,
            metadata={
                "features": tuple(feature.value for feature in feature_list),
                "bcpa_penalty": self.penalty,
                "bcpa_min_segment_size": self.min_segment_size,
                "bcpa_num_segments": len(segments),
                "segment_state_method": self.segment_state_method,
                "segment_states": segment_states,
                "model_state_mapping": mapping,
                "inverse_model_state_mapping": inverse_mapping,
                "state_values": state_values,
                "state_names": state_names,
                "scaler": scaler,
                **fit_metadata,
            },
        )
        return self.finalise_result(result)

    def _segment_state(self, states: np.ndarray, posteriors: np.ndarray) -> int:
        if self.segment_state_method == "majority_vote":
            counts = np.bincount(states.astype(int), minlength=self.num_states)
            return int(np.argmax(counts))
        posterior_sums = posteriors.sum(axis=0)
        return int(np.argmax(posterior_sums))

