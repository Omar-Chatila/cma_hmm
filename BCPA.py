"""Behavioural change point analysis on the shared trajectory pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import ruptures as rpt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:  # Support both package and repository-root imports.
    from .annotation import AnnotationResult, BaseAnnotator
    from .preprocessing import ColumnConfig, Feature, PreparedSequence, feature_matrix, normalise_features
except ImportError:  # pragma: no cover - exercised by direct script usage
    from annotation import AnnotationResult, BaseAnnotator
    from preprocessing import ColumnConfig, Feature, PreparedSequence, feature_matrix, normalise_features


def segment_features(values: np.ndarray, change_points: list[int]) -> list[tuple[int, int, np.ndarray]]:
    """Summarise each BCPA segment with the same statistics as the old script."""
    summaries: list[tuple[int, int, np.ndarray]] = []
    start = 0
    for end in [*change_points, len(values)]:
        segment = values[start:end]
        if len(segment):
            summary = np.concatenate(
                (np.mean(segment, axis=0), np.std(segment, axis=0), np.median(segment, axis=0), np.sum(segment, axis=0))
            )
            summaries.append((start, end, summary))
        start = end
    return summaries


class BCPA(BaseAnnotator):
    """Detect change points and cluster resulting behavioural segments.

    It accepts exactly the same collection, ``features`` and ``ColumnConfig``
    arguments as :class:`state_annotation.HMM`, and returns the same
    :class:`annotation.AnnotationResult` type.
    """

    def __init__(
        self,
        features: Iterable[Feature | str] | None = None,
        columns: ColumnConfig | None = None,
        penalty: float = 5.0,
        num_clusters: int = 3,
        min_segment_size: int = 3,
        max_gap_factor: float = 5.0,
        random_state: int | None = 0,
    ):
        super().__init__(features=features, columns=columns, max_gap_factor=max_gap_factor)
        if penalty <= 0:
            raise ValueError("penalty must be positive.")
        if num_clusters < 1 or min_segment_size < 1:
            raise ValueError("num_clusters and min_segment_size must be at least one.")
        self.penalty = penalty
        self.num_clusters = num_clusters
        self.min_segment_size = min_segment_size
        self.random_state = random_state

    def _change_points(self, sequence: PreparedSequence) -> list[int]:
        values = feature_matrix(sequence, self.features or self.columns.feature_cols)
        # PELT cannot form an internal segment at this size.  It is still a
        # valid single behavioural segment and will be clustered below.
        if len(values) < self.min_segment_size * 2:
            return []
        scaled = StandardScaler().fit_transform(values)
        breakpoints = rpt.Pelt(model="rbf", min_size=self.min_segment_size).fit(scaled).predict(pen=self.penalty)
        return [int(point) for point in breakpoints[:-1] if 0 < point < len(values)]

    def annotate(self, trajectory_collection) -> AnnotationResult:
        prepared = self.prepare(trajectory_collection)
        self.initialise_annotations(prepared)
        feature_list = normalise_features(self.features or self.columns.feature_cols)

        all_segments: list[tuple[PreparedSequence, int, int, int, np.ndarray]] = []
        change_points: dict[object, list] = defaultdict(list)
        next_segment_id = 0
        for sequence_index, sequence in enumerate(prepared.sequences):
            values = feature_matrix(sequence, feature_list)
            points = self._change_points(sequence)
            for point in points:
                change_points[sequence.trajectory_id].append(sequence.frame.iloc[point][prepared.time_col])
            for start, end, summary in segment_features(values, points):
                all_segments.append((sequence, sequence_index, start, end, summary))
                next_segment_id += 1

        labels_by_row: dict[int, int] = {}
        segment_by_row: dict[int, int] = {}
        change_by_row: set[int] = set()
        if all_segments:
            summaries = np.vstack([segment[4] for segment in all_segments])
            n_clusters = min(self.num_clusters, len(all_segments))
            # KMeans behaves sensibly for a single segment and avoids a GMM
            # covariance failure in that common edge case.
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10).fit_predict(
                StandardScaler().fit_transform(summaries)
            )
            for segment_id, ((sequence, _, start, end, _), label) in enumerate(zip(all_segments, cluster_labels)):
                row_ids = sequence.row_ids[start:end]
                for row_id in row_ids:
                    labels_by_row[int(row_id)] = int(label)
                    segment_by_row[int(row_id)] = segment_id
                if end < len(sequence.frame):
                    change_by_row.add(int(sequence.row_ids[end]))
        else:
            cluster_labels = np.array([], dtype=int)

        row_ids = prepared.points["_hmmcma_row_id"]
        labels = row_ids.map(labels_by_row).fillna(-1).astype(int)
        prepared.points["cluster"] = labels
        prepared.points["state"] = labels
        prepared.points["segment_id"] = row_ids.map(segment_by_row).fillna(-1).astype(int)
        prepared.points["change_point"] = row_ids.isin(change_by_row)

        result = AnnotationResult(
            trajectory_collection=prepared.to_trajectory_collection(),
            labels=labels.to_numpy(),
            change_points=dict(change_points),
            metadata={
                "features": tuple(feature.value for feature in feature_list),
                "num_clusters": int(len(np.unique(cluster_labels))) if len(cluster_labels) else 0,
            },
        )
        self.last_result = result
        return result


BCPAAnnotator = BCPA
