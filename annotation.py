"""Shared annotation result type and base class for behavioural methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable

import movingpandas as mpd
import numpy as np
import pandas as pd

try:  # Support both ``import hmmcma`` and running from the repository root.
    from .preprocessing import (
        DEFAULT_FEATURES,
        Feature,
        PreparedTrajectories,
        normalise_features,
        prepare_trajectory_collection,
    )
except ImportError:  # pragma: no cover - exercised by direct script usage
    from preprocessing import (
        DEFAULT_FEATURES,
        Feature,
        PreparedTrajectories,
        normalise_features,
        prepare_trajectory_collection,
    )


@dataclass
class AnnotationResult:
    """The uniform result returned by behavioural annotators.

    ``trajectory_collection`` contains the original points plus common output
    columns: ``state``, ``cluster``, ``segment_id`` and ``change_point``.
    Unmodelled points have ``-1`` for labels and ``False`` for change points.
    """

    trajectory_collection: mpd.TrajectoryCollection
    labels: np.ndarray
    change_points: dict[object, list]
    metadata: dict[str, Any] = field(default_factory=dict)


DEFAULT_STATE_FEATURE_STATS = ("count", "mean", "median", "std")


def _coerce_feature_name(feature: Feature | str) -> str:
    return feature.value if isinstance(feature, Feature) else str(feature)


def _as_feature_iterable(features: Iterable[Feature | str] | Feature | str | None):
    if features is None:
        return None
    if isinstance(features, (Feature, str)):
        return (features,)
    return features


def _resolve_summary_target(
    annotation: AnnotationResult | mpd.TrajectoryCollection | Any,
) -> tuple[AnnotationResult | None, mpd.TrajectoryCollection]:
    if isinstance(annotation, AnnotationResult):
        return annotation, annotation.trajectory_collection
    if isinstance(annotation, mpd.TrajectoryCollection):
        return None, annotation
    result = getattr(annotation, "last_result", None)
    if isinstance(result, AnnotationResult):
        return result, result.trajectory_collection
    raise TypeError("Expected an AnnotationResult, TrajectoryCollection, or annotator with a latest result.")


def _summary_feature_names(
    result: AnnotationResult | None,
    points: pd.DataFrame,
    features: Iterable[Feature | str] | None,
    default_features: Iterable[Feature | str] | None,
) -> list[str]:
    configured = features
    if configured is None and result is not None:
        configured = result.metadata.get("features")
    if configured is None:
        configured = default_features
    if configured is None:
        configured = [feature.value for feature in Feature if feature.value in points.columns]
    configured = _as_feature_iterable(configured)

    names = []
    for feature in configured:
        name = _coerce_feature_name(feature)
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("No feature columns were selected for the state summary.")

    missing = [name for name in names if name not in points.columns]
    if missing:
        raise ValueError(f"Cannot summarise missing feature columns: {', '.join(missing)}")
    return names


def state_feature_summary(
    annotation: AnnotationResult | mpd.TrajectoryCollection | Any,
    features: Iterable[Feature | str] | None = None,
    stats: Iterable[str] = DEFAULT_STATE_FEATURE_STATS,
    include_unmodelled: bool = False,
    default_features: Iterable[Feature | str] | None = None,
) -> pd.DataFrame:
    """Summarise selected feature distributions per annotated state.

    The returned frame has one row per state and MultiIndex columns of
    ``feature`` x ``statistic``, matching ``df.groupby("state")[features].agg``.
    """
    stats = tuple(stats)
    result, collection = _resolve_summary_target(annotation)
    points = collection.to_point_gdf().copy()
    if "state" not in points.columns:
        raise ValueError("Cannot build a state summary because the 'state' column is missing.")

    feature_names = _summary_feature_names(result, points, features, default_features)
    selected = points[["state", *feature_names]].copy()
    selected["state"] = pd.to_numeric(selected["state"], errors="coerce")
    selected = selected.dropna(subset=["state"])
    if not include_unmodelled:
        selected = selected[selected["state"] >= 0]
    selected["state"] = selected["state"].astype(int)
    for feature in feature_names:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")

    if selected.empty:
        columns = pd.MultiIndex.from_product([feature_names, tuple(stats)], names=["feature", "stat"])
        return pd.DataFrame(columns=columns).rename_axis("state")
    summary = selected.groupby("state", sort=True)[feature_names].agg(list(stats))
    summary.columns.names = ["feature", "stat"]
    return summary


def compare_state_feature_summaries(
    annotations: Mapping[str, AnnotationResult | mpd.TrajectoryCollection | Any],
    features: Iterable[Feature | str] | None = None,
    stats: Iterable[str] = DEFAULT_STATE_FEATURE_STATS,
    include_unmodelled: bool = False,
) -> pd.DataFrame:
    """Stack per-state feature summaries for several methods."""
    frames = []
    for method, annotation in annotations.items():
        default_features = getattr(annotation, "features", None)
        summary = state_feature_summary(
            annotation,
            features=features,
            stats=stats,
            include_unmodelled=include_unmodelled,
            default_features=default_features,
        )
        summary.index = pd.MultiIndex.from_product([[str(method)], summary.index], names=["method", "state"])
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


class BaseAnnotator(ABC):
    """Shared collection-to-features-to-annotated-collection pipeline."""

    def __init__(
        self,
        features: Iterable[Feature | str] | None = None,
        max_gap_factor: float = 5.0,
    ):
        self.features = normalise_features(features) if features is not None else None
        self.max_gap_factor = max_gap_factor
        self.last_result: AnnotationResult | None = None

    @property
    def result(self) -> AnnotationResult | None:
        """Alias for the latest annotation result."""
        return self.last_result

    def prepare(self, trajectory_collection: mpd.TrajectoryCollection) -> PreparedTrajectories:
        return prepare_trajectory_collection(
            trajectory_collection,
            features=self.features,
            max_gap_factor=self.max_gap_factor,
        )

    @staticmethod
    def initialise_annotations(prepared: PreparedTrajectories) -> None:
        points = prepared.points
        points["state"] = -1
        points["cluster"] = -1
        points["segment_id"] = -1
        points["change_point"] = False

    def plot(self, path, annotation: AnnotationResult | mpd.TrajectoryCollection | None = None, **kwargs) -> None:
        """Save a state plot to ``path`` from this method's latest result."""
        result = annotation or self.last_result
        if result is None:
            raise ValueError("Call annotate() first or pass an annotation result to plot().")
        try:
            from .plot import plot_states
        except ImportError:  # pragma: no cover - direct repository-root usage
            from plot import plot_states
        plot_states(result, path, **kwargs)

    def evaluation_segment_basis(self) -> str:
        """Return the segment family used for the primary segment metrics."""
        return "segment_id"

    def evaluation_metadata(self, annotation: AnnotationResult | mpd.TrajectoryCollection) -> dict[str, Any]:
        """Hook for method-specific evaluation details."""
        return {}

    def evaluate(self, annotation: AnnotationResult | mpd.TrajectoryCollection | None = None) -> dict[str, Any]:
        """Calculate behavioural segmentation metrics for an annotation result.

        ``num_segments`` and ``mean_segment_length`` use the method's primary
        segment basis.  ``segment_id`` metrics describe algorithm boundaries,
        while ``state_run`` metrics describe contiguous runs of the same state.
        Segment lengths are counts of annotated observations.
        """
        target = annotation or self.last_result
        if target is None:
            raise ValueError("Call annotate() first or pass an annotation result to evaluate().")
        collection = target.trajectory_collection if isinstance(target, AnnotationResult) else target
        points = self._evaluation_points(collection)
        self._require_evaluation_columns(points)

        segment_id_summary = self._segment_id_summary(points, collection)
        state_run_summary = self._state_run_summary(points, collection)
        primary_basis = self.evaluation_segment_basis()
        summaries = {
            "segment_id": segment_id_summary,
            "state_run": state_run_summary,
        }
        if primary_basis not in summaries:
            raise ValueError("evaluation_segment_basis() must return 'segment_id' or 'state_run'.")
        primary = summaries[primary_basis]

        modelled = pd.to_numeric(points["state"], errors="coerce") >= 0
        total_points = int(len(points))
        modelled_points = int(modelled.sum())
        state_counts = points.loc[modelled, "state"].astype(int).value_counts(sort=False).sort_index()
        num_change_points = int(points["change_point"].fillna(False).astype(bool).sum())
        transition_count = int(state_run_summary["transition_count"])

        metrics = {
            "method": type(self).__name__,
            "segment_basis": primary_basis,
            "total_points": total_points,
            "modelled_points": modelled_points,
            "unmodelled_points": total_points - modelled_points,
            "coverage_ratio": self._safe_ratio(modelled_points, total_points),
            "num_states": int(len(state_counts)),
            "state_counts": {int(state): int(count) for state, count in state_counts.items()},
            "state_proportions": {
                int(state): self._safe_ratio(int(count), modelled_points) for state, count in state_counts.items()
            },
            "num_change_points": num_change_points,
            "change_point_rate": self._safe_ratio(num_change_points, max(modelled_points - 1, 0)),
            "transition_count": transition_count,
            "transition_rate": self._safe_ratio(transition_count, max(modelled_points - 1, 0)),
            "num_segments": primary["count"],
            "mean_segment_length": primary["mean_length"],
            "median_segment_length": primary["median_length"],
            "min_segment_length": primary["min_length"],
            "max_segment_length": primary["max_length"],
            "std_segment_length": primary["std_length"],
            "mean_segment_duration_seconds": primary["mean_duration_seconds"],
            "median_segment_duration_seconds": primary["median_duration_seconds"],
            "num_segment_id_segments": segment_id_summary["count"],
            "mean_segment_id_length": segment_id_summary["mean_length"],
            "median_segment_id_length": segment_id_summary["median_length"],
            "mean_segment_id_duration_seconds": segment_id_summary["mean_duration_seconds"],
            "num_state_runs": state_run_summary["count"],
            "mean_state_run_length": state_run_summary["mean_length"],
            "median_state_run_length": state_run_summary["median_length"],
            "mean_state_run_duration_seconds": state_run_summary["mean_duration_seconds"],
            "segments_per_trajectory": primary["per_trajectory"],
            "mean_segments_per_trajectory": primary["mean_per_trajectory"],
            **self.evaluation_metadata(target),
        }
        return metrics

    def finalise_result(self, result: AnnotationResult) -> AnnotationResult:
        """Attach shared metrics and store the latest result."""
        result.metadata["evaluation"] = self.evaluate(result)
        result.metadata["state_feature_summary"] = self.state_feature_summary(result)
        self.last_result = result
        return result

    def state_feature_summary(
        self,
        annotation: AnnotationResult | mpd.TrajectoryCollection | None = None,
        features: Iterable[Feature | str] | None = None,
        stats: Iterable[str] = DEFAULT_STATE_FEATURE_STATS,
        include_unmodelled: bool = False,
    ) -> pd.DataFrame:
        """Return selected feature statistics grouped by annotated state."""
        target = annotation or self.last_result
        if target is None:
            raise ValueError("Call annotate() first or pass an annotation result to state_feature_summary().")
        return state_feature_summary(
            target,
            features=features,
            stats=stats,
            include_unmodelled=include_unmodelled,
            default_features=self.features or DEFAULT_FEATURES,
        )

    @staticmethod
    def _require_evaluation_columns(points: pd.DataFrame) -> None:
        missing = [column for column in ("state", "segment_id", "change_point") if column not in points.columns]
        if missing:
            raise ValueError(f"Cannot evaluate annotation; missing columns: {', '.join(missing)}")

    @staticmethod
    def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
        return float(numerator) / float(denominator) if denominator else None

    @staticmethod
    def _evaluation_points(collection: mpd.TrajectoryCollection) -> pd.DataFrame:
        points = collection.to_point_gdf().copy()
        points["_hmmcma_eval_order"] = np.arange(len(points), dtype=int)
        points["_hmmcma_eval_time"] = pd.to_datetime(points.index, errors="coerce")
        id_col = collection.get_traj_id_col()
        sort_cols = [id_col, "_hmmcma_eval_time", "_hmmcma_eval_order"] if id_col in points.columns else [
            "_hmmcma_eval_time",
            "_hmmcma_eval_order",
        ]
        return points.sort_values(sort_cols, kind="mergesort")

    @classmethod
    def _segment_id_summary(cls, points: pd.DataFrame, collection: mpd.TrajectoryCollection) -> dict[str, Any]:
        id_col = collection.get_traj_id_col()
        segment_ids = pd.to_numeric(points["segment_id"], errors="coerce")
        valid = points.loc[segment_ids >= 0].copy()
        if valid.empty:
            return cls._empty_segment_summary()

        group_cols = [id_col, "segment_id"] if id_col in valid.columns else ["segment_id"]
        group_key = group_cols if len(group_cols) > 1 else group_cols[0]
        records = []
        for keys, group in valid.groupby(group_key, sort=False, dropna=False):
            trajectory_id = keys[0] if isinstance(keys, tuple) else None
            if id_col in valid.columns and not isinstance(keys, tuple):
                trajectory_id = group[id_col].iloc[0]
            records.append(cls._segment_record(group, trajectory_id))
        return cls._summarise_segments(records)

    @classmethod
    def _state_run_summary(cls, points: pd.DataFrame, collection: mpd.TrajectoryCollection) -> dict[str, Any]:
        id_col = collection.get_traj_id_col()
        groups = points.groupby(id_col, sort=False, dropna=False) if id_col in points.columns else [(None, points)]
        records = []
        transition_count = 0

        for trajectory_id, group in groups:
            run_rows = []
            previous_state = None
            for _, row in group.iterrows():
                state = pd.to_numeric(pd.Series([row["state"]]), errors="coerce").iloc[0]
                state = int(state) if pd.notna(state) and state >= 0 else None
                if state is None:
                    if run_rows:
                        records.append(cls._segment_record(pd.DataFrame(run_rows), trajectory_id))
                        run_rows = []
                    previous_state = None
                    continue
                if previous_state is None or state != previous_state:
                    if run_rows:
                        records.append(cls._segment_record(pd.DataFrame(run_rows), trajectory_id))
                        transition_count += 1
                    run_rows = [row]
                    previous_state = state
                else:
                    run_rows.append(row)
            if run_rows:
                records.append(cls._segment_record(pd.DataFrame(run_rows), trajectory_id))

        summary = cls._summarise_segments(records)
        summary["transition_count"] = transition_count
        return summary

    @staticmethod
    def _segment_record(group: pd.DataFrame, trajectory_id: object) -> dict[str, Any]:
        start_time = group["_hmmcma_eval_time"].iloc[0]
        end_time = group["_hmmcma_eval_time"].iloc[-1]
        duration = None
        if pd.notna(start_time) and pd.notna(end_time):
            duration = float((end_time - start_time).total_seconds())
        return {
            "trajectory_id": trajectory_id,
            "length": int(len(group)),
            "duration_seconds": duration,
        }

    @classmethod
    def _summarise_segments(cls, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return cls._empty_segment_summary()

        lengths = np.asarray([record["length"] for record in records], dtype=float)
        durations = np.asarray(
            [record["duration_seconds"] for record in records if record["duration_seconds"] is not None],
            dtype=float,
        )
        per_trajectory: dict[str, int] = {}
        for record in records:
            key = str(record["trajectory_id"])
            per_trajectory[key] = per_trajectory.get(key, 0) + 1

        duration_stats = cls._stats(durations)
        return {
            "count": int(len(records)),
            "trajectory_count": int(len(per_trajectory)),
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "std_length": float(np.std(lengths)),
            "mean_duration_seconds": duration_stats["mean"],
            "median_duration_seconds": duration_stats["median"],
            "min_duration_seconds": duration_stats["min"],
            "max_duration_seconds": duration_stats["max"],
            "std_duration_seconds": duration_stats["std"],
            "per_trajectory": per_trajectory,
            "mean_per_trajectory": float(np.mean(list(per_trajectory.values()))),
            "transition_count": 0,
        }

    @staticmethod
    def _stats(values: np.ndarray) -> dict[str, float | None]:
        if len(values) == 0:
            return {"mean": None, "median": None, "min": None, "max": None, "std": None}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }

    @staticmethod
    def _empty_segment_summary() -> dict[str, Any]:
        return {
            "count": 0,
            "trajectory_count": 0,
            "mean_length": None,
            "median_length": None,
            "min_length": None,
            "max_length": None,
            "std_length": None,
            "mean_duration_seconds": None,
            "median_duration_seconds": None,
            "min_duration_seconds": None,
            "max_duration_seconds": None,
            "std_duration_seconds": None,
            "per_trajectory": {},
            "mean_per_trajectory": None,
            "transition_count": 0,
        }

    @abstractmethod
    def annotate(self, trajectory_collection: mpd.TrajectoryCollection) -> AnnotationResult:
        """Annotate a collection and return the shared result shape."""
