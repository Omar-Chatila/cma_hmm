"""Shared annotation result type and base class for behavioural methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import movingpandas as mpd
import numpy as np

try:  # Support both ``import hmmcma`` and running from the repository root.
    from .preprocessing import ColumnConfig, Feature, PreparedTrajectories, prepare_trajectory_collection
except ImportError:  # pragma: no cover - exercised by direct script usage
    from preprocessing import ColumnConfig, Feature, PreparedTrajectories, prepare_trajectory_collection


@dataclass
class AnnotationResult:
    """The uniform result returned by :class:`HMM` and :class:`BCPA`.

    ``trajectory_collection`` contains the original points plus common output
    columns: ``state``, ``cluster``, ``segment_id`` and ``change_point``.
    Unmodelled points have ``-1`` for labels and ``False`` for change points.
    """

    trajectory_collection: mpd.TrajectoryCollection
    labels: np.ndarray
    change_points: dict[object, list]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAnnotator(ABC):
    """Shared collection-to-features-to-annotated-collection pipeline."""

    def __init__(
        self,
        features: Iterable[Feature | str] | None = None,
        columns: ColumnConfig | None = None,
        max_gap_factor: float = 5.0,
    ):
        self.features = tuple(features) if features is not None else None
        self.columns = columns or ColumnConfig()
        self.max_gap_factor = max_gap_factor
        self.last_result: AnnotationResult | None = None

    def prepare(self, trajectory_collection: mpd.TrajectoryCollection) -> PreparedTrajectories:
        return prepare_trajectory_collection(
            trajectory_collection,
            features=self.features,
            columns=self.columns,
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

    @abstractmethod
    def annotate(self, trajectory_collection: mpd.TrajectoryCollection) -> AnnotationResult:
        """Annotate a collection and return the shared result shape."""
