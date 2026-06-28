"""Shared trajectory preparation for behavioural annotation methods.

Annotators operate on :class:`movingpandas.TrajectoryCollection` objects. The
trajectory collection supplies trajectory ids and timestamps; movement features
are calculated from canonical projected coordinate columns ``utm_x`` and
``utm_y``. The original geometry is retained in the returned collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd


PROJECTED_X_COL = "utm_x"
PROJECTED_Y_COL = "utm_y"
INTERNAL_TIME_COL = "_hmmcma_time"


class Feature(str, Enum):
    """Movement features available to the annotators."""

    DISTANCE = "distance"
    SPEED = "speed"
    DIRECTION = "direction"
    ANGULAR_DIFFERENCE = "angular_difference"
    TURN_ANGLE = "turn_angle"
    PERSISTENCE_VELOCITY = "persistence_velocity"
    TERRAIN = "terrain"


DEFAULT_FEATURES = (
    Feature.DISTANCE,
    Feature.ANGULAR_DIFFERENCE,
    Feature.SPEED,
)


@dataclass
class PreparedSequence:
    """One contiguous, feature-complete portion of an individual's track."""

    trajectory_id: object
    frame: pd.DataFrame

    @property
    def row_ids(self) -> np.ndarray:
        return self.frame["_hmmcma_row_id"].to_numpy()


@dataclass
class PreparedTrajectories:
    """Common input and output carrier for annotation methods."""

    points: gpd.GeoDataFrame
    sequences: list[PreparedSequence]
    id_col: str
    time_col: str
    index_name: str | None
    geometry_col: str
    crs: object

    def to_trajectory_collection(self) -> mpd.TrajectoryCollection:
        """Build an annotated collection without exposing implementation keys."""
        output = self.points.drop(columns=["_hmmcma_row_id"], errors="ignore").copy()
        output = gpd.GeoDataFrame(output, geometry=self.geometry_col, crs=self.crs)
        output = output.set_index(self.time_col)
        output.index.name = self.index_name
        return mpd.TrajectoryCollection(output, traj_id_col=self.id_col)


def normalise_features(features: Iterable[Feature | str] | Feature | str | None) -> tuple[Feature, ...]:
    """Validate feature input while accepting enum members and their strings."""
    if features is None:
        return DEFAULT_FEATURES
    if isinstance(features, (Feature, str)):
        features = (features,)

    result = []
    for feature in features:
        try:
            value = feature if isinstance(feature, Feature) else Feature(feature)
        except ValueError as exc:
            valid = ", ".join(item.value for item in Feature)
            raise ValueError(f"Unknown feature {feature!r}. Choose from: {valid}") from exc
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one feature must be selected.")
    return tuple(result)


def _resolve_trajectory_id(trajectory_collection: mpd.TrajectoryCollection, points: gpd.GeoDataFrame) -> str:
    id_col = trajectory_collection.get_traj_id_col()
    if not id_col or id_col not in points.columns:
        raise ValueError("The trajectory collection must expose a trajectory id column.")
    return id_col


def _temporary_utm_collection(points: gpd.GeoDataFrame, id_col: str, time_col: str) -> mpd.TrajectoryCollection:
    """Make a temporary planar collection used only for movingpandas features."""
    temporary = points.copy()
    temporary["geometry"] = gpd.points_from_xy(temporary["_hmmcma_x"], temporary["_hmmcma_y"])
    temporary = gpd.GeoDataFrame(temporary, geometry="geometry", crs="EPSG:3857")
    return mpd.TrajectoryCollection(temporary, traj_id_col=id_col, t=time_col)


def _add_movingpandas_features(
    points: gpd.GeoDataFrame, features: Sequence[Feature], id_col: str, time_col: str
) -> gpd.GeoDataFrame:
    """Use movingpandas' vectorised trajectory methods on UTM-derived geometry."""
    needed = set(features)
    if Feature.TURN_ANGLE in needed:
        needed.add(Feature.ANGULAR_DIFFERENCE)
    if not needed.intersection(
        {Feature.DISTANCE, Feature.SPEED, Feature.DIRECTION, Feature.ANGULAR_DIFFERENCE}
    ):
        return points

    temporary_collection = _temporary_utm_collection(points, id_col, time_col)
    if Feature.DISTANCE in needed:
        temporary_collection.add_distance(overwrite=True, name=Feature.DISTANCE.value)
    if Feature.SPEED in needed:
        temporary_collection.add_speed(overwrite=True, name=Feature.SPEED.value)
    if Feature.DIRECTION in needed or Feature.ANGULAR_DIFFERENCE in needed:
        temporary_collection.add_direction(overwrite=True, name=Feature.DIRECTION.value)
    if Feature.ANGULAR_DIFFERENCE in needed:
        temporary_collection.add_angular_difference(overwrite=True, name=Feature.ANGULAR_DIFFERENCE.value)

    calculated = temporary_collection.to_point_gdf().reset_index(drop=True)
    calculated = calculated.set_index("_hmmcma_row_id")
    for feature in (
        Feature.DISTANCE,
        Feature.SPEED,
        Feature.DIRECTION,
        Feature.ANGULAR_DIFFERENCE,
    ):
        if feature in needed:
            points[feature.value] = points["_hmmcma_row_id"].map(calculated[feature.value])
    if Feature.TURN_ANGLE in features:
        points[Feature.TURN_ANGLE.value] = np.deg2rad(
            points[Feature.ANGULAR_DIFFERENCE.value].fillna(0.0)
        )
    return points


def _add_persistence_velocity(points: gpd.GeoDataFrame, id_col: str) -> None:
    """Add velocity projected onto the previous unit direction for each track."""
    result = pd.Series(0.0, index=points.index)
    for _, group in points.groupby(id_col, sort=False):
        x = group["_hmmcma_x"].to_numpy(dtype=float)
        y = group["_hmmcma_y"].to_numpy(dtype=float)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        lengths = np.hypot(dx, dy)
        unit_x = np.divide(dx, lengths, out=np.zeros_like(dx), where=lengths > 0)
        unit_y = np.divide(dy, lengths, out=np.zeros_like(dy), where=lengths > 0)
        persistence = dx * np.roll(unit_x, 1) + dy * np.roll(unit_y, 1)
        persistence[0] = 0.0
        result.loc[group.index] = persistence
    points[Feature.PERSISTENCE_VELOCITY.value] = result


def _split_on_gaps(frame: pd.DataFrame, time_col: str, max_gap_factor: float) -> list[pd.DataFrame]:
    """Split a track at gaps larger than a multiple of its median interval."""
    if len(frame) < 2:
        return []
    deltas = frame[time_col].diff().dt.total_seconds()
    positive = deltas[deltas > 0]
    if positive.empty:
        return [frame]
    threshold = float(positive.median()) * max_gap_factor
    starts = np.flatnonzero((deltas > threshold).to_numpy())
    boundaries = [0, *starts.tolist(), len(frame)]
    return [frame.iloc[start:end].copy() for start, end in zip(boundaries, boundaries[1:]) if end - start >= 2]


def prepare_trajectory_collection(
    trajectory_collection: mpd.TrajectoryCollection,
    features: Iterable[Feature | str] | Feature | str | None = None,
    max_gap_factor: float = 5.0,
) -> PreparedTrajectories:
    """Apply the shared UTM feature and gap preprocessing pipeline."""
    if max_gap_factor <= 0:
        raise ValueError("max_gap_factor must be positive.")

    features = normalise_features(features)
    source = trajectory_collection.to_point_gdf().copy()
    id_col = _resolve_trajectory_id(trajectory_collection, source)
    index_name = source.index.name
    geometry_col = source.geometry.name

    points = source.reset_index(drop=True).copy()
    points[INTERNAL_TIME_COL] = pd.to_datetime(source.index.to_numpy())
    points["_hmmcma_row_id"] = np.arange(len(points), dtype=int)
    for column in (PROJECTED_X_COL, PROJECTED_Y_COL):
        if column not in points.columns:
            raise ValueError(f"Required projected coordinate column {column!r} is missing.")
    points["_hmmcma_x"] = pd.to_numeric(points[PROJECTED_X_COL], errors="coerce")
    points["_hmmcma_y"] = pd.to_numeric(points[PROJECTED_Y_COL], errors="coerce")

    valid = points.dropna(subset=["_hmmcma_x", "_hmmcma_y", INTERNAL_TIME_COL]).copy()
    valid = valid.sort_values([id_col, INTERNAL_TIME_COL]).reset_index(drop=True)
    valid = _add_movingpandas_features(valid, features, id_col, INTERNAL_TIME_COL)
    if Feature.PERSISTENCE_VELOCITY in features:
        _add_persistence_velocity(valid, id_col)
    if Feature.TERRAIN in features and Feature.TERRAIN.value not in valid.columns:
        raise ValueError("Feature.TERRAIN was requested, but no 'terrain' column is present.")

    for feature in features:
        if feature.value not in points.columns:
            points[feature.value] = np.nan
        values = valid.set_index("_hmmcma_row_id")[feature.value]
        points[feature.value] = points["_hmmcma_row_id"].map(values)

    sequences: list[PreparedSequence] = []
    for trajectory_id, group in valid.groupby(id_col, sort=False):
        for sequence in _split_on_gaps(group, INTERNAL_TIME_COL, max_gap_factor):
            sequences.append(PreparedSequence(trajectory_id=trajectory_id, frame=sequence))

    points = points.sort_values([id_col, INTERNAL_TIME_COL]).reset_index(drop=True)
    return PreparedTrajectories(
        points=gpd.GeoDataFrame(points, geometry=geometry_col, crs=source.crs),
        sequences=sequences,
        id_col=id_col,
        time_col=INTERNAL_TIME_COL,
        index_name=index_name,
        geometry_col=geometry_col,
        crs=source.crs,
    )


def feature_matrix(sequence: PreparedSequence, features: Iterable[Feature | str] | Feature | str | None) -> np.ndarray:
    """Return a finite numeric matrix for one already prepared sequence."""
    feature_names = [feature.value for feature in normalise_features(features)]
    return sequence.frame[feature_names].fillna(0.0).to_numpy(dtype=float)
