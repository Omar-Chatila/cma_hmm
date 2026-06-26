"""Shared trajectory preparation for behavioural annotation methods.

The original geometry is kept for visualisation and for the returned
``TrajectoryCollection``.  Movement measurements, however, are always made
from the configured projected coordinate columns (``utm_x`` and ``utm_y`` by
default).  This prevents a geographic geometry column from accidentally being
used by one method but not another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd


class Feature(str, Enum):
    """Features understood by :class:`HMM` and :class:`BCPA`.

    Values are also the column names added to the returned point data.
    ``TERRAIN`` is a pass-through feature: the input collection must already
    contain a ``terrain`` column.
    """

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
class ColumnConfig:
    """Names of the input columns used by the common preprocessing pipeline.

    ``id_col`` and ``time_col`` may be left as ``None`` to use the trajectory
    collection's id column and point-data index respectively.  ``geom_col`` is
    deliberately not used to calculate features.
    """

    id_cols: str | None = None
    time_col: str | None = None
    geom_col: str = "geometry"
    x_col: str = "utm_x"
    y_col: str = "utm_y"
    provided_dir_col: str | None = None
    feature_cols: Sequence[Feature | str] = DEFAULT_FEATURES

    @property
    def id_col(self) -> str | None:
        """Backward-compatible singular spelling."""
        return self.id_cols


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
    columns: ColumnConfig
    id_col: str
    time_col: str
    index_name: str | None
    crs: object

    def to_trajectory_collection(self) -> mpd.TrajectoryCollection:
        """Build an annotated collection without exposing implementation keys."""
        output = self.points.drop(columns=["_hmmcma_row_id"], errors="ignore").copy()
        output = gpd.GeoDataFrame(output, geometry=self.columns.geom_col, crs=self.crs)
        output = output.set_index(self.time_col)
        output.index.name = self.index_name
        return mpd.TrajectoryCollection(output, traj_id_col=self.id_col)


def normalise_features(features: Iterable[Feature | str] | None) -> tuple[Feature, ...]:
    """Validate feature input while accepting enum members and their strings."""
    if features is None:
        return DEFAULT_FEATURES
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


def _resolve_columns(
    trajectory_collection: mpd.TrajectoryCollection, points: gpd.GeoDataFrame, columns: ColumnConfig
) -> tuple[str, str]:
    id_col = columns.id_col or trajectory_collection.get_traj_id_col()
    if not id_col or id_col not in points.columns:
        raise ValueError("Could not determine a trajectory id column.")

    # Point data normally stores time in the index.  A configured column takes
    # precedence only when it is actually available.
    if columns.time_col and columns.time_col in points.columns:
        time_col = columns.time_col
    else:
        time_col = points.index.name or "timestamp"
    return id_col, time_col


def _temporary_utm_collection(points: gpd.GeoDataFrame, id_col: str, time_col: str) -> mpd.TrajectoryCollection:
    """Make a temporary planar collection used only for movingpandas features."""
    temporary = points.copy()
    temporary["geometry"] = gpd.points_from_xy(temporary["_hmmcma_x"], temporary["_hmmcma_y"])
    # The CRS is intentionally planar.  The source geometry can be geographic
    # while x/y are UTM metres, and must not influence feature computation.
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
    features: Iterable[Feature | str] | None = None,
    columns: ColumnConfig | None = None,
    max_gap_factor: float = 5.0,
) -> PreparedTrajectories:
    """Apply the shared UTM feature and gap preprocessing pipeline.

    The returned points retain every source point.  Points with missing UTM
    coordinates are kept in the final collection but are not sent to a model
    and consequently retain the annotation value ``-1``.
    """
    if max_gap_factor <= 0:
        raise ValueError("max_gap_factor must be positive.")
    columns = columns or ColumnConfig()
    features = normalise_features(features if features is not None else columns.feature_cols)
    source = trajectory_collection.to_point_gdf().copy()
    id_col, time_col = _resolve_columns(trajectory_collection, source, columns)
    index_name = source.index.name

    points = source.reset_index(drop=True).copy()
    if time_col in source.columns:
        points[time_col] = source[time_col].to_numpy()
    else:
        points[time_col] = source.index.to_numpy()
    points[time_col] = pd.to_datetime(points[time_col])
    points["_hmmcma_row_id"] = np.arange(len(points), dtype=int)
    for col in (columns.x_col, columns.y_col):
        if col not in points.columns:
            raise ValueError(f"Required projected coordinate column {col!r} is missing.")
    points["_hmmcma_x"] = pd.to_numeric(points[columns.x_col], errors="coerce")
    points["_hmmcma_y"] = pd.to_numeric(points[columns.y_col], errors="coerce")

    valid = points.dropna(subset=["_hmmcma_x", "_hmmcma_y", time_col]).copy()
    valid = valid.sort_values([id_col, time_col]).reset_index(drop=True)
    valid = _add_movingpandas_features(valid, features, id_col, time_col)
    if Feature.PERSISTENCE_VELOCITY in features:
        _add_persistence_velocity(valid, id_col)
    if Feature.TERRAIN in features and Feature.TERRAIN.value not in valid.columns:
        raise ValueError("Feature.TERRAIN was requested, but no 'terrain' column is present.")

    # Feature columns are deliberately available on the returned collection.
    for feature in features:
        if feature.value not in points.columns:
            points[feature.value] = np.nan
        values = valid.set_index("_hmmcma_row_id")[feature.value]
        points[feature.value] = points["_hmmcma_row_id"].map(values)

    sequences: list[PreparedSequence] = []
    for trajectory_id, group in valid.groupby(id_col, sort=False):
        for sequence in _split_on_gaps(group, time_col, max_gap_factor):
            sequences.append(PreparedSequence(trajectory_id=trajectory_id, frame=sequence))

    points = points.sort_values([id_col, time_col]).reset_index(drop=True)
    return PreparedTrajectories(
        points=gpd.GeoDataFrame(points, geometry=columns.geom_col, crs=source.crs),
        sequences=sequences,
        columns=columns,
        id_col=id_col,
        time_col=time_col,
        index_name=index_name,
        crs=source.crs,
    )


def feature_matrix(sequence: PreparedSequence, features: Iterable[Feature | str]) -> np.ndarray:
    """Return a finite numeric matrix for one already prepared sequence."""
    feature_names = [feature.value for feature in normalise_features(features)]
    return sequence.frame[feature_names].fillna(0.0).to_numpy(dtype=float)


# Compatibility adapter for callers using the previous gdf-based helper.
def preprocess_hmm(gdf, columns: ColumnConfig | None = None, scale: bool = True):
    """Prepare a GeoDataFrame for the legacy ``apply_hmm`` function.

    New code should instantiate :class:`state_annotation.HMM` and call
    ``annotate`` with a ``TrajectoryCollection``.
    """
    from sklearn.preprocessing import StandardScaler

    columns = columns or ColumnConfig()
    id_col = columns.id_col
    if not id_col:
        raise ValueError("ColumnConfig.id_cols is required when preprocessing a GeoDataFrame directly.")
    collection = mpd.TrajectoryCollection(gdf, traj_id_col=id_col, t=columns.time_col)
    prepared = prepare_trajectory_collection(collection, columns.feature_cols, columns)
    arrays = [feature_matrix(sequence, columns.feature_cols) for sequence in prepared.sequences]
    scaler = StandardScaler().fit(np.vstack(arrays)) if scale and arrays else None
    if scaler is not None:
        arrays = [scaler.transform(array) for array in arrays]
    return arrays, scaler, [sequence.frame for sequence in prepared.sequences]
