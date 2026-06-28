import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import pandas as pd
from shapely.geometry import Point

from BCPA import BCPA
from plot import plot_states
from preprocessing import Feature
from state_annotation import HMM
from hybrid_annotation import BCPAHMM
from annotation import compare_state_feature_summaries


def relabel_states(states: pd.Series, mapping: Mapping[int, int] | Sequence[int] | None) -> pd.Series:
    """Relabel HMM states for comparison plots.

    Examples
    --------
    ``{1: 0, 0: 1}`` swaps two states and leaves unspecified states unchanged.
    ``[1, 0, 2]`` means old state 0 -> 1, old state 1 -> 0, old state 2 -> 2.
    """
    states = states.astype(int)
    if mapping is None:
        return states
    if isinstance(mapping, Mapping):
        state_map = {int(old): int(new) for old, new in mapping.items()}
    else:
        state_map = {old: int(new) for old, new in enumerate(mapping)}
    return states.replace(state_map).astype(int)


def load_movehmm_output(
    path: str | Path,
    state_mapping: Mapping[int, int] | Sequence[int] | None = None,
) -> mpd.TrajectoryCollection:
    """Load the moveHMM reference CSV as an annotated trajectory collection."""
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)
    df["state"] = relabel_states(df["state"], state_mapping)

    crs = df["utm_crs"].dropna().iloc[0] if "utm_crs" in df and df["utm_crs"].notna().any() else "EPSG:32633"
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["utm_x"], df["utm_y"])],
        crs=crs,
    )
    gdf = gdf.sort_values(["individual_local_identifier", "timestamp_utc"])
    return mpd.TrajectoryCollection(
        gdf,
        traj_id_col="individual_local_identifier",
        t="timestamp_utc",
    )



movehmm_collection = load_movehmm_output("output_states.csv", state_mapping={0: 1, 1: 0})
plot_states(movehmm_collection, path="plots/movehmm.png")
