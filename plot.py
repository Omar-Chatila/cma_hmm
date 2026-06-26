"""Plots for the common HMM/BCPA state annotations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def _point_data(annotation):
    """Accept either an AnnotationResult or an annotated collection."""
    collection = getattr(annotation, "trajectory_collection", annotation)
    if not hasattr(collection, "to_point_gdf"):
        raise TypeError("annotation must be an AnnotationResult or a movingpandas TrajectoryCollection.")
    points = collection.to_point_gdf().copy()
    id_col = collection.get_traj_id_col()
    if not id_col or id_col not in points.columns:
        raise ValueError("Could not determine the trajectory id column.")
    if "state" not in points.columns:
        raise ValueError("The collection is not annotated: missing 'state'.")
    return points, id_col


def _state_colours(states: pd.Series, cmap: str) -> Mapping[int, tuple]:
    valid_ids = sorted(int(state) for state in states.dropna().unique() if int(state) >= 0)
    colour_map = plt.get_cmap(cmap, max(len(valid_ids), 1))
    colours = {segment: colour_map(index) for index, segment in enumerate(valid_ids)}
    colours[-1] = (0.65, 0.65, 0.65, 1.0)  # points omitted by preprocessing
    return colours


def _runs(frame: pd.DataFrame, time_col: str):
    """Yield contiguous (state, start, end) pieces for a timeline row."""
    timestamps = frame[time_col].to_numpy()
    states = frame["state"].fillna(-1).astype(int).to_numpy()
    if len(frame) == 0:
        return
    intervals = pd.Series(timestamps).diff().dt.total_seconds().dropna()
    final_width = float(intervals[intervals > 0].median()) if not intervals.empty else 1.0
    final_width = final_width if np.isfinite(final_width) and final_width > 0 else 1.0

    start = 0
    for index in range(1, len(frame) + 1):
        if index != len(frame) and states[index] == states[start]:
            continue
        start_time = pd.Timestamp(timestamps[start])
        end_time = pd.Timestamp(timestamps[index]) if index < len(frame) else start_time + pd.Timedelta(seconds=final_width)
        yield states[start], start_time, end_time
        start = index


def plot_states(
    annotation,
    path: str | Path,
    x_col: str = "utm_x",
    y_col: str = "utm_y",
    cmap: str = "tab20",
    figsize: tuple[float, float] = (12, 9),
    show_legend: bool = True,
    dpi: int = 200,
) -> None:
    """Save spatial state points and state timelines to ``path``.

    Parameters
    ----------
    annotation:
        An :class:`annotation.AnnotationResult` or the annotated
        ``movingpandas.TrajectoryCollection`` returned by HMM/BCPA.
    x_col, y_col:
        Coordinate columns for panel A.  They default to the common UTM
        coordinates, but any two numeric columns can be used.

    """
    points, id_col = _point_data(annotation)
    for column in (x_col, y_col):
        if column not in points.columns:
            raise ValueError(f"Missing coordinate column {column!r}.")
    if not isinstance(points.index, pd.DatetimeIndex):
        raise ValueError("The annotated collection must have a DatetimeIndex for the timeline plot.")

    time_col = "_hmmcma_plot_time"
    points[time_col] = pd.to_datetime(points.index)
    points = points.sort_values([id_col, time_col])
    colours = _state_colours(points["state"], cmap)
    fig, (trajectory_axis, timeline_axis) = plt.subplots(
        2, 1, figsize=figsize, constrained_layout=True, gridspec_kw={"height_ratios": (3, 1)}
    )

    # A: retain each track's shape with a subtle line, and show state colour
    # directly on its points.
    for trajectory_id, frame in points.groupby(id_col, sort=False):
        trajectory_axis.plot(frame[x_col], frame[y_col], color="0.75", linewidth=0.8, zorder=1)
        point_colours = [colours.get(int(state), colours[-1]) for state in frame["state"].fillna(-1)]
        trajectory_axis.scatter(frame[x_col], frame[y_col], c=point_colours, s=14, zorder=2)
    trajectory_axis.set(title="Trajectory states", xlabel=x_col, ylabel=y_col, aspect="equal")

    # B: one horizontal track per individual, one coloured bar per contiguous
    # state.  ``date2num`` lets Matplotlib render an actual time axis.
    trajectory_ids = list(points[id_col].drop_duplicates())
    for y_position, trajectory_id in enumerate(trajectory_ids):
        frame = points.loc[points[id_col] == trajectory_id]
        for state, start, end in _runs(frame, time_col):
            start_number = mdates.date2num(start)
            width = mdates.date2num(end) - start_number
            timeline_axis.barh(
                y_position,
                width,
                left=start_number,
                height=0.7,
                color=colours.get(state, colours[-1]),
                edgecolor="none",
            )
    timeline_axis.set(
        title="State timeline",
        xlabel="Time",
        ylabel="Trajectory",
        yticks=range(len(trajectory_ids)),
        yticklabels=[str(trajectory_id) for trajectory_id in trajectory_ids],
    )
    timeline_axis.xaxis_date()
    timeline_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(timeline_axis.xaxis.get_major_locator()))

    if show_legend:
        handles = [Patch(color=colour, label=f"state {state}") for state, colour in colours.items()]
        trajectory_axis.legend(handles=handles, title="States", loc="best", fontsize="small")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# Previous public name kept as a compatibility alias.  The plot is now based
# on the shared state column rather than BCPA-specific segment identifiers.
plot_segments = plot_states
