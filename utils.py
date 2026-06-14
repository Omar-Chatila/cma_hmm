from collections import Counter


def angle_diff(a, b):
    import numpy as np

    d = b - a
    return (d + np.pi) % (2 * np.pi) - np.pi


def detect_typical_interval(entries):
    diffs = []
    for i in range(1, len(entries)):
        delta = ((entries[i][2] - entries[i - 1][2]).total_seconds() + 0.5) // 60
        diffs.append(delta)
    if len(diffs) == 0:
        return None
    return Counter(diffs).most_common(1)[0][0]


def merge_states_to_gdf(gdf, seq_dfs, columns):
    import geopandas as gpd
    import pandas as pd

    state_rows = []
    for seq in seq_dfs:
        if {columns.time_col, "state"}.issubset(seq.columns):
            state_rows.append(seq[[columns.time_col, "state"]])

    if not state_rows:
        gdf["state"] = -1
        return gdf

    states_df = (
        pd.concat(state_rows, ignore_index=True)
        .drop_duplicates(subset=columns.time_col)
    )

    gdf_tmp = gdf.reset_index()
    gdf_tmp = gdf_tmp.merge(
        states_df,
        on=columns.time_col,
        how="left",
    )
    assigned = gdf_tmp["state"].notna().sum()
    print(f"States zugewiesen: {assigned} von {len(gdf_tmp)} Punkten")

    gdf_tmp = gdf_tmp.sort_values(
        by=[columns.id_col, columns.time_col]
    )

    gdf_tmp["state"] = (
        gdf_tmp
        .groupby(columns.id_col)["state"]
        .ffill()
        .bfill()
        .fillna(-1)
        .astype(int)
    )
    gdf_out = gpd.GeoDataFrame(
        gdf_tmp,
        geometry=columns.geom_col,
        crs=gdf.crs,
    ).set_index(columns.time_col)

    print("\nFinal State-Distribution:")
    print(gdf_out["state"].value_counts().sort_index())

    return gdf_out
