class ColumnConfig:
    def __init__(
        self,
        id_cols="individual-local-identifier",
        time_col="timestamp",
        geom_col="geometry",
        provided_dir_col="direction",
        feature_cols=("distance", "angular_difference", "speed", "terrain"),
    ):
        self.time_col = time_col
        self.geom_col = geom_col
        self.id_col = id_cols
        self.provided_dir_col = provided_dir_col
        self.feature_cols = feature_cols


def preprocess_hmm(gdf, columns: ColumnConfig, scale=True):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from .utils import angle_diff

    time_col = columns.time_col
    geom_col = columns.geom_col
    id_cols = columns.id_col
    provided_dir_col = columns.provided_dir_col
    feature_cols = columns.feature_cols

    df = gdf.copy()
    df = df.reset_index()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna(subset=[time_col, geom_col])

    grouped = df.groupby(id_cols)
    print(f"Found {len(grouped)} animals")
    seq_dfs = []
    for _, group in grouped:
        group = group.sort_values(time_col)
        group[time_col] = pd.to_datetime(group[time_col])
        group["dt"] = group[time_col].diff().dt.total_seconds().fillna(0.0)

        xs = group.geometry.x.values
        ys = group.geometry.y.values
        dx = np.concatenate([[0.0], np.diff(xs)])
        dy = np.concatenate([[0.0], np.diff(ys)])
        group["step_length"] = np.sqrt(dx**2 + dy**2)

        group["speed"] = group["speed"].fillna(0.0).astype(float)
        headings = np.deg2rad(group[provided_dir_col].fillna(0.0).astype(float).values)
        if len(headings) > 1 and np.isnan(headings[0]):
            headings[0] = headings[1]
        turn_angles = np.concatenate(
            [[0.0], [angle_diff(headings[i - 1], headings[i]) for i in range(1, len(headings))]]
        )
        group["turn_angle"] = turn_angles
        group["heading_rad"] = headings

        local_dts = group["dt"].values[1:]
        dt_mode = np.median(local_dts)
        max_gap_seconds = 5 * dt_mode

        gaps = group["dt"].values
        split_idx = np.where(gaps > max_gap_seconds)[0]
        start = 0
        for idx in split_idx:
            seq = group.iloc[start:idx].copy().reset_index(drop=True)
            if len(seq) >= 2:
                seq_dfs.append(seq)
            start = idx

        final = group.iloc[start:].copy().reset_index(drop=True)
        if len(final) >= 2:
            seq_dfs.append(final)

    arrays = []
    for seq in seq_dfs:
        x = seq[list(feature_cols)].fillna(0.0).to_numpy(dtype=float)
        if x.shape[0] >= 2:
            arrays.append(x)

    scaler = None
    if scale and len(arrays) > 0:
        stacked = np.vstack(arrays)
        scaler = StandardScaler().fit(stacked)
        arrays = [scaler.transform(x) for x in arrays]
    return arrays, scaler, seq_dfs


def process_trajectories(data_list, columns=None):
    import numpy as np
    from collections import Counter

    from .utils import detect_typical_interval

    animal_trajectories = {}
    print("Length" + str(len(data_list)))

    for data in data_list:
        for _, row in data.iterrows():
            animal_id = row[columns.id_col]
            if animal_id not in animal_trajectories:
                animal_trajectories[animal_id] = []
            animal_trajectories[animal_id].append(
                (int(row["geometry"].x), int(row["geometry"].y), row[columns.time_col], row["state"] + 1)
            )

    for animal_id in animal_trajectories:
        animal_trajectories[animal_id].sort(key=lambda entry: entry[2])

    all_states = []
    for _, entries in animal_trajectories.items():
        for row in entries:
            all_states.append(row[3])

    print(Counter(all_states))
    intervals = []
    for _, entries in animal_trajectories.items():
        val = detect_typical_interval(entries)
        if val:
            intervals.append(val)
    print("alle intervalle", intervals)
    global_interval = np.median(intervals)
    print("Erkannter Zeitabstand:", global_interval)

    return animal_trajectories, global_interval
