from dataclasses import dataclass

from .models import apply_hmm
from .preprocessing import ColumnConfig, preprocess_hmm
from .utils import merge_states_to_gdf


@dataclass
class HMMStateAnnotator:
    columns: ColumnConfig
    scale: bool = True
    num_states: int = 3

    def annotate(self, gdf):
        arrays, _, seq_dfs = preprocess_hmm(gdf, self.columns, self.scale)
        trajectories, threshold, state_mapping = apply_hmm(
            arrays,
            seq_dfs,
            n_components=self.num_states,
            columns=self.columns,
        )
        annotated = merge_states_to_gdf(gdf, seq_dfs, self.columns)
        return annotated, trajectories, threshold, state_mapping


def annotate_states_gdf(
    gdf,
    id_cols="individual_local_identifier",
    time_col="timestamp",
    geom_col="geometry",
    provided_dir_col="direction",
    feature_cols=("distance", "angular_difference", "speed"),
    scale=True,
    num_states=3,
):
    columns = ColumnConfig(
        id_cols=id_cols,
        time_col=time_col,
        geom_col=geom_col,
        provided_dir_col=provided_dir_col,
        feature_cols=feature_cols,
    )
    annotator = HMMStateAnnotator(columns=columns, scale=scale, num_states=num_states)
    return annotator.annotate(gdf)

def annotate_states(traj_col, num_states=3):
    import movingpandas as mpd
    traj_col.add_speed()
    traj_col.add_angular_difference()
    traj_col.add_distance()
    traj_col.add_direction()
    annotated, trajectories, threshold, state_mapping = annotate_states_gdf(gdf=traj_col.to_point_gdf(), num_states=num_states)
    output = mpd.TrajectoryCollection(annotated, traj_col.get_traj_id_col())
    print(state_mapping)
    return output
