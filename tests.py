import pickle
import movingpandas as mpd

from move_apps_patch import apply_moveapps_id_dtype_patch
from state_annotation import annotate_states

trajcol:mpd.TrajectoryCollection =pickle.load(open("tests/annotated.pickle", "rb"))

apply_moveapps_id_dtype_patch()

num_states=4

annotated = annotate_states(trajcol, num_states=num_states)

pickle.dump(annotated, open("tests/annotated_new.pickle", "wb"))
