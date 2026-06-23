import pickle
from hmmcma import *
import movingpandas as mpd


trajcol:mpd.TrajectoryCollection =pickle.load(open("tests/turtles.pickle", "rb"))

apply_moveapps_id_dtype_patch()

num_states=3

annotate_states(trajcol, num_states=num_states)