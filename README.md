# hmmcma

`HMM` and `BCPA` share one MovingPandas trajectory preprocessing pipeline.
Movement features are calculated from `utm_x` and `utm_y`; the original
geometry is retained only in the returned annotated collection.

```python
from BCPA import BCPA
from preprocessing import Feature
from state_annotation import HMM

features = [Feature.SPEED, Feature.ANGULAR_DIFFERENCE, Feature.PERSISTENCE_VELOCITY]

hmm = HMM(features=features, num_states=3)
bcpa = BCPA(features=features, penalty=10, num_clusters=3)
hmm_result = hmm.annotate(trajectory_collection)
bcpa_result = bcpa.annotate(trajectory_collection)

# Both result objects have the same shape.
annotated = bcpa_result.trajectory_collection
points = annotated.to_point_gdf()
```

Both outputs contain the selected feature columns plus `state`, `cluster`,
`segment_id`, and `change_point`.  BCPA sets `change_point` at the first point
of each new detected segment; HMM leaves it `False` so the schemas remain
interchangeable.  See `Feature` for all available features and `ColumnConfig`
to configure id, time, or UTM column names.

## State plots

```python
bcpa.plot("states.png")
```

The top panel draws trajectory points in one colour per shared `state`; the
bottom panel shows the same states as horizontal, time-based bars. Both `HMM`
and `BCPA` inherit the same `plot()` method. To plot a different result, use
`bcpa.plot("other-states.png", annotation=other_result)`.
