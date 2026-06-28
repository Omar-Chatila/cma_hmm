# hmmcma

Behavioural annotation for `movingpandas.TrajectoryCollection` objects.

Input collections are expected to expose their trajectory id through
MovingPandas and to contain projected coordinate columns `utm_x` and `utm_y`.
Timestamps are taken from the point index.

```python
from hmmcma import BCPA, BCPAHMM, Feature, HMM, compare_state_feature_summaries

features = [Feature.PERSISTENCE_VELOCITY, Feature.DISTANCE, Feature.TURN_ANGLE]

hmm = HMM(features=features, num_states=3)
bcpa = BCPA(features=features, penalty=10, num_clusters=3)
hybrid = BCPAHMM(features=features, penalty=10, num_states=3)

hmm_result = hmm.annotate(trajectory_collection)
bcpa_result = bcpa.annotate(trajectory_collection)
hybrid_result = hybrid.annotate(trajectory_collection)
```

All results contain the selected feature columns plus `state`, `cluster`,
`segment_id`, and `change_point`.

`BCPAHMM` fits the HMM normally, uses BCPA for temporal segment boundaries, and
assigns one HMM state to each BCPA segment. Segment state assignment uses
posterior probability sums by default; pass
`segment_state_method="majority_vote"` to use hard Viterbi states.

## Diagnostics

```python
metrics = hybrid.evaluate()
state_summary = hybrid.state_feature_summary(features=features)

comparison = compare_state_feature_summaries(
    {
        "hmm": hmm_result,
        "bcpa": bcpa_result,
        "hybrid": hybrid_result,
    },
    features=features,
)
```

`evaluate()` reports segment counts, segment lengths, state counts, transition
counts, change-point counts, and coverage. `state_feature_summary()` reports
`count`, `mean`, `median`, and `std` per selected feature and state.

```python
hybrid.plot("states.png")
```
