import pickle

from BCPA import BCPA
from preprocessing import Feature
from state_annotation import HMM
from hybrid_annotation import BCPAHMM
from annotation import compare_state_feature_summaries

trajectory_collection = pickle.load(open("tests/annotated.pickle", "rb"))
df = trajectory_collection.to_point_gdf()
df.to_csv("tests/annotated.csv")

features = [Feature.PERSISTENCE_VELOCITY, Feature.DISTANCE, Feature.TURN_ANGLE]
NUM_STATES = 3

hmm = HMM(features=features, num_states=NUM_STATES)
bcpa = BCPA(features=features, penalty=10, num_clusters=NUM_STATES)
hybrid = BCPAHMM(features=features, penalty=10, num_states=NUM_STATES)

hmm_result = hmm.annotate(trajectory_collection)
bcpa_result = bcpa.annotate(trajectory_collection)
hybrid_result = hybrid.annotate(trajectory_collection)

bcpa.plot(path="plots/bcpa.png")
hmm.plot(path="plots/hmm.png")
hybrid.plot(path="plots/bcpahmm.png")

hmm_metrics = hmm.evaluate()
bcpa_metrics = bcpa.evaluate()
hybrid_metrics = hybrid.evaluate()

print(hmm_metrics)
print(bcpa_metrics)
print(hybrid_metrics)

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

print(comparison)
