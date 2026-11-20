import pandas as pd, numpy as np
from synaptix.automl import AutoPipeline
np.random.seed(0)
X = pd.DataFrame({
    "a": np.random.randn(20),
    "b": np.random.randn(20),
    "c": np.random.randn(20),
    "d": np.random.randn(20),
})
X["target"] = (X["a"] + X["b"] > 0).astype(int)
pipe = AutoPipeline(
    mode="fast",
    feature_search="exhaustive",
    feature_search_min_features=1,
    feature_search_max_features=2,
    feature_search_max_combinations=10,
    verbose=False,
    random_state=42,
)
pipe.fit(X, target="target")
print("best_features_", pipe.best_features_)
if getattr(pipe, "feature_search_results_", None) is not None:
    print("results rows", pipe.feature_search_results_.shape)
else:
    print("no results")
