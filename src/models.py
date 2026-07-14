"""
models.py
---------
Defines the four ensemble paradigms under comparison and their
hyperparameter search spaces for tuning:

  1. Random Forest        (Bagging)
  2. Gradient Boosting    (Vanilla / sequential boosting)
  3. XGBoost              (Optimized boosting, 2nd-order gradients)
  4. LightGBM             (Leaf-wise histogram-based boosting)
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
N_JOBS = -1


def get_baseline_models(n_classes: int = 7):
    """
    Returns a dict of {name: unfitted estimator} using sensible baseline
    hyperparameters (i.e. before tuning). Used for the initial "unified
    conditions" comparison in Phase 2.

    Note: GradientBoosting uses a smaller n_estimators/subsample than the
    other three models. Unlike XGBoost/LightGBM (histogram-based, highly
    memory-efficient) or RandomForest (embarrassingly parallel, trees are
    independent), sklearn's GradientBoostingClassifier builds trees
    sequentially without histogram binning, so its peak memory on 400k+
    rows with 7 classes can exceed what smaller machines (e.g. 8GB RAM
    laptops) have available. This asymmetry is itself part of RQ1
    (computational efficiency) -- it's expected that GradientBoosting
    looks the least scalable of the four; that's the finding, not a bug.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            objective="multiclass",
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=-1,
        ),
    }


def get_param_distributions():
    """
    Hyperparameter search spaces for RandomizedSearchCV, one per model.
    Kept intentionally compact so a full search finishes in reasonable
    time on a laptop; widen these if you have more compute available.
    """
    return {
        "RandomForest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300, 500],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
    }


MODEL_COLORS = {
    "RandomForest": "#2E86AB",
    "GradientBoosting": "#A23B72",
    "XGBoost": "#F18F01",
    "LightGBM": "#3B8C6E",
}
