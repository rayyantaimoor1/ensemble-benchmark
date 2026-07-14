"""
tuning.py
---------
Hyperparameter tuning for each model using RandomizedSearchCV.
For a portfolio/course project this is deliberately capped (n_iter, cv)
to keep runtime reasonable on a single machine; increase both for a more
exhaustive search if you have GPU/cluster access.
"""

import json
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from models import get_baseline_models, get_param_distributions, RANDOM_STATE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def tune_model(name, base_model, param_dist, X_train, y_train,
                n_iter=20, cv=3, scoring="f1_macro", n_jobs=-1):
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_all_models(X_train, y_train, n_iter=20, cv=3, save=True):
    """
    Tunes all four models and returns:
      - dict of name -> best fitted estimator
      - dict of name -> best hyperparameters
    Optionally saves the best hyperparameters to results/metrics/best_params.json
    """
    baselines = get_baseline_models()
    param_dists = get_param_distributions()

    best_models, best_params, best_scores = {}, {}, {}

    for name in baselines:
        print(f"\n[tuning] Searching hyperparameters for {name}...")
        best_est, params, score = tune_model(
            name, baselines[name], param_dists[name], X_train, y_train,
            n_iter=n_iter, cv=cv,
        )
        best_models[name] = best_est
        best_params[name] = params
        best_scores[name] = score
        print(f"[tuning]   best macro-F1 (CV): {score:.4f}")
        print(f"[tuning]   best params: {params}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "best_params.json")
        with open(out_path, "w") as f:
            json.dump({"best_params": best_params, "cv_scores": best_scores}, f, indent=2)
        print(f"\n[tuning] Saved best hyperparameters to {out_path}")

    return best_models, best_params
