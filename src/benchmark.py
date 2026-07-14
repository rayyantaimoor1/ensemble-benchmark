"""
benchmark.py
------------
Core benchmarking utilities: wall-clock training time, memory usage,
inference latency, and predictive-performance metrics (accuracy,
macro-F1), plus the scalability sweep across increasing training-set
sizes (10k / 50k / 100k / 500k rows).
"""

import gc
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_pipeline import stratified_subsample


def _peak_memory_mb(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs), returning (result, peak_memory_in_MB)."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 ** 2)


def train_and_time(model, X_train, y_train):
    """Fits a model, returning (fitted_model, train_seconds, peak_mem_mb)."""
    start = time.perf_counter()
    fitted, peak_mem = _peak_memory_mb(model.fit, X_train, y_train)
    elapsed = time.perf_counter() - start
    return fitted, elapsed, peak_mem


def predict_and_time(model, X_test):
    """Runs inference, returning (predictions, inference_seconds)."""
    start = time.perf_counter()
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return preds, elapsed


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def benchmark_model(name, model, X_train, y_train, X_test, y_test):
    """
    Full single-run benchmark for one model: fit, predict, time, and score.
    Returns a flat dict suitable for appending to a results DataFrame.
    """
    fitted, train_time, train_mem = train_and_time(model, X_train, y_train)
    preds, infer_time = predict_and_time(fitted, X_test)
    metrics = evaluate_predictions(y_test, preds)

    return {
        "model": name,
        "train_rows": len(X_train),
        "train_time_sec": train_time,
        "peak_train_memory_mb": train_mem,
        "inference_time_sec": infer_time,
        "inference_latency_ms_per_row": (infer_time / len(X_test)) * 1000,
        **metrics,
    }


def run_full_benchmark(models: dict, X_train, y_train, X_test, y_test):
    """
    Runs benchmark_model for every model in `models` (dict of name -> estimator)
    and returns a results DataFrame, one row per model.

    Calls gc.collect() between models to release memory from the previous
    model's fitted trees before starting the next one, and catches
    MemoryError so a single model running out of RAM (most likely
    GradientBoosting on machines with <16GB RAM) doesn't abort the whole
    benchmark -- that model is skipped with a warning instead.
    """
    rows = []
    for name, model in models.items():
        print(f"[benchmark] Training {name} on {len(X_train):,} rows...")
        try:
            row = benchmark_model(name, model, X_train, y_train, X_test, y_test)
        except MemoryError as e:
            print(
                f"[benchmark]   SKIPPED {name}: ran out of memory ({e}). "
                f"Try reducing n_estimators for this model in models.py, "
                f"or run on a machine with more RAM."
            )
            continue
        rows.append(row)
        print(
            f"[benchmark]   done in {row['train_time_sec']:.2f}s | "
            f"acc={row['accuracy']:.4f} | macro-F1={row['macro_f1']:.4f}"
        )
        del model
        gc.collect()
    return pd.DataFrame(rows)


def run_scalability_sweep(model_factory, X_train_full, y_train_full, X_test, y_test,
                           row_sizes=(10_000, 50_000, 100_000, 500_000)):
    """
    For each model produced by model_factory() (dict of name -> fresh
    unfitted estimator), retrains on stratified subsamples of increasing
    size and records training time / memory to trace scalability curves.

    model_factory must be a callable that returns a *fresh* dict of models
    each time it's called (so earlier fits don't leak state between sizes).
    """
    all_rows = []
    for n_rows in row_sizes:
        X_sub, y_sub = stratified_subsample(X_train_full, y_train_full, n_rows)
        models = model_factory()
        print(f"\n=== Scalability sweep: {n_rows:,} rows ===")
        df = run_full_benchmark(models, X_sub, y_sub, X_test, y_test)
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)


def get_learning_curve(model_name, model, X_train, y_train, X_val, y_val, max_stages=None):
    """
    Extracts train vs. validation loss (log-loss) at each boosting
    iteration for tree-ensemble models, used for the overfitting /
    generalization diagnostic (Research Question 3).

    Handles the differing native APIs of each library:
      - XGBoost / LightGBM: native eval_set + eval_history
      - GradientBoostingClassifier: staged_predict_proba
      - RandomForestClassifier: staged predictions aren't natively
        supported (bagging, not sequential), so we approximate by
        varying n_estimators and refitting (bagging does not overfit
        with more trees the way boosting does, which is itself part
        of the finding for RQ3).
    """
    from sklearn.metrics import log_loss
    from sklearn.base import clone

    if model_name == "XGBoost":
        m = clone(model)
        m.set_params(eval_metric="mlogloss")
        m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        evals = m.evals_result()
        return evals["validation_0"]["mlogloss"], evals["validation_1"]["mlogloss"]

    if model_name == "LightGBM":
        m = clone(model)
        m.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="multi_logloss",
        )
        evals = m.evals_result_
        train_key, val_key = list(evals.keys())[0], list(evals.keys())[1]
        return evals[train_key]["multi_logloss"], evals[val_key]["multi_logloss"]

    if model_name == "GradientBoosting":
        m = clone(model)
        m.fit(X_train, y_train)
        train_losses, val_losses = [], []
        for train_proba, val_proba in zip(
            m.staged_predict_proba(X_train), m.staged_predict_proba(X_val)
        ):
            train_losses.append(log_loss(y_train, train_proba, labels=m.classes_))
            val_losses.append(log_loss(y_val, val_proba, labels=m.classes_))
        return train_losses, val_losses

    if model_name == "RandomForest":
        from sklearn.base import clone as _clone
        stages = max_stages or [10, 25, 50, 75, 100, 150, 200]
        train_losses, val_losses = [], []
        for n in stages:
            m = _clone(model)
            m.set_params(n_estimators=n)
            m.fit(X_train, y_train)
            train_losses.append(log_loss(y_train, m.predict_proba(X_train), labels=m.classes_))
            val_losses.append(log_loss(y_val, m.predict_proba(X_val), labels=m.classes_))
        return train_losses, val_losses

    raise ValueError(f"Unknown model_name: {model_name}")


def get_feature_importances(model_name, fitted_model, feature_names):
    """
    Returns a Series of feature -> importance, normalized to sum to 1,
    sorted descending. Handles the slightly different attribute names
    across libraries.
    """
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
    else:
        raise ValueError(f"{model_name} has no feature_importances_ attribute")

    importances = np.asarray(importances, dtype=float)
    if importances.sum() > 0:
        importances = importances / importances.sum()

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)
