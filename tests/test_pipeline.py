"""
Basic smoke tests for the data pipeline and benchmarking utilities.
Run with: pytest tests/
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from data_pipeline import get_feature_target_split, make_splits, stratified_subsample
from models import get_baseline_models
from benchmark import benchmark_model, evaluate_predictions


def _make_fake_covertype(n_samples=600, n_classes=7):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=10,
        n_classes=n_classes, n_clusters_per_class=1, random_state=0,
    )
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["Cover_Type"] = y + 1  # mimic raw 1-indexed labels
    return df


def test_target_is_zero_indexed():
    df = _make_fake_covertype()
    X, y = get_feature_target_split(df)
    assert y.min() == 0
    assert y.max() == df["Cover_Type"].max() - 1


def test_make_splits_shapes_and_ratios():
    df = _make_fake_covertype(n_samples=1000)
    splits = make_splits(df, train_size=0.7, val_size=0.15, test_size=0.15)
    total = len(df)
    assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == total
    assert abs(len(splits["X_train"]) / total - 0.7) < 0.02


def test_stratified_subsample_respects_size():
    df = _make_fake_covertype(n_samples=1000)
    X, y = get_feature_target_split(df)
    X_sub, y_sub = stratified_subsample(X, y, n_rows=200)
    assert len(X_sub) == 200
    assert len(y_sub) == 200


def test_stratified_subsample_caps_at_full_size():
    df = _make_fake_covertype(n_samples=200)
    X, y = get_feature_target_split(df)
    X_sub, y_sub = stratified_subsample(X, y, n_rows=10_000)
    assert len(X_sub) == 200


def test_benchmark_model_returns_expected_keys():
    df = _make_fake_covertype(n_samples=400)
    splits = make_splits(df)
    models = get_baseline_models()
    model = models["RandomForest"]
    model.set_params(n_estimators=10)

    row = benchmark_model(
        "RandomForest", model,
        splits["X_train"], splits["y_train"],
        splits["X_test"], splits["y_test"],
    )
    expected_keys = {
        "model", "train_rows", "train_time_sec", "peak_train_memory_mb",
        "inference_time_sec", "inference_latency_ms_per_row",
        "accuracy", "macro_f1", "weighted_f1",
    }
    assert expected_keys.issubset(row.keys())
    assert 0.0 <= row["accuracy"] <= 1.0
    assert 0.0 <= row["macro_f1"] <= 1.0


def test_evaluate_predictions_perfect_score():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
