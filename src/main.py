"""
main.py
-------
End-to-end pipeline entry point. Runs, in order:

  1. Data loading + preprocessing (data_pipeline.py)
  2. Baseline benchmark on the full training set (benchmark.py)
  3. Scalability sweep across 10k/50k/100k/500k rows (benchmark.py)
  4. Optional hyperparameter tuning (tuning.py) -- gated by --tune flag
  5. Learning curve extraction for the overfitting diagnostic
  6. Feature importance extraction
  7. All four required plots (plots.py)
  8. Saves all metrics to results/metrics/*.csv

Usage:
    python src/main.py                 # baseline models, quick run
    python src/main.py --tune          # also run hyperparameter tuning
    python src/main.py --full-sweep    # use full 500k row for the largest sweep step
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from data_pipeline import load_raw_data, make_splits
from models import get_baseline_models
from benchmark import (
    run_full_benchmark,
    run_scalability_sweep,
    get_learning_curve,
    get_feature_importances,
)
from tuning import tune_all_models
from plots import (
    plot_speed_vs_performance,
    plot_scalability_curves,
    plot_loss_convergence,
    plot_feature_importance_divergence,
)

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def main():
    parser = argparse.ArgumentParser(description="Ensemble Learning Benchmark on Covertype")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--tune-iter", type=int, default=20, help="RandomizedSearchCV n_iter")
    parser.add_argument(
        "--sweep-sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000, 500_000],
        help="Row counts for the scalability sweep",
    )
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Data
    print("=== Loading data ===")
    df = load_raw_data()
    splits = make_splits(df)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 2. Baseline (or tuned) models
    if args.tune:
        print("\n=== Hyperparameter tuning ===")
        models, best_params = tune_all_models(X_train, y_train, n_iter=args.tune_iter)
    else:
        print("\n=== Using baseline models (skip tuning with default run) ===")
        models = get_baseline_models()

    # 3. Full benchmark on the complete training set
    print("\n=== Full benchmark (all training rows) ===")
    results_df = run_full_benchmark(models, X_train, y_train, X_test, y_test)
    results_df.to_csv(os.path.join(METRICS_DIR, "full_benchmark_results.csv"), index=False)
    print(results_df.to_string(index=False))

    # 4. Scalability sweep (uses fresh baseline models at each size)
    print("\n=== Scalability sweep ===")
    sweep_df = run_scalability_sweep(
        get_baseline_models, X_train, y_train, X_test, y_test,
        row_sizes=tuple(args.sweep_sizes),
    )
    sweep_df.to_csv(os.path.join(METRICS_DIR, "scalability_sweep_results.csv"), index=False)

    # 5. Learning curves (overfitting diagnostic) - uses baseline models, fresh clones
    print("\n=== Learning curves ===")
    fresh_models = get_baseline_models()
    curves = {}
    for name, model in fresh_models.items():
        train_loss, val_loss = get_learning_curve(name, model, X_train, y_train, X_val, y_val)
        curves[name] = (train_loss, val_loss)

    # 6. Feature importances - refit each tuned/baseline model on full train set
    print("\n=== Feature importances ===")
    importances = {}
    for name, model in models.items():
        fitted = model.fit(X_train, y_train)
        importances[name] = get_feature_importances(name, fitted, X_train.columns)
        importances[name].to_csv(os.path.join(METRICS_DIR, f"feature_importance_{name}.csv"))

    # 7. Plots
    print("\n=== Generating plots ===")
    plot_speed_vs_performance(results_df)
    plot_scalability_curves(sweep_df)
    plot_loss_convergence(curves)
    plot_feature_importance_divergence(importances)

    print("\nAll done. Results saved to results/metrics/, figures saved to results/figures/")


if __name__ == "__main__":
    main()
