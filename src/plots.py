"""
plots.py
--------
Generates the four visual deliverables specified in the proposal:

  1. Speed vs. Performance Frontier   (scatter: train_time vs macro-F1)
  2. Scalability Curves               (line: train_time vs training rows)
  3. Loss Convergence Profiles        (learning curves: train vs val loss)
  4. Feature Importance Divergence    (horizontal bar charts, per model)

All figures are saved to results/figures/ as PNG (300 DPI) for direct
inclusion in a report or README.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from models import MODEL_COLORS

sns.set_theme(style="whitegrid", context="talk")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plots] saved {path}")
    plt.close(fig)


def plot_speed_vs_performance(results_df, filename="speed_vs_performance_frontier.png"):
    """
    Scatter plot: training time (X) vs. macro-F1 (Y). One point per model
    (use the full-dataset / final-tuned results). Ideal models sit in the
    top-left (fast + accurate).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in results_df.iterrows():
        color = MODEL_COLORS.get(row["model"], "gray")
        ax.scatter(row["train_time_sec"], row["macro_f1"], s=220, color=color,
                   edgecolor="black", linewidth=1.2, zorder=3, label=row["model"])
        ax.annotate(row["model"], (row["train_time_sec"], row["macro_f1"]),
                    textcoords="offset points", xytext=(10, 8), fontsize=11)

    ax.set_xlabel("Training Time (seconds, log scale)")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Speed vs. Performance Frontier")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, filename)


def plot_scalability_curves(sweep_df, metric="train_time_sec",
                             filename="scalability_curves.png"):
    """
    Line chart: metric (default training time) vs. training-set size,
    one line per model. Demonstrates non-linear scaling of standard
    algorithms vs. near-linear scaling of XGBoost/LightGBM.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for model_name in sweep_df["model"].unique():
        sub = sweep_df[sweep_df["model"] == model_name].sort_values("train_rows")
        color = MODEL_COLORS.get(model_name, "gray")
        ax.plot(sub["train_rows"], sub[metric], marker="o", linewidth=2.5,
                markersize=8, color=color, label=model_name)

    ax.set_xlabel("Training Set Size (rows)")
    ylabel = "Training Time (seconds)" if metric == "train_time_sec" else metric
    ax.set_ylabel(ylabel)
    ax.set_title("Scalability Curves: Training Time vs. Dataset Size")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Model")
    _save(fig, filename)


def plot_loss_convergence(curves: dict, filename="loss_convergence_profiles.png"):
    """
    curves: dict of {model_name: (train_losses_list, val_losses_list)}
    Produces a side-by-side grid of learning curves, one subplot per model,
    to diagnose overfitting (divergence between train and val loss).
    """
    n_models = len(curves)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (model_name, (train_loss, val_loss)) in zip(axes, curves.items()):
        iterations = range(1, len(train_loss) + 1)
        color = MODEL_COLORS.get(model_name, "gray")
        ax.plot(iterations, train_loss, label="Train Loss", color=color, linewidth=2.2)
        ax.plot(iterations, val_loss, label="Validation Loss", color=color,
                linewidth=2.2, linestyle="--", alpha=0.75)
        ax.set_title(model_name)
        ax.set_xlabel("Boosting Iteration / Tree Count")
        ax.set_ylabel("Log Loss")
        ax.legend()

    for ax in axes[len(curves):]:
        ax.axis("off")

    fig.suptitle("Loss Convergence Profiles: Train vs. Validation", y=1.02, fontsize=18)
    fig.tight_layout()
    _save(fig, filename)


def plot_feature_importance_divergence(importances: dict, top_n=15,
                                        filename="feature_importance_divergence.png"):
    """
    importances: dict of {model_name: pd.Series(feature -> importance)}
    Produces a grid of horizontal bar charts, one per model, each showing
    its top_n most important features.
    """
    n_models = len(importances)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (model_name, series) in zip(axes, importances.items()):
        top = series.head(top_n).sort_values()
        color = MODEL_COLORS.get(model_name, "gray")
        ax.barh(top.index, top.values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(model_name)
        ax.set_xlabel("Normalized Importance")

    for ax in axes[len(importances):]:
        ax.axis("off")

    fig.suptitle("Feature Importance Divergence Across Ensemble Paradigms", y=1.02, fontsize=18)
    fig.tight_layout()
    _save(fig, filename)
