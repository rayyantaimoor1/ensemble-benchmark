"""
app.py — Streamlit dashboard for the Ensemble Learning Benchmark project.

Reads the CSV/JSON results produced by `src/main.py` (in results/metrics/)
and renders them as interactive charts. If results haven't been generated
yet, gives clear instructions instead of crashing.

Run with:
    streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Ensemble Learning Benchmark",
    page_icon="🌲",
    layout="wide",
)

BASE_DIR = os.path.dirname(__file__)
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

MODEL_COLORS = {
    "RandomForest": "#2E86AB",
    "GradientBoosting": "#A23B72",
    "XGBoost": "#F18F01",
    "LightGBM": "#3B8C6E",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(filename):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_json(filename):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_importances():
    importances = {}
    if not os.path.isdir(METRICS_DIR):
        return importances
    for fname in os.listdir(METRICS_DIR):
        if fname.startswith("feature_importance_") and fname.endswith(".csv"):
            model_name = fname[len("feature_importance_"):-len(".csv")]
            df = pd.read_csv(os.path.join(METRICS_DIR, fname), index_col=0)
            importances[model_name] = df.iloc[:, 0]
    return importances


results_df = load_csv("full_benchmark_results.csv")
sweep_df = load_csv("scalability_sweep_results.csv")
best_params = load_json("best_params.json")
importances = load_feature_importances()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌲 Ensemble Learning Benchmark Dashboard")
st.markdown(
    "Comparative study of **Random Forest**, **Gradient Boosting**, **XGBoost**, "
    "and **LightGBM** on the UCI Forest Covertype dataset."
)

if results_df is None:
    st.warning(
        "No results found yet. Run the pipeline first, from the project root:\n\n"
        "```bash\npython src/main.py\n```\n\n"
        "Then restart this dashboard (`streamlit run app.py`). "
        "It reads from `results/metrics/`, which is populated by `main.py`."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")
all_models = results_df["model"].tolist()
selected_models = st.sidebar.multiselect(
    "Models to display", options=all_models, default=all_models
)

metric_choice = st.sidebar.radio(
    "Primary metric", options=["accuracy", "macro_f1", "weighted_f1"], index=1
)

filtered_results = results_df[results_df["model"].isin(selected_models)]


# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------

st.subheader("Summary")
cols = st.columns(len(filtered_results) if len(filtered_results) else 1)
for col, (_, row) in zip(cols, filtered_results.iterrows()):
    with col:
        st.metric(
            label=row["model"],
            value=f"{row[metric_choice]:.3f}",
            delta=f"{row['train_time_sec']:.1f}s train",
            delta_color="off",
        )

st.dataframe(
    filtered_results[
        ["model", "train_rows", "train_time_sec", "peak_train_memory_mb",
         "inference_latency_ms_per_row", "accuracy", "macro_f1", "weighted_f1"]
    ].round(4),
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Tabs for the 4 deliverables + extras
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Speed vs. Performance",
    "Scalability",
    "Loss Convergence",
    "Feature Importance",
    "Hyperparameters",
])

with tab1:
    st.markdown("#### Speed vs. Performance Frontier")
    st.caption("Fast + accurate models sit toward the top-left.")
    fig = px.scatter(
        filtered_results, x="train_time_sec", y=metric_choice, text="model",
        color="model", color_discrete_map=MODEL_COLORS,
        log_x=True, size_max=20,
    )
    fig.update_traces(marker=dict(size=18, line=dict(width=1, color="black")),
                       textposition="top center")
    fig.update_layout(showlegend=False, xaxis_title="Training Time (s, log scale)",
                       yaxis_title=metric_choice.replace("_", " ").title())
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Scalability Curves")
    if sweep_df is None:
        st.info("Scalability sweep results not found — run `main.py` to generate `scalability_sweep_results.csv`.")
    else:
        sweep_filtered = sweep_df[sweep_df["model"].isin(selected_models)]
        sweep_metric = st.selectbox(
            "Metric to plot", options=["train_time_sec", "peak_train_memory_mb", metric_choice],
            index=0,
        )
        fig = px.line(
            sweep_filtered.sort_values("train_rows"),
            x="train_rows", y=sweep_metric, color="model",
            color_discrete_map=MODEL_COLORS, markers=True,
        )
        fig.update_layout(xaxis_title="Training Rows", yaxis_title=sweep_metric.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Raw sweep data"):
            st.dataframe(sweep_filtered, use_container_width=True)

with tab3:
    st.markdown("#### Loss Convergence Profiles")
    st.info(
        "Learning curves are computed live (not cached to CSV by default). "
        "Run the notebook or `main.py`, then check `results/figures/loss_convergence_profiles.png` "
        "for the static version, shown below if present."
    )
    fig_path = os.path.join(BASE_DIR, "results", "figures", "loss_convergence_profiles.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    else:
        st.warning("Figure not found yet — run `python src/main.py` to generate it.")

with tab4:
    st.markdown("#### Feature Importance Divergence")
    if not importances:
        st.info("No feature importance CSVs found — run `main.py` to generate `feature_importance_<model>.csv` files.")
    else:
        available = [m for m in importances if m in selected_models]
        top_n = st.slider("Top N features", min_value=5, max_value=30, value=15)
        chosen = st.selectbox("Model", options=available or list(importances.keys()))
        if chosen:
            series = importances[chosen].sort_values(ascending=False).head(top_n)
            fig = px.bar(
                x=series.values, y=series.index, orientation="h",
                color_discrete_sequence=[MODEL_COLORS.get(chosen, "#666")],
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Normalized Importance", yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Compare all models side-by-side"):
            n_models = len(importances)
            grid_cols = st.columns(min(2, n_models) or 1)
            for i, (name, series) in enumerate(importances.items()):
                with grid_cols[i % len(grid_cols)]:
                    top = series.sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=top.values, y=top.index, orientation="h",
                        title=name, color_discrete_sequence=[MODEL_COLORS.get(name, "#666")],
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
                    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("#### Best Hyperparameters (from tuning)")
    if best_params is None:
        st.info(
            "No tuning results found. Run `python src/main.py --tune` to generate "
            "`best_params.json`, or use the notebook's tuning cell."
        )
    else:
        for model_name, params in best_params.get("best_params", {}).items():
            with st.expander(f"{model_name}  —  CV macro-F1: {best_params['cv_scores'].get(model_name, 'N/A'):.4f}"
                              if isinstance(best_params.get("cv_scores", {}).get(model_name), float)
                              else model_name):
                st.json(params)


st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: results/metrics/ (generated by src/main.py). "
    "Re-run the pipeline and refresh this page to update the dashboard."
)
