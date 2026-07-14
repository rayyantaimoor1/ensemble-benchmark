"""
data_pipeline.py
-----------------
Downloads (or loads a cached copy of) the UCI Forest Covertype dataset,
applies preprocessing (scaling of continuous features, stratified
train/val/test split), and exposes helpers for producing sub-samples of
different sizes for the scalability benchmark.

Dataset: Covertype (UCI ML Repository, id=31)
  - 581,012 instances
  - 54 features: 10 continuous + 44 binary (4 wilderness area + 40 soil type)
  - Target: Cover_Type (7 classes, 1-indexed in the raw data)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

CONTINUOUS_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "covertype.csv")


def _download_via_ucimlrepo() -> pd.DataFrame:
    """Primary download path using the official ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo

    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features
    y = covertype.data.targets
    df = pd.concat([X, y], axis=1)
    target_col = y.columns[0]
    df = df.rename(columns={target_col: "Cover_Type"})
    return df


def _download_via_openml() -> pd.DataFrame:
    """Fallback download path using sklearn's built-in fetcher."""
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(as_frame=True)
    df = bunch.frame
    df = df.rename(columns={"Cover_Type": "Cover_Type"})
    return df


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Loads the raw Covertype dataset, using a local CSV cache when available.
    Falls back from ucimlrepo -> sklearn/OpenML if the primary source is
    unreachable (e.g. no internet access in a CI environment).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force_download:
        return pd.read_csv(CACHE_PATH)

    try:
        df = _download_via_ucimlrepo()
    except Exception as e:
        print(f"[data_pipeline] ucimlrepo fetch failed ({e}); falling back to sklearn/OpenML.")
        df = _download_via_openml()

    df.to_csv(CACHE_PATH, index=False)
    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Splits features/target. The target is re-mapped from the raw 1-7
    label space to 0-6, since XGBoost's sklearn API requires zero-indexed
    class labels for multi:softprob. The other three models are agnostic
    to this shift, so we apply it uniformly for consistency.
    """
    y = df["Cover_Type"].astype(int) - 1
    X = df.drop(columns=["Cover_Type"])
    return X, y


def scale_continuous_features(X_train, X_val, X_test):
    """
    Fits a RobustScaler on the training split's continuous columns only
    (binary wilderness/soil columns are left untouched) and applies it to
    all three splits. Robust scaling is preferred over standard scaling
    here because several continuous features (e.g. distance-to-hydrology)
    are right-skewed with outliers.
    """
    scaler = RobustScaler()
    present_cols = [c for c in CONTINUOUS_COLS if c in X_train.columns]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    if present_cols:
        X_train[present_cols] = scaler.fit_transform(X_train[present_cols])
        X_val[present_cols] = scaler.transform(X_val[present_cols])
        X_test[present_cols] = scaler.transform(X_test[present_cols])

    return X_train, X_val, X_test, scaler


def make_splits(df: pd.DataFrame, train_size=0.70, val_size=0.15, test_size=0.15,
                 random_state=RANDOM_STATE):
    """
    Stratified 70/15/15 train/val/test split (stratified on the target to
    preserve class-imbalance ratios across all three splits).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    X, y = get_feature_target_split(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val, stratify=y_temp, random_state=random_state
    )

    X_train, X_val, X_test, scaler = scale_continuous_features(X_train, X_val, X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
    }


def stratified_subsample(X, y, n_rows, random_state=RANDOM_STATE):
    """
    Returns a stratified subsample of size n_rows (or the full data if
    n_rows exceeds the available rows). Used for the scalability benchmark
    (10k / 50k / 100k / 500k row increments).
    """
    n_rows = min(n_rows, len(X))
    if n_rows == len(X):
        return X, y

    frac = n_rows / len(X)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(df["Cover_Type"].value_counts(normalize=True).round(3))

    splits = make_splits(df)
    print(f"Train: {len(splits['X_train']):,} | Val: {len(splits['X_val']):,} | Test: {len(splits['X_test']):,}")
