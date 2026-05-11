"""Baseline regression model for L1 -> L3 prediction.

Default model is HistGradientBoostingRegressor (handles NaN natively, fast).
RandomForestRegressor available as an alternative via kind="rf".
xgboost/lightgbm are intentionally NOT required; add them later if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .feature_engineering import time_split


def make_model(numeric_features: list[str],
               categorical_features: list[str],
               kind: str = "hgb",
               seed: int = 42) -> Pipeline:
    """ColumnTransformer (passthrough numeric + one-hot cat) + regressor."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ],
        remainder="drop",
    )
    if kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=seed,
        )
    else:
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed,
        )
    return Pipeline([("pre", pre), ("reg", reg)])


def train_and_score(df: pd.DataFrame,
                    target_col: str,
                    numeric_features: list[str],
                    categorical_features: list[str],
                    split: str = "random",
                    kind: str = "hgb",
                    seed: int = 42) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame]:
    """Train baseline model and compute hold-out metrics.

    Returns (model, metrics, train_df, test_df).
    """
    features = numeric_features + categorical_features

    if split == "time":
        train, test = time_split(df)
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=seed)

    model = make_model(numeric_features, categorical_features, kind=kind, seed=seed)
    model.fit(train[features], train[target_col])
    pred = model.predict(test[features])

    metrics = {
        "model": kind,
        "split": split,
        "n_train": len(train),
        "n_test": len(test),
        "MAE": float(mean_absolute_error(test[target_col], pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test[target_col], pred))),
        "R2": float(r2_score(test[target_col], pred)),
        "target": target_col,
    }
    return model, metrics, train, test


def feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series,
                       method: str = "permutation",
                       seed: int = 42,
                       n_repeats: int = 3) -> pd.DataFrame:
    """Return DataFrame [feature, importance]. SHAP used only if explicitly available."""
    if method == "shap":
        try:
            import shap  # type: ignore
            transformed = model.named_steps["pre"].transform(X)
            explainer = shap.Explainer(model.named_steps["reg"])
            shap_vals = explainer(transformed)
            enc_names = model.named_steps["pre"].get_feature_names_out()
            imp = np.abs(shap_vals.values).mean(axis=0)
            return (pd.DataFrame({"feature": enc_names, "importance": imp})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
        except ImportError:
            pass  # fall through to permutation

    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
