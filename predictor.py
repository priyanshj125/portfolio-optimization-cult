"""
Module 2: Predictive Engine
============================
Random Forest / XGBoost regression pipeline to predict Forward_21_Return.
Includes time-series-aware validation, evaluation metrics, and feature importances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, build_master_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Container for model evaluation metrics."""

    rmse: float
    mae: float
    r2: float
    directional_accuracy: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"RMSE:              {self.rmse:.4f}\n"
            f"MAE:               {self.mae:.4f}\n"
            f"R² Score:          {self.r2:.4f}\n"
            f"Directional Acc.:  {self.directional_accuracy:.2%}\n"
            f"Samples:           {self.n_samples}"
        )


@dataclass
class PredictorResult:
    """Full output from training and evaluation."""

    model: Any
    evaluation: EvaluationResult
    feature_importances: pd.DataFrame
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------

def prepare_xy(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from the master dataset.

    Drops rows with NaN in features or target.
  """
    feature_cols = feature_cols or FEATURE_COLUMNS
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols]
    y = subset[target_col]
    return X, y


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
  Chronological split: earliest (1-test_size) for train, latest test_size for test.

  Critical for financial time series — never shuffle randomly.
  """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    if split_idx < 1 or split_idx >= n:
        raise ValueError(f"Invalid split: n={n}, test_size={test_size}")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: str = "random_forest",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """
    Build a sklearn Pipeline with scaling + regressor.

    Parameters
    ----------
    model_type : str
        'random_forest' or 'xgboost'.
    random_state : int
        Reproducibility seed.
    **kwargs
        Passed to the underlying regressor constructor.
    """
    if model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 12),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Use model_type='random_forest' "
                "or pip install xgboost."
            )
        regressor = XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions that correctly sign the forward return
    (positive vs negative).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0  # exclude flat returns
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> EvaluationResult:
    """
    Compute RMSE, MAE, R², and Directional Accuracy.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    dir_acc = directional_accuracy(y_true_arr, y_pred_arr)

    return EvaluationResult(
        rmse=rmse,
        mae=mae,
        r2=r2,
        directional_accuracy=dir_acc,
        n_samples=len(y_true_arr),
    )


def print_evaluation(result: EvaluationResult, title: str = "Model Evaluation") -> None:
    """Pretty-print evaluation metrics to stdout."""
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    print(result)
    print(f"{'=' * 40}\n")


# ---------------------------------------------------------------------------
# Feature importances (XAI)
# ---------------------------------------------------------------------------

def extract_feature_importances(
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract and rank feature importances from a fitted Pipeline.

    Works with tree-based models that expose feature_importances_.
    """
    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        raise AttributeError(
            f"{type(regressor).__name__} does not expose feature_importances_."
        )

    importances = regressor.feature_importances_
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    df["importance_pct"] = 100.0 * df["importance"] / df["importance"].sum()
    return df.reset_index(drop=True)


def print_feature_importances(importances: pd.DataFrame, top_n: int = 10) -> None:
    """Print top-N feature importances."""
    print(f"\n--- Top {top_n} Feature Importances (XAI) ---")
    for _, row in importances.head(top_n).iterrows():
        print(f"  {row['feature']:<22} {row['importance_pct']:6.2f}%")
    print()


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: Optional[list[str]] = None,
    **model_kwargs: Any,
) -> PredictorResult:
    """
    Full training workflow:
      1. Prepare X, y
      2. Chronological train/test split
      3. Fit model
      4. Evaluate on hold-out set
      5. Extract feature importances

    Returns
    -------
    PredictorResult
        Fitted model, metrics, importances, and test predictions.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    X, y = prepare_xy(df, feature_cols=feature_cols)

    X_train, X_test, y_train, y_test = time_series_train_test_split(
        X, y, test_size=test_size
    )

    model = build_model(model_type=model_type, random_state=random_state, **model_kwargs)
    logger.info(
        "Training %s on %d samples, testing on %d samples.",
        model_type,
        len(X_train),
        len(X_test),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluation = evaluate_predictions(y_test, y_pred)
    importances = extract_feature_importances(model, feature_cols)

    predictions = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
        },
        index=y_test.index,
    )

    return PredictorResult(
        model=model,
        evaluation=evaluation,
        feature_importances=importances,
        predictions=predictions,
    )


def cross_validate_time_series(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    TimeSeriesSplit cross-validation returning per-fold metrics.

    Useful for robustness checks in the technical report.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    X, y = prepare_xy(df, feature_cols=feature_cols)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model(model_type=model_type, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        fold_results.append(
            {
                "fold": fold + 1,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "directional_accuracy": metrics.directional_accuracy,
                "n_test": metrics.n_samples,
            }
        )

    return pd.DataFrame(fold_results)


def predict_forward_returns(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> pd.Series:
    """
    Generate forward-return predictions for all rows with valid features.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    valid = df[feature_cols].dropna()
    preds = model.predict(valid)
    return pd.Series(preds, index=valid.index, name="Predicted_Forward_21_Return")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    DATA_DIR = Path(__file__).parent / "archive"
    master_df = build_master_dataset(DATA_DIR)

    result = train_and_evaluate(master_df, model_type="random_forest")
    print_evaluation(result.evaluation, title="Random Forest — Hold-out Test")
    print_feature_importances(result.feature_importances)

    cv_df = cross_validate_time_series(master_df, n_splits=5)
    print("\n--- Time-Series CV Summary ---")
    print(cv_df.describe().loc[["mean", "std"]].round(4))
