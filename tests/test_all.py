"""
Быстрые unit-тесты для ChurnGuard.
Запуск: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import detect_csv_params, load_demo_dataset, validate_columns
from src.data_profiler import auto_fix_dataframe, profile_dataframe
from src.rfm_analyzer import compute_rfm, get_segment_stats
from src.feature_engineering import engineer_features
from src.churn_trainer import train_and_compare
from src.recommendations import generate_recommendations
from src.drift_detector import compute_psi, detect_drift
from src.report_generator import export_predictions_csv, export_model_pickle


@pytest.fixture
def sample_df():
    """Данные для тестов."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "client_id": [f"C{i:04d}" for i in range(n)],
        "activity_date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "days_since_last_login": np.random.poisson(15, n),
        "total_orders": np.random.poisson(8, n),
        "total_revenue": np.random.lognormal(3.8, 0.7, n),
        "support_tickets_last_30d": np.random.poisson(1.5, n),
        "subscription_tier": np.random.choice(["Basic", "Premium"], n),
        "churned": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


# --- data_loader tests ---
def test_detect_csv_params():
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6"
    params = detect_csv_params(csv_bytes)
    assert params["delimiter"] == ","


def test_validate_columns(sample_df):
    is_valid, missing, _ = validate_columns(sample_df)
    assert is_valid
    assert len(missing) == 0


def test_validate_columns_missing():
    bad_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    is_valid, missing, _ = validate_columns(bad_df)
    assert not is_valid
    assert "client_id" in missing


# --- data_profiler tests ---
def test_profile_dataframe(sample_df):
    report = profile_dataframe(sample_df, "test.csv")
    assert 0 <= report.overall_score <= 100
    assert len(report.columns) == len(sample_df.columns)


def test_auto_fix(sample_df):
    # Внесём пропуски
    df_dirty = sample_df.copy()
    df_dirty.loc[0, "total_revenue"] = np.nan
    report = profile_dataframe(df_dirty, "dirty.csv")
    df_clean = auto_fix_dataframe(df_dirty, report)
    assert df_clean["total_revenue"].isna().sum() == 0


# --- rfm_analyzer tests ---
def test_compute_rfm(sample_df):
    rfm = compute_rfm(sample_df)
    assert "rfm_segment" in rfm.columns
    assert "r_score" in rfm.columns
    assert rfm["rfm_segment"].nunique() >= 2


def test_segment_stats(sample_df):
    rfm = compute_rfm(sample_df)
    stats = get_segment_stats(rfm)
    assert len(stats) > 0
    assert "client_count" in stats.columns


# --- feature_engineering tests ---
def test_engineer_features(sample_df):
    X, names, meta = engineer_features(sample_df)
    assert len(names) > 0
    assert len(X) == len(sample_df)
    assert "churned" not in names  # target не должен быть в признаках


# --- churn_trainer tests ---
def test_train_and_compare(sample_df):
    X, names, _ = engineer_features(sample_df)
    y = sample_df["churned"]
    report = train_and_compare(X, y, cv_folds=2)
    assert len(report.models) == 3
    assert report.best_auc > 0.5
    assert report.best_model_name in ["Logistic Regression", "Random Forest", "XGBoost"]


# --- recommendations tests ---
def test_generate_recommendations(sample_df):
    rfm = compute_rfm(sample_df)
    churn_probs = np.random.random(len(rfm)) * 0.8
    recs = generate_recommendations(rfm, churn_probs)
    assert len(recs) == len(rfm)
    assert "action" in recs.columns
    assert "churn_prob" in recs.columns


# --- drift_detector tests ---
def test_compute_psi():
    np.random.seed(42)
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(0, 1, 1000)
    psi = compute_psi(a, b)
    assert psi < 0.3  # одинаковые распределения → низкий PSI

    c = np.random.normal(3, 1, 1000)
    psi2 = compute_psi(a, c)
    assert psi2 > 0.3  # разные распределения → высокий PSI


# --- report_generator tests ---
def test_export_predictions_csv(sample_df):
    rfm = compute_rfm(sample_df)
    churn_probs = np.random.random(len(rfm))
    recs = generate_recommendations(rfm, churn_probs)
    csv_str = export_predictions_csv(recs)
    assert "client_id" in csv_str
    assert len(csv_str) > 100


def test_export_model_pickle(sample_df):
    X, names, _ = engineer_features(sample_df)
    y = sample_df["churned"]
    report = train_and_compare(X, y, cv_folds=2)
    model = report.model_objects[report.best_model_name]
    pkl = export_model_pickle(model)
    assert len(pkl) > 100
