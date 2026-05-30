"""
Генератор синтетических демо-датасетов для ChurnGuard.

Генерирует три типа данных с реалистичными распределениями:
- SaaS (B2B подписки): 5 000 клиентов
- E-commerce (розница): 12 450 клиентов
- Telecom (оператор связи): 8 200 клиентов

Каждый датасет содержит целевую переменную 'churned' и набор признаков,
достаточный для RFM-сегментации и ML-моделирования оттока.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ruff: noqa: E501
# isort: skip_file
# fmt: off

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------

def _add_noise(series: pd.Series, pct_missing: float = 0.03, pct_outliers: float = 0.02) -> pd.Series:
    """Добавляет реалистичный «грязный» шум: пропуски и выбросы."""
    s = series.copy()
    n = len(s)
    if pct_missing > 0:
        mask = np.random.choice([True, False], size=n, p=[pct_missing, 1 - pct_missing])
        s[mask] = np.nan
    if pct_outliers > 0:
        mask = np.random.choice([True, False], size=n, p=[pct_outliers, 1 - pct_outliers])
        s[mask] = s[mask] * np.random.uniform(3, 8, size=mask.sum())
    return s


def _generate_client_ids(n: int, prefix: str = "C") -> np.ndarray:
    return np.array([f"{prefix}{i:06d}" for i in range(1, n + 1)])


# ---------------------------------------------------------------------------
# 1. SaaS-датасет (B2B подписки)
# ---------------------------------------------------------------------------

def generate_saas_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует SaaS-датасет.

    Признаки:
    - client_id, activity_date, subscription_tier, monthly_fee
    - days_since_last_login, avg_session_duration_min
    - support_tickets_last_30d, feature_adoption_count
    - login_frequency_trend, nps_score
    - contract_length_months, has_dedicated_manager
    - industry, country
    - churned (целевая)
    """
    rng = np.random.default_rng(seed)

    # --- Демография ---
    tiers = rng.choice(["Basic", "Premium", "Enterprise"], size=n, p=[0.50, 0.35, 0.15])
    industries = rng.choice(
        ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing", "Education"],
        size=n, p=[0.35, 0.20, 0.15, 0.12, 0.10, 0.08],
    )
    countries = rng.choice(
        ["US", "UK", "DE", "FR", "CA", "AU", "Other"],
        size=n, p=[0.45, 0.15, 0.12, 0.08, 0.07, 0.05, 0.08],
    )

    # --- Поведенческие признаки (коррелируют с оттоком) ---
    monthly_fee = np.where(
        tiers == "Basic", rng.integers(29, 99, size=n),
        np.where(tiers == "Premium", rng.integers(100, 299, size=n),
                 rng.integers(300, 999, size=n)),
    ).astype(float)

    days_since_last_login = rng.poisson(lam=15, size=n).astype(float)
    avg_session_min = np.clip(rng.normal(loc=25, scale=12, size=n), 1, 120)
    support_tickets = rng.poisson(lam=1.5, size=n)
    feature_adoption = rng.poisson(lam=4, size=n)
    contract_months = rng.integers(1, 37, size=n)
    has_manager = rng.choice([True, False], size=n, p=[0.30, 0.70])

    login_trend = rng.choice(
        ["increasing", "stable", "declining"],
        size=n, p=[0.30, 0.45, 0.25],
    )
    nps = np.clip(rng.integers(0, 11, size=n).astype(float), 0, 10)

    # --- Генерация оттока (целевая) ---
    # Логит: чем больше дней без логина, тикетов, и чем ниже NPS — тем выше риск
    logit = (
        -2.0
        + 0.08 * days_since_last_login
        + 0.35 * support_tickets
        - 0.25 * np.log1p(feature_adoption)
        - 0.30 * (nps / 10.0)
        - 0.40 * (contract_months > 12).astype(float)
        - 0.50 * has_manager.astype(float)
        + 0.60 * (tiers == "Basic").astype(float)
        - 0.30 * (tiers == "Enterprise").astype(float)
        + 0.50 * (login_trend == "declining").astype(float)
        - 0.20 * (login_trend == "increasing").astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < prob).astype(int)

    # Добавляем случайный шум в 5% оттока (необъяснимый)
    noise_mask = rng.random(n) < 0.05
    churned[noise_mask] = 1 - churned[noise_mask]

    # --- Дата последней активности ---
    base_date = pd.Timestamp("2026-05-30")
    activity_dates = base_date - pd.to_timedelta(days_since_last_login, unit="D")

    # --- Собираем DataFrame ---
    df = pd.DataFrame({
        "client_id": _generate_client_ids(n, "S"),
        "activity_date": activity_dates,
        "subscription_tier": tiers,
        "monthly_fee": monthly_fee,
        "days_since_last_login": days_since_last_login,
        "avg_session_duration_min": np.round(avg_session_min, 1),
        "support_tickets_last_30d": support_tickets,
        "feature_adoption_count": feature_adoption,
        "login_frequency_trend": login_trend,
        "nps_score": nps.astype(int),
        "contract_length_months": contract_months,
        "has_dedicated_manager": has_manager,
        "industry": industries,
        "country": countries,
        "churned": churned,
    })

    # --- Грязный шум ---
    df["days_since_last_login"] = _add_noise(df["days_since_last_login"], pct_missing=0.04, pct_outliers=0.01)
    df["avg_session_duration_min"] = _add_noise(df["avg_session_duration_min"], pct_missing=0.02, pct_outliers=0.02)
    df["nps_score"] = _add_noise(df["nps_score"], pct_missing=0.06, pct_outliers=0.00)
    df["monthly_fee"] = _add_noise(df["monthly_fee"], pct_missing=0.01, pct_outliers=0.01)

    return df


# ---------------------------------------------------------------------------
# 2. E-commerce датасет (розница)
# ---------------------------------------------------------------------------

def generate_ecommerce_data(n: int = 12450, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует E-commerce датасет.

    Признаки:
    - client_id, activity_date, total_orders, total_revenue
    - avg_order_value, return_rate, days_since_last_order
    - product_categories_used, device_type
    - has_app_installed, email_open_rate
    - discount_usage_pct, reviews_written
    - country, churned (целевая)
    """
    rng = np.random.default_rng(seed)

    # --- Демография ---
    devices = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    countries = rng.choice(
        ["US", "UK", "DE", "FR", "BR", "IN", "Other"],
        size=n, p=[0.40, 0.12, 0.10, 0.08, 0.08, 0.10, 0.12],
    )
    has_app = rng.choice([True, False], size=n, p=[0.40, 0.60])

    # --- Поведенческие признаки ---
    total_orders = rng.poisson(lam=8, size=n)
    avg_order_value = np.clip(rng.lognormal(mean=3.8, sigma=0.7, size=n), 5, 500)
    total_revenue = total_orders * avg_order_value
    return_rate = np.clip(rng.beta(a=1.5, b=8, size=n), 0, 0.6)
    days_since_last_order = rng.poisson(lam=25, size=n).astype(float)
    product_categories = rng.poisson(lam=3, size=n) + 1
    email_open_rate = np.clip(rng.beta(a=3, b=4, size=n), 0.01, 0.95)
    discount_pct = np.clip(rng.beta(a=2, b=5, size=n), 0, 0.8)
    reviews = rng.poisson(lam=1, size=n)

    # --- Генерация оттока ---
    logit = (
        -1.5
        + 0.06 * days_since_last_order
        + 1.5 * return_rate
        - 0.10 * np.log1p(total_orders)
        - 0.60 * has_app.astype(float)
        - 0.80 * email_open_rate
        + 0.40 * discount_pct
        + 0.30 * (devices == "mobile").astype(float)
        - 0.15 * (devices == "desktop").astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < prob).astype(int)

    base_date = pd.Timestamp("2026-05-30")
    activity_dates = base_date - pd.to_timedelta(days_since_last_order, unit="D")

    df = pd.DataFrame({
        "client_id": _generate_client_ids(n, "E"),
        "activity_date": activity_dates,
        "total_orders": total_orders,
        "total_revenue": np.round(total_revenue, 2),
        "avg_order_value": np.round(avg_order_value, 2),
        "return_rate": np.round(return_rate, 3),
        "days_since_last_order": days_since_last_order,
        "product_categories_used": product_categories,
        "device_type": devices,
        "has_app_installed": has_app,
        "email_open_rate": np.round(email_open_rate, 3),
        "discount_usage_pct": np.round(discount_pct, 3),
        "reviews_written": reviews,
        "country": countries,
        "churned": churned,
    })

    # --- Грязный шум ---
    df["days_since_last_order"] = _add_noise(df["days_since_last_order"], pct_missing=0.03, pct_outliers=0.01)
    df["total_revenue"] = _add_noise(df["total_revenue"], pct_missing=0.01, pct_outliers=0.02)
    df["email_open_rate"] = _add_noise(df["email_open_rate"], pct_missing=0.05, pct_outliers=0.00)
    df["return_rate"] = _add_noise(df["return_rate"], pct_missing=0.02, pct_outliers=0.01)

    return df


# ---------------------------------------------------------------------------
# 3. Телеком-датасет
# ---------------------------------------------------------------------------

def generate_telecom_data(n: int = 8200, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует телеком-датасет.

    Признаки:
    - client_id, activity_date, contract_type, monthly_charges
    - total_charges, tenure_months, calls_to_support
    - avg_call_duration_min, data_usage_gb
    - international_plan, has_technical_issues
    - payment_delay_count, competitor_offer_seen
    - region, churned (целевая)
    """
    rng = np.random.default_rng(seed)

    # --- Демография ---
    contract_types = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.25, 0.20],
    )
    regions = rng.choice(
        ["North", "South", "East", "West", "Central"],
        size=n, p=[0.25, 0.22, 0.20, 0.18, 0.15],
    )
    intl_plan = rng.choice([True, False], size=n, p=[0.15, 0.85])
    tech_issues = rng.choice([True, False], size=n, p=[0.25, 0.75])
    competitor_offer = rng.choice([True, False], size=n, p=[0.30, 0.70])

    # --- Поведенческие признаки ---
    tenure = rng.poisson(lam=24, size=n) + 1
    monthly_charges = np.clip(rng.normal(loc=65, scale=30, size=n), 20, 150)
    total_charges = tenure * monthly_charges * rng.normal(loc=1.0, scale=0.05, size=n)
    calls_to_support = rng.poisson(lam=2, size=n)
    avg_call_duration = np.clip(rng.normal(loc=8, scale=4, size=n), 1, 30)
    data_usage = np.clip(rng.lognormal(mean=2.5, sigma=0.8, size=n), 0.5, 100)
    payment_delays = rng.poisson(lam=0.8, size=n)

    # --- Генерация оттока ---
    logit = (
        -2.0
        - 0.05 * np.log1p(tenure)
        + 0.45 * calls_to_support
        + 0.55 * tech_issues.astype(float)
        + 0.70 * competitor_offer.astype(float)
        + 0.90 * (contract_types == "Month-to-month").astype(float)
        - 0.80 * (contract_types == "Two year").astype(float)
        + 0.30 * payment_delays
        + 0.25 * intl_plan.astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < prob).astype(int)

    base_date = pd.Timestamp("2026-05-30")
    activity_dates = base_date - pd.to_timedelta(
        rng.poisson(lam=10, size=n), unit="D"
    )

    df = pd.DataFrame({
        "client_id": _generate_client_ids(n, "T"),
        "activity_date": activity_dates,
        "contract_type": contract_types,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "tenure_months": tenure,
        "calls_to_support": calls_to_support,
        "avg_call_duration_min": np.round(avg_call_duration, 1),
        "data_usage_gb": np.round(data_usage, 2),
        "international_plan": intl_plan,
        "has_technical_issues": tech_issues,
        "payment_delay_count": payment_delays,
        "competitor_offer_seen": competitor_offer,
        "region": regions,
        "churned": churned,
    })

    # --- Грязный шум ---
    df["monthly_charges"] = _add_noise(df["monthly_charges"], pct_missing=0.02, pct_outliers=0.01)
    df["total_charges"] = _add_noise(df["total_charges"], pct_missing=0.01, pct_outliers=0.02)
    df["data_usage_gb"] = _add_noise(df["data_usage_gb"], pct_missing=0.04, pct_outliers=0.01)
    df["avg_call_duration_min"] = _add_noise(df["avg_call_duration_min"], pct_missing=0.03, pct_outliers=0.01)

    return df


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    """Генерирует все три демо-датасета и сохраняет в data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating ChurnGuard demo datasets...")

    datasets = {
        "saas_sample.csv": generate_saas_data,
        "ecommerce_sample.csv": generate_ecommerce_data,
        "telecom_sample.csv": generate_telecom_data,
    }

    for filename, generator in datasets.items():
        df = generator()
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        churn_rate = df["churned"].mean()
        print(f"  [OK] {filename}: {len(df):,} rows, {len(df.columns)} cols, "
              f"churn {churn_rate:.1%}")

    print(f"\nDone. Files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
