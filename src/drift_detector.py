"""
Детектор дрифта данных для ChurnGuard.

Обнаруживает изменения в распределении признаков, которые могут
привести к деградации модели:
- PSI (Population Stability Index) — основной индикатор
- KS-тест (Kolmogorov-Smirnov) — статистический тест
- Сравнение базового и текущего распределения
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ruff: noqa: E501


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Вычисляет Population Stability Index (PSI).

    PSI < 0.1 → нет дрифта
    PSI 0.1–0.25 → умеренный дрифт
    PSI > 0.25 → значительный дрифт

    Args:
        expected: эталонное распределение (обучающая выборка)
        actual: текущее распределение (новые данные)
        bins: количество бинов
        epsilon: защита от деления на ноль

    Returns:
        Значение PSI
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Общие бины для обоих распределений
    all_data = np.concatenate([expected, actual])
    bin_edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
    # Убираем дубликаты границ
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    expected_percents = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    expected_percents = np.clip(expected_percents, epsilon, 1)
    actual_percents = np.clip(actual_percents, epsilon, 1)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log(actual_percents / expected_percents)
    )
    return float(psi)


def compute_ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Двухвыборочный KS-тест.

    Args:
        expected: эталонное распределение
        actual: текущее распределение
        alpha: уровень значимости

    Returns:
        {"statistic": float, "p_value": float, "drift_detected": bool}
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    ks_stat, p_value = stats.ks_2samp(expected, actual)
    drift = p_value < alpha

    return {
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": drift,
    }


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    psi_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Сравнивает распределения признаков между обучающей (reference)
    и текущей (current) выборками.

    Args:
        reference_df: обучающая выборка
        current_df: текущая выборка
        feature_cols: список колонок для проверки (если None — все числовые)
        psi_threshold: порог PSI для алерта

    Returns:
        DataFrame с колонками:
        - feature: название признака
        - psi: значение PSI
        - ks_statistic: KS-статистика
        - ks_p_value: p-value KS-теста
        - drift_detected: булевый флаг
        - severity: "none" / "moderate" / "severe"
    """
    if feature_cols is None:
        feature_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in feature_cols
            if c in current_df.columns
        ]

    results = []
    for col in feature_cols:
        ref_vals = reference_df[col].values.astype(float)
        cur_vals = current_df[col].values.astype(float)

        psi = compute_psi(ref_vals, cur_vals)
        ks = compute_ks_test(ref_vals, cur_vals)

        drift = psi > psi_threshold or ks["drift_detected"]

        if psi >= 0.25:
            severity = "severe"
        elif psi >= psi_threshold:
            severity = "moderate"
        else:
            severity = "none"

        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "drift_detected": drift,
            "severity": severity,
        })

    drift_df = pd.DataFrame(results)
    drift_df = drift_df.sort_values("psi", ascending=False)
    return drift_df


def get_drift_alerts(drift_df: pd.DataFrame) -> list[str]:
    """
    Формирует список текстовых алертов по результатам дрифта.

    Args:
        drift_df: результат detect_drift()

    Returns:
        Список строк-алертов
    """
    alerts = []
    drifted = drift_df[drift_df["drift_detected"]]

    if drifted.empty:
        alerts.append("✅ Дрифт данных не обнаружен. Модель стабильна.")
        return alerts

    severe = drifted[drifted["severity"] == "severe"]
    moderate = drifted[drifted["severity"] == "moderate"]

    if len(severe) > 0:
        names = ", ".join(severe["feature"].tolist())
        alerts.append(
            f"🔴 Критический дрифт в признаках: **{names}**. "
            f"Рекомендуется переобучить модель."
        )

    if len(moderate) > 0:
        names = ", ".join(moderate["feature"].tolist())
        alerts.append(
            f"🟡 Умеренный дрифт в признаках: **{names}**. "
            f"Рекомендуется мониторинг и проверка качества прогнозов."
        )

    return alerts
