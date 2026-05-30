"""
RFM-сегментация клиентов для ChurnGuard.

Методология:
1. Recency (R): сколько дней с последней активности (чем меньше, тем лучше)
2. Frequency (F): количество транзакций/заказов/взаимодействий
3. Monetary (M): суммарная выручка с клиента

Каждый параметр разбивается на квартили (1–4), затем RFM-ячейки
агрегируются в 5 бизнес-сегментов:
- Champions: R4 F4 M3-4 — лучшие клиенты
- Loyal: R3-4 F2-3 M2-3 — лояльные
- At Risk: R1-2 F3-4 M3-4 — были хорошими, перестали заходить
- Hibernating: R1-2 F1-2 M1-3 — низкая активность
- Lost: R1 F1 M1-2 — потерянные
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ruff: noqa: E501

# Имена колонок для поиска в данных
RECENCY_CANDIDATES = [
    "days_since_last_login", "days_since_last_order",
    "days_since_last_activity", "recency_days",
]

FREQUENCY_CANDIDATES = [
    "total_orders", "transaction_count",
    "calls_to_support", "feature_adoption_count",
    "total_orders", "product_categories_used",
]

MONETARY_CANDIDATES = [
    "total_revenue", "monthly_fee", "monthly_charges",
    "total_charges", "avg_order_value",
]

SEGMENT_LABELS = {
    "Champions": "🏆 Champions",
    "Loyal": "💙 Loyal",
    "At Risk": "⚠️ At Risk",
    "Hibernating": "💤 Hibernating",
    "Lost": "⚫ Lost",
}

SEGMENT_COLORS = {
    "Champions": "#4CAF50",
    "Loyal": "#2196F3",
    "At Risk": "#FF9800",
    "Hibernating": "#9C27B0",
    "Lost": "#757575",
}


def _auto_detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Автоматически находит колонки для R, F, M по словарю синонимов.

    Returns:
        (recency_col, frequency_col, monetary_col)
    """
    cols_lower = {c.lower().strip().replace(" ", "_") for c in df.columns}

    recency_col = None
    for cand in RECENCY_CANDIDATES:
        if cand in cols_lower:
            recency_col = cand
            break
    if recency_col is None:
        # Fallback: ищем колонку с 'days_since' в названии
        for c in df.columns:
            if "days_since" in c.lower():
                recency_col = c.lower().strip().replace(" ", "_")
                break

    frequency_col = None
    for cand in FREQUENCY_CANDIDATES:
        if cand in cols_lower:
            frequency_col = cand
            break
    if frequency_col is None:
        for c in df.columns:
            cl = c.lower()
            if "count" in cl or "orders" in cl or "total" in cl:
                if recency_col and c.lower() != recency_col:
                    frequency_col = c.lower().strip().replace(" ", "_")
                    break

    monetary_col = None
    for cand in MONETARY_CANDIDATES:
        if cand in cols_lower:
            monetary_col = cand
            break
    if monetary_col is None:
        for c in df.columns:
            cl = c.lower()
            if ("revenue" in cl or "fee" in cl or "charges" in cl or "spent" in cl):
                monetary_col = c.lower().strip().replace(" ", "_")
                break

    return recency_col, frequency_col, monetary_col


def compute_rfm(
    df: pd.DataFrame,
    recency_col: str | None = None,
    frequency_col: str | None = None,
    monetary_col: str | None = None,
) -> pd.DataFrame:
    """
    Вычисляет R, F, M скоры (1–4) для каждого клиента.

    Args:
        df: DataFrame с клиентскими данными
        recency_col: название колонки recency (авто-определение если None)
        frequency_col: название колонки frequency
        monetary_col: название колонки monetary

    Returns:
        DataFrame с добавленными колонками:
        - r_score, f_score, m_score (1-4)
        - rfm_score (строка "444", "123" и т.д.)
        - rfm_segment (название сегмента)
    """
    result = df.copy()

    # Авто-определение колонок
    if recency_col is None or frequency_col is None or monetary_col is None:
        auto_r, auto_f, auto_m = _auto_detect_columns(df)
        recency_col = recency_col or auto_r
        frequency_col = frequency_col or auto_f
        monetary_col = monetary_col or auto_m

    if recency_col is None:
        raise ValueError(
            "Не удалось найти колонку Recency. "
            f"Ожидаемые названия: {RECENCY_CANDIDATES}"
        )
    if frequency_col is None:
        raise ValueError(
            "Не удалось найти колонку Frequency. "
            f"Ожидаемые названия: {FREQUENCY_CANDIDATES}"
        )
    if monetary_col is None:
        raise ValueError(
            "Не удалось найти колонку Monetary. "
            f"Ожидаемые названия: {MONETARY_CANDIDATES}"
        )

    # Нормализуем имена — ищем реальную колонку
    col_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    r_col = col_map.get(recency_col, recency_col)
    f_col = col_map.get(frequency_col, frequency_col)
    m_col = col_map.get(monetary_col, monetary_col)

    # --- Recency: чем меньше дней, тем выше скор (инвертируем) ---
    r_values = result[r_col].fillna(result[r_col].median())
    try:
        result["r_score"] = pd.qcut(r_values, q=4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    except ValueError:
        # Если не получается разбить на 4 квантиля (мало уникальных значений)
        result["r_score"] = pd.cut(r_values, bins=4, labels=[4, 3, 2, 1], include_lowest=True).astype(int)

    # --- Frequency: чем больше, тем выше скор ---
    f_values = result[f_col].fillna(0)
    try:
        result["f_score"] = pd.qcut(f_values, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    except ValueError:
        result["f_score"] = pd.cut(f_values, bins=4, labels=[1, 2, 3, 4], include_lowest=True).astype(int)

    # --- Monetary: чем больше, тем выше скор ---
    m_values = result[m_col].fillna(0)
    try:
        result["m_score"] = pd.qcut(m_values, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    except ValueError:
        result["m_score"] = pd.cut(m_values, bins=4, labels=[1, 2, 3, 4], include_lowest=True).astype(int)

    # --- RFM-строка ---
    result["rfm_score"] = (
        result["r_score"].astype(str)
        + result["f_score"].astype(str)
        + result["m_score"].astype(str)
    )

    # --- Сегментация по правилам ---
    result["rfm_segment"] = result.apply(_classify_segment, axis=1)

    return result


def _classify_segment(row: pd.Series) -> str:
    """
    Классифицирует клиента в один из 5 сегментов на основе R, F, M скоров.

    Правила:
    - Champions: R=4, F=4, M≥3
    - Loyal: R≥3, F≥2, M≥2 (но не Champions)
    - At Risk: R≤2, F≥3, M≥3 (были хорошими, перестали заходить)
    - Hibernating: R≤2, F≤2, M≤3 (но не Lost)
    - Lost: R=1, F=1, M≤2
    """
    r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])

    if r == 4 and f == 4 and m >= 3:
        return "Champions"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    elif r == 1 and f == 1 and m <= 2:
        return "Lost"
    elif r <= 2 and f <= 2:
        return "Hibernating"
    elif r >= 3 and f >= 2 and m >= 2:
        return "Loyal"
    elif r <= 2:
        return "Hibernating"
    else:
        return "Loyal"


def get_segment_stats(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает агрегированную статистику по сегментам.

    Returns:
        DataFrame с колонками: segment, client_count, client_pct,
        avg_recency, avg_frequency, avg_monetary, total_value
    """
    stats = rfm_df.groupby("rfm_segment").agg(
        client_count=("client_id", "count"),
        avg_r_score=("r_score", "mean"),
        avg_f_score=("f_score", "mean"),
        avg_m_score=("m_score", "mean"),
    ).reset_index()

    total_clients = stats["client_count"].sum()
    stats["client_pct"] = (stats["client_count"] / total_clients * 100).round(1)

    # Сортируем по размеру сегмента
    segment_order = ["Champions", "Loyal", "At Risk", "Hibernating", "Lost"]
    stats["_order"] = stats["rfm_segment"].apply(
        lambda s: segment_order.index(s) if s in segment_order else 99
    )
    stats = stats.sort_values("_order").drop(columns=["_order"])

    return stats


def get_rfm_pivot(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит сводную таблицу R×F×M для heatmap.

    Returns:
        DataFrame с колонками: r_score, f_score, m_score, count, segment
    """
    pivot = (
        rfm_df.groupby(["r_score", "f_score", "m_score"])
        .agg(
            count=("client_id", "count"),
            avg_churn=("churned", "mean") if "churned" in rfm_df.columns else pd.Series(dtype=float),
        )
        .reset_index()
    )

    # Добавляем сегмент для каждой ячейки
    def _seg_for_row(r):
        return _classify_segment(pd.Series({
            "r_score": r["r_score"],
            "f_score": r["f_score"],
            "m_score": r["m_score"],
        }))

    pivot["segment"] = pivot.apply(_seg_for_row, axis=1)
    return pivot
