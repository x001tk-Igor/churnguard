"""
Генерация признаков (feature engineering) для модели оттока.

Генерирует:
- Lag-признаки (если данные позволяют)
- Ratio-признаки (соотношения между числовыми колонками)
- Trend-признаки (если есть временной ряд)
- Кодирование категориальных переменных (OneHot / Ordinal)
- Нормализацию числовых признаков
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# ruff: noqa: E501


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "churned",
    encode_categorical: bool = True,
    scale_numeric: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Генерирует признаки для ML-модели.

    Шаги:
    1. Выделяет ID-колонки (не используются как признаки)
    2. Генерирует ratio-признаки
    3. Кодирует категориальные переменные
    4. Нормализует числовые признаки
    5. Возвращает матрицу признаков X и целевую переменную y

    Args:
        df: входной DataFrame
        target_col: название целевой переменной
        encode_categorical: кодировать ли категориальные колонки
        scale_numeric: нормализовать ли числовые признаки

    Returns:
        (X, feature_names, metadata)
        - X: матрица признаков (DataFrame)
        - feature_names: список названий всех признаков
        - metadata: словарь с информацией о трансформациях
    """
    df = df.copy()
    metadata: dict = {
        "original_columns": list(df.columns),
        "id_columns": [],
        "categorical_columns": [],
        "numeric_columns": [],
        "dropped_columns": [],
        "generated_features": [],
        "encoders": {},
    }

    # --- Шаг 1: ID-колонки ---
    id_patterns = ["client_id", "customer_id", "user_id", "id"]
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_")
        if any(pat in col_lower for pat in id_patterns) and col != target_col:
            metadata["id_columns"].append(col)

    # --- Шаг 2: Ratio-признаки ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in metadata["id_columns"] + [target_col]]

    # Генерируем отношения между связанными величинами
    ratio_pairs = _find_ratio_pairs(df, numeric_cols)
    for num, denom, name in ratio_pairs:
        if denom in df.columns and num in df.columns:
            denom_vals = df[denom].replace(0, np.nan)
            df[name] = df[num] / denom_vals
            df[name] = df[name].fillna(0).clip(lower=0, upper=df[name].quantile(0.99))
            metadata["generated_features"].append(name)

    # --- Шаг 3: Категориальные переменные ---
    categorical_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    categorical_cols = [
        c for c in categorical_cols
        if c not in metadata["id_columns"] + [target_col]
    ]

    # Отделяем high-cardinality (слишком много уникальных значений)
    low_card_cols = []
    high_card_cols = []
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique <= 15:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)
            metadata["dropped_columns"].append(f"{col} (high cardinality: {n_unique})")

    metadata["categorical_columns"] = low_card_cols

    if encode_categorical and low_card_cols:
        df = _encode_categorical(df, low_card_cols, metadata)

    # --- Шаг 4: Числовые признаки ---
    numeric_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in metadata["id_columns"] + [target_col]
    ]
    metadata["numeric_columns"] = numeric_feature_cols

    if scale_numeric and numeric_feature_cols:
        scaler = StandardScaler()
        df[numeric_feature_cols] = scaler.fit_transform(
            df[numeric_feature_cols].fillna(0)
        )
        metadata["encoders"]["scaler"] = scaler

    # --- Шаг 5: Формируем X и y ---
    feature_cols = [
        c for c in df.columns
        if c not in metadata["id_columns"] + [target_col]
        and c not in [d.split(" (")[0] for d in metadata["dropped_columns"]]
    ]

    # Убираем не-numeric колонки, которые не закодировали
    feature_cols = [
        c for c in feature_cols
        if np.issubdtype(df[c].dtype, np.number)
        or df[c].dtype == bool
    ]

    X = df[feature_cols].fillna(0)
    feature_names = list(X.columns)

    return X, feature_names, metadata


def _find_ratio_pairs(df: pd.DataFrame, numeric_cols: list[str]) -> list[tuple[str, str, str]]:
    """
    Находит пары колонок, из которых можно построить ratio-признаки.

    Ищет:
    - X / tenure_months (нормализация на срок)
    - tickets / calls (нагрузка)
    - returns / orders (уровень возвратов)
    """
    pairs = []

    # support_tickets / tenure_months → tickets_per_month
    ticket_cols = [c for c in numeric_cols if "ticket" in c.lower() or "call" in c.lower()]
    tenure_cols = [c for c in numeric_cols if "tenure" in c.lower()]
    for t in ticket_cols:
        for ten in tenure_cols:
            pairs.append((t, ten, f"ratio_{t}_per_{ten}"))

    # total_revenue / total_orders → avg_order_value (если ещё нет)
    rev_cols = [c for c in numeric_cols if "revenue" in c.lower() or "charges" in c.lower()]
    order_cols = [c for c in numeric_cols if "order" in c.lower() or "transaction" in c.lower()]
    for r in rev_cols:
        for o in order_cols:
            ratio_name = f"ratio_{r}_per_{o}"
            if ratio_name.replace("ratio_", "") not in [cc.lower() for cc in df.columns]:
                pairs.append((r, o, ratio_name))

    # return_rate (если не в долях, а в count / orders)
    return_cols = [c for c in numeric_cols if "return" in c.lower() and "rate" not in c.lower()]
    for ret in return_cols:
        for o in order_cols:
            pairs.append((ret, o, f"ratio_{ret}_to_{o}"))

    return pairs


def _encode_categorical(
    df: pd.DataFrame,
    cat_cols: list[str],
    metadata: dict,
) -> pd.DataFrame:
    """
    Кодирует категориальные переменные.

    - Бинарные (True/False, 2 уникальных значения) → Ordinal 0/1
    - Небинарные (3-15 категорий) → OneHot
    """
    binary_cols = []
    multi_cols = []

    for col in cat_cols:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    # Бинарные → OrdinalEncoder
    if binary_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[binary_cols] = oe.fit_transform(df[binary_cols].astype(str).fillna("MISSING"))
        metadata["encoders"]["ordinal"] = oe
        metadata["generated_features"].extend(binary_cols)

    # Мультикатегорные → OneHotEncoder
    if multi_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        encoded = ohe.fit_transform(df[multi_cols].astype(str).fillna("MISSING"))
        encoded_cols = [
            f"{col}_{cat}"
            for col, cats in zip(multi_cols, ohe.categories_)
            for cat in cats[1:]  # drop="first"
        ]
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=multi_cols), encoded_df], axis=1)
        metadata["encoders"]["onehot"] = ohe
        metadata["generated_features"].extend(encoded_cols)

    return df
