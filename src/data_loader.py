"""
Загрузка данных для ChurnGuard.

Поддерживает:
- Пользовательский CSV (авто-определение разделителя и кодировки)
- Встроенные демо-датасеты (SaaS / E-commerce / Телеком)
- Валидацию минимально необходимых колонок
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ruff: noqa: E501

DATA_DIR = Path(__file__).parent.parent / "data"

REQUIRED_COLUMNS = {
    "client_id",
    "activity_date",
    "churned",
}

# Колонки для RFM (нужна хотя бы одна метрика ценности)
RFM_RECOMMENDED = {
    "transaction_count",    # Frequency
    "total_orders",         # Frequency (альтернатива)
    "total_revenue",        # Monetary
    "monthly_fee",          # Monetary (альтернатива)
    "monthly_charges",      # Monetary (альтернатива)
}

DEMO_DATASETS = {
    "SaaS-метрики (5 000 клиентов)": "saas_sample.csv",
    "E-commerce (12 450 клиентов)": "ecommerce_sample.csv",
    "Телеком (8 200 клиентов)": "telecom_sample.csv",
}


def detect_csv_params(file_bytes: bytes) -> dict:
    """
    Авто-определение разделителя и кодировки CSV.

    Пробует разделители: запятая, точка с запятой, табуляция.
    Выбирает тот, который даёт одинаковое количество полей в первых 3 строках.
    """
    # Пробуем кодировки
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1251"]:
        try:
            text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = file_bytes.decode("utf-8", errors="replace")

    # Детектируем разделитель: пробуем каждый и смотрим на консистентность
    sniff = csv.Sniffer()
    try:
        dialect = sniff.sniff(text[:4096])
        delimiter = dialect.delimiter
    except csv.Error:
        # Fallback: пробуем распространённые разделители
        best, best_count = ",", 0
        for delim in [",", ";", "\t", "|"]:
            lines = text.strip().split("\n")[:3]
            counts = [len(line.split(delim)) for line in lines if line.strip()]
            if counts and len(set(counts)) == 1 and counts[0] > best_count:
                best, best_count = delim, counts[0]
        delimiter = best

    return {"text": text, "delimiter": delimiter, "encoding": encoding}


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    """
    Проверяет, что в DataFrame есть минимально необходимые колонки.

    Returns:
        (is_valid, missing_required, missing_recommended)
    """
    cols_lower = {c.lower().strip() for c in df.columns}
    required_lower = {c.lower() for c in REQUIRED_COLUMNS}

    missing_required = [c for c in REQUIRED_COLUMNS if c.lower() not in cols_lower]
    missing_recommended = [
        c for c in RFM_RECOMMENDED
        if c.lower() not in cols_lower
    ]

    # Проверяем, есть ли хотя бы одна метрика ценности для RFM
    has_value_col = any(
        c.lower() in cols_lower
        for c in ["total_revenue", "monthly_fee", "monthly_charges", "total_charges"]
    )
    has_frequency_col = any(
        c.lower() in cols_lower
        for c in ["transaction_count", "total_orders", "calls_to_support"]
    )

    is_valid = len(missing_required) == 0

    if not has_value_col and "total_revenue" not in missing_recommended:
        missing_recommended.append("total_revenue (или аналогичная)")
    if not has_frequency_col and "transaction_count" not in missing_recommended:
        missing_recommended.append("transaction_count (или аналогичная)")

    return is_valid, missing_required, missing_recommended


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит названия колонок к snake_case нижнего регистра.
    Маппит распространённые названия к ожидаемым.
    """
    # Маппинг синонимов
    SYNONYMS = {
        "customer_id": "client_id",
        "user_id": "client_id",
        "last_activity": "activity_date",
        "last_login": "activity_date",
        "last_order_date": "activity_date",
        "is_churned": "churned",
        "churn": "churned",
        "attrited": "churned",
        "revenue": "total_revenue",
        "spent": "total_revenue",
        "amount": "total_revenue",
        "orders": "total_orders",
        "transactions": "transaction_count",
        "tenure": "tenure_months",
    }

    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in SYNONYMS.items() if k in df.columns})
    return df


def load_demo_dataset(name: str) -> pd.DataFrame | None:
    """
    Загружает один из встроенных демо-датасетов.

    Args:
        name: ключ из DEMO_DATASETS или имя файла

    Returns:
        DataFrame или None, если датасет не найден
    """
    filename = DEMO_DATASETS.get(name)
    if filename is None:
        # Может быть напрямую имя файла
        filename = name

    path = DATA_DIR / filename
    if not path.exists():
        return None

    df = pd.read_csv(path)
    # Конвертируем activity_date в datetime если ещё не
    if "activity_date" in df.columns:
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")

    return df


def load_user_csv(uploaded_file) -> pd.DataFrame | None:
    """
    Загружает и нормализует пользовательский CSV.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Нормализованный DataFrame или None при ошибке
    """
    try:
        file_bytes = uploaded_file.read()
        params = detect_csv_params(file_bytes)

        df = pd.read_csv(
            StringIO(params["text"]),
            sep=params["delimiter"],
            encoding=params["encoding"],
        )

        if df.empty:
            return None

        df = normalize_columns(df)

        # Конвертируем activity_date
        if "activity_date" in df.columns:
            df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Ошибка загрузки CSV: {e}")
        return None
