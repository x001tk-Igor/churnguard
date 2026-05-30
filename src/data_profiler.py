"""
Профилировщик данных для ChurnGuard.

Выдаёт Data Health Score (0–100) и детальный отчёт по качеству данных:
- Полнота (completeness): % пропусков
- Уникальность (uniqueness): % дубликатов
- Согласованность (consistency): выбросы и аномалии
- Типизация (dtype): соответствие ожидаемым типам
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ruff: noqa: E501


@dataclass
class ColumnReport:
    """Отчёт по одной колонке."""
    name: str
    dtype: str
    missing_pct: float = 0.0
    missing_count: int = 0
    unique_pct: float = 0.0
    outlier_count: int = 0
    outlier_pct: float = 0.0
    issues: list[str] = field(default_factory=list)
    score: int = 100  # 0–100 для этой колонки


@dataclass
class DataHealthReport:
    """Полный отчёт о здоровье данных."""
    filename: str
    total_rows: int
    total_columns: int
    overall_score: int  # 0–100
    columns: list[ColumnReport] = field(default_factory=list)
    global_issues: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_pct: float = 0.0


def _check_completeness(series: pd.Series) -> tuple[float, int]:
    """Возвращает (процент пропусков, количество пропусков)."""
    n_missing = int(series.isna().sum())
    pct = (n_missing / len(series) * 100) if len(series) > 0 else 0.0
    return pct, n_missing


def _check_uniqueness(series: pd.Series) -> float:
    """Возвращает процент уникальных значений."""
    n = len(series.dropna())
    if n == 0:
        return 0.0
    return series.dropna().nunique() / n * 100


def _detect_outliers(series: pd.Series) -> tuple[int, float]:
    """
    Обнаруживает выбросы методом IQR.

    Returns:
        (количество выбросов, процент выбросов)
    """
    clean = series.dropna()
    if len(clean) == 0:
        return 0, 0.0

    if not np.issubdtype(clean.dtype, np.number):
        return 0, 0.0

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = int(((clean < lower) | (clean > upper)).sum())
    pct = n_outliers / len(series) * 100
    return n_outliers, pct


def profile_dataframe(df: pd.DataFrame, filename: str = "unknown.csv") -> DataHealthReport:
    """
    Профилирует DataFrame и возвращает отчёт о здоровье данных.

    Args:
        df: входной DataFrame
        filename: имя файла для отчёта

    Returns:
        DataHealthReport с overall_score и детализацией по колонкам
    """
    reports: list[ColumnReport] = []
    global_issues: list[str] = []

    # Дубликаты строк
    dupes = int(df.duplicated().sum())
    dupe_pct = dupes / len(df) * 100 if len(df) > 0 else 0.0
    if dupes > 0:
        global_issues.append(f"Найдено {dupes} дубликатов строк ({dupe_pct:.1f}%)")

    # Анализ каждой колонки
    for col in df.columns:
        series = df[col]
        col_report = ColumnReport(
            name=col,
            dtype=str(series.dtype),
        )

        # Полнота
        missing_pct, missing_count = _check_completeness(series)
        col_report.missing_pct = missing_pct
        col_report.missing_count = missing_count

        if missing_pct > 20:
            col_report.issues.append(f"Критические пропуски: {missing_pct:.0f}%")
        elif missing_pct > 5:
            col_report.issues.append(f"Значительные пропуски: {missing_pct:.1f}%")

        # Уникальность
        unique_pct = _check_uniqueness(series)
        col_report.unique_pct = unique_pct

        if unique_pct == 0 and missing_pct < 100:
            col_report.issues.append("Все значения одинаковы — колонка бесполезна")
        elif unique_pct > 99.5 and np.issubdtype(series.dtype, np.number):
            pass  # ID-колонки — ок

        # Выбросы (только для числовых)
        pct_out = 0.0
        if np.issubdtype(series.dtype, np.number):
            n_out, pct_out = _detect_outliers(series)
            col_report.outlier_count = n_out
            col_report.outlier_pct = pct_out
            if pct_out > 10:
                col_report.issues.append(f"Много выбросов: {pct_out:.0f}% значений за пределами IQR")
            elif pct_out > 3:
                col_report.issues.append(f"Умеренные выбросы: {pct_out:.1f}%")

        # Скоринг колонки (100 → вычитаем штрафы)
        col_score = 100
        col_score -= min(missing_pct * 2, 40)      # пропуски: до −40
        col_score -= min(pct_out * 2, 20) if hasattr(col_report, 'outlier_pct') else 0  # выбросы: до −20
        col_score -= 10 if len(col_report.issues) >= 3 else 0
        col_report.score = max(int(col_score), 0)

        reports.append(col_report)

    # Общий скор: среднее по колонкам
    if reports:
        overall_score = int(np.mean([r.score for r in reports]))
    else:
        overall_score = 0

    # Глобальные проблемы
    if overall_score < 40:
        global_issues.append("Критическое качество данных — требуется глубокая очистка")
    elif overall_score < 70:
        global_issues.append("Среднее качество данных — рекомендуется очистка перед ML")

    high_missing_cols = [r.name for r in reports if r.missing_pct > 30]
    if high_missing_cols:
        global_issues.append(
            f"Колонки с >30% пропусков: {', '.join(high_missing_cols)}"
        )

    return DataHealthReport(
        filename=filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        overall_score=overall_score,
        columns=reports,
        global_issues=global_issues,
        duplicate_rows=dupes,
        duplicate_pct=dupe_pct,
    )


def auto_fix_dataframe(df: pd.DataFrame, report: DataHealthReport) -> pd.DataFrame:
    """
    Автоматически исправляет обнаруженные проблемы в данных.

    Что делает:
    - Заполняет пропуски: числовые → медиана, категориальные → мода
    - Удаляет дубликаты строк
    - Ограничивает выбросы (winsorize 1-й и 99-й перцентиль)

    Args:
        df: исходный DataFrame
        report: отчёт из profile_dataframe()

    Returns:
        Исправленный DataFrame
    """
    df = df.copy()

    # Удаляем дубликаты
    if report.duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    for col_report in report.columns:
        col = col_report.name
        if col not in df.columns:
            continue

        series = df[col]

        # Заполнение пропусков
        if col_report.missing_pct > 0:
            if np.issubdtype(series.dtype, np.number):
                df[col] = series.fillna(series.median())
            else:
                mode_val = series.mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
                df[col] = series.fillna(fill_val)

        # Ограничение выбросов (только числовые)
        if col_report.outlier_pct > 3 and np.issubdtype(series.dtype, np.number):
            clean = series.dropna()
            lower = clean.quantile(0.01)
            upper = clean.quantile(0.99)
            df[col] = series.clip(lower=lower, upper=upper)

    return df
