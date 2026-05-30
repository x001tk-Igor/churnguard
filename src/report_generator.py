"""
Генератор отчётов для ChurnGuard.

Поддерживает экспорт:
- PDF (executive summary + графики)
- CSV с прогнозами (client_id, churn_prob, segment, recommendations)
- pickle/ONNX модели для интеграции в backend
"""

from __future__ import annotations

import base64
import io
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ruff: noqa: E501


def export_predictions_csv(
    recommendations_df: pd.DataFrame,
) -> str:
    """
    Экспортирует таблицу с прогнозами в CSV-строку.

    Колонки: client_id, rfm_segment, churn_prob, churn_risk_level,
    ltv, action, channel, offer, urgency, roi_estimate

    Returns:
        CSV-строка для скачивания
    """
    cols = [
        "client_id", "rfm_segment", "churn_prob", "churn_risk_level",
        "ltv", "action", "channel", "offer", "urgency", "roi_estimate",
    ]
    available = [c for c in cols if c in recommendations_df.columns]
    csv_str = recommendations_df[available].to_csv(index=False)
    return csv_str


def export_model_pickle(model) -> bytes:
    """
    Сериализует модель в pickle-байты.

    Args:
        model: обученная sklearn/XGBoost модель

    Returns:
        Байты для скачивания
    """
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def generate_pdf_report(
    executive_summary: str,
    segment_stats: pd.DataFrame,
    model_results: list,  # list[ModelResult]
    recommendations_df: pd.DataFrame,
    churn_rate: float,
    total_clients: int,
) -> bytes:
    """
    Генерирует PDF-отчёт.

    Args:
        executive_summary: текст executive summary
        segment_stats: статистика по сегментам
        model_results: результаты обучения моделей
        recommendations_df: таблица рекомендаций
        churn_rate: уровень оттока
        total_clients: общее количество клиентов

    Returns:
        PDF-байты
    """
    try:
        from fpdf import FPDF
        return _generate_pdf_fpdf(
            executive_summary, segment_stats, model_results,
            recommendations_df, churn_rate, total_clients,
        )
    except ImportError:
        # Fallback: возвращаем текстовый файл как PDF (не идеально, но работает)
        text = executive_summary.replace("*", "").replace("#", "").replace("_", "")
        text += "\n\n[PDF-отчёт не может быть сгенерирован: установите fpdf2]"
        return text.encode("utf-8")


def _generate_pdf_fpdf(
    executive_summary: str,
    segment_stats: pd.DataFrame,
    model_results: list,
    recommendations_df: pd.DataFrame,
    churn_rate: float,
    total_clients: int,
) -> bytes:
    """Генерирует PDF через библиотеку fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    # Встроенный шрифт (без кириллицы — используем транслит)
    pdf.set_font("Helvetica", size=12)

    # Заголовок
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "ChurnGuard Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Метрики
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"Total Clients: {total_clients:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Churn Rate: {churn_rate:.1%}", new_x="LMARGIN", new_y="NEXT")

    # Лучшая модель
    if model_results:
        best = max(model_results, key=lambda m: m.auc_roc)
        pdf.cell(0, 7, f"Best Model: {best.name} (AUC-ROC: {best.auc_roc:.3f})",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Сегменты
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Segment Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for _, row in segment_stats.iterrows():
        seg = row.get("rfm_segment", "Unknown")
        cnt = int(row.get("client_count", 0))
        pct = float(row.get("client_pct", 0))
        pdf.cell(0, 6, f"  {seg}: {cnt:,} clients ({pct:.1f}%)",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Топ-10 клиентов под угрозой
    if "churn_prob" in recommendations_df.columns:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Top-10 At-Risk Clients", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)

        top10 = recommendations_df.head(10)
        for i, (_, row) in enumerate(top10.iterrows()):
            cid = str(row.get("client_id", "N/A"))
            prob = float(row.get("churn_prob", 0))
            seg = str(row.get("rfm_segment", "N/A"))
            pdf.cell(0, 5, f"  {i+1}. {cid} | {seg} | Churn: {prob:.1%}",
                     new_x="LMARGIN", new_y="NEXT")

    # Сохраняем в байты
    return pdf.output()


def get_csv_download_link(csv_str: str, filename: str = "churnguard_predictions.csv") -> str:
    """
    Генерирует base64-ссылку для скачивания CSV через Streamlit.

    Args:
        csv_str: CSV-строка
        filename: имя файла

    Returns:
        base64 data URI
    """
    b64 = base64.b64encode(csv_str.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Скачать {filename}</a>'


def get_pickle_download_link(pickle_bytes: bytes, filename: str = "churnguard_model.pkl") -> str:
    """
    Генерирует base64-ссылку для скачивания pickle-модели.

    Args:
        pickle_bytes: байты модели
        filename: имя файла

    Returns:
        base64 data URI
    """
    b64 = base64.b64encode(pickle_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">🧠 Скачать модель ({filename})</a>'
