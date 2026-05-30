"""
SHAP-интерпретация модели оттока для ChurnGuard.

Объясняет:
- Глобальную важность признаков (feature importance)
- Локальные объяснения для отдельных клиентов (waterfall)
- Топ-факторы риска для каждого предсказания
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

# ruff: noqa: E501

logger = logging.getLogger(__name__)


def explain_model(
    model,
    X: pd.DataFrame,
    max_display: int = 15,
    sample_size: int = 500,
) -> dict:
    """
    Вычисляет SHAP-значения и глобальную важность признаков.

    Использует TreeExplainer для деревьев (XGBoost, RF)
    или KernelExplainer для линейных моделей.

    Args:
        model: обученная модель (sklearn/XGBoost)
        X: матрица признаков
        max_display: сколько признаков показывать
        sample_size: размер выборки для KernelExplainer (если не дерево)

    Returns:
        {
            "feature_importance": {feature_name: mean_abs_shap, ...},
            "shap_values": np.ndarray (n_samples, n_features),
            "base_value": float,
        }
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return _explain_with_shap(model, X, max_display, sample_size)
        except Exception as e:
            logger.warning(f"SHAP не удался, использую fallback: {e}")
            return _explain_fallback(model, X, max_display)


def _explain_with_shap(
    model,
    X: pd.DataFrame,
    max_display: int,
    sample_size: int,
) -> dict:
    """SHAP-объяснение через библиотеку shap."""
    import shap

    X_sample = X.values.astype(float)

    # Определяем тип объяснителя
    try:
        # Сначала пробуем TreeExplainer (для XGBoost, RandomForest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # TreeExplainer может вернуть список для мультиклассовой
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Берём класс "churned"
    except Exception:
        # Fallback: KernelExplainer на подвыборке
        n_background = min(sample_size, len(X_sample))
        background = X_sample[np.random.choice(len(X_sample), n_background, replace=False)]
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            background,
        )
        shap_values = explainer.shap_values(X_sample[:sample_size], nsamples=100)

    # Глобальная важность: среднее абсолютных SHAP-значений
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = {
        name: round(float(val), 6)
        for name, val in zip(X.columns, mean_abs_shap)
    }
    # Сортируем по убыванию и берём top
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:max_display]
    )

    base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, list) else float(explainer.expected_value[0])

    return {
        "feature_importance": feature_importance,
        "shap_values": shap_values,
        "base_value": base_value,
    }


def _explain_fallback(model, X: pd.DataFrame, max_display: int) -> dict:
    """Fallback-объяснение через feature_importances_ или coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = np.ones(len(X.columns))

    feat_imp = {
        name: round(float(imp), 6)
        for name, imp in zip(X.columns, importances)
    }
    feat_imp = dict(
        sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:max_display]
    )

    return {
        "feature_importance": feat_imp,
        "shap_values": np.zeros((len(X), len(X.columns))),
        "base_value": 0.5,
    }


def explain_client(
    shap_explanation: dict,
    X: pd.DataFrame,
    client_index: int = 0,
    top_factors: int = 5,
) -> dict:
    """
    Возвращает текстовое объяснение для конкретного клиента.

    Args:
        shap_explanation: результат explain_model()
        X: матрица признаков (DataFrame)
        client_index: индекс клиента в X
        top_factors: сколько факторов показывать

    Returns:
        {
            "client_index": int,
            "top_risk_factors": [(feature, contribution, direction), ...],
            "top_protective_factors": [(feature, contribution, direction), ...],
            "summary": str,
        }
    """
    shap_vals = shap_explanation["shap_values"]
    base = shap_explanation["base_value"]
    feature_names = list(X.columns)

    if client_index >= len(shap_vals):
        client_index = 0

    client_shap = shap_vals[client_index]

    # Сортируем признаки по абсолютному влиянию
    ranked = sorted(
        zip(feature_names, client_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    risk_factors = []
    protective_factors = []

    for feat, val in ranked[:top_factors]:
        direction = "повышает риск" if val > 0 else "снижает риск"
        contribution = abs(val)
        if val > 0:
            risk_factors.append((feat, round(contribution, 4), direction))
        else:
            protective_factors.append((feat, round(contribution, 4), direction))

    # Генерируем текстовое резюме
    total_effect = float(client_shap.sum())
    predicted_prob = float(1.0 / (1.0 + np.exp(-(base + total_effect))))

    if risk_factors:
        top_risk = risk_factors[0]
        summary = (
            f"Основной фактор риска: **{top_risk[0]}** ({top_risk[1]:.3f}). "
            f"Вероятность оттока: **{predicted_prob:.1%}**."
        )
    else:
        summary = f"Значимых факторов риска не выявлено. Вероятность оттока: **{predicted_prob:.1%}**."

    return {
        "client_index": client_index,
        "top_risk_factors": risk_factors,
        "top_protective_factors": protective_factors,
        "total_shap_effect": total_effect,
        "base_value": base,
        "predicted_probability": predicted_prob,
        "summary": summary,
    }


def format_shap_for_display(feature_importance: dict[str, float]) -> list[dict]:
    """
    Форматирует важность признаков для отображения в UI.

    Returns:
        [
            {"feature": "days_since_last_login", "importance": 0.34, "pct": 34},
            ...
        ]
    """
    total = sum(feature_importance.values())
    if total == 0:
        total = 1

    return [
        {
            "feature": feat,
            "importance": round(val, 4),
            "pct": round(val / total * 100, 1),
        }
        for feat, val in feature_importance.items()
    ]
