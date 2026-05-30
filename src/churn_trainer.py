"""
Обучение и сравнение моделей оттока для ChurnGuard.

Обучает 3 модели:
- Logistic Regression (базовый бенчмарк, хорошо интерпретируемый)
- Random Forest (ловит нелинейные зависимости)
- XGBoost (чемпион на табличных данных)

Автоматически выбирает лучшую по AUC-ROC.
Балансирует классы через SMOTE.
Калибрует вероятности через Platt scaling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# ruff: noqa: E501


@dataclass
class ModelResult:
    """Результаты одной модели."""
    name: str
    auc_roc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    train_time_sec: float = 0.0
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    thresholds: np.ndarray | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class TrainingReport:
    """Полный отчёт об обучении."""
    models: list[ModelResult] = field(default_factory=list)
    best_model_name: str = ""
    best_auc: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    total_time_sec: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    churn_rate: float = 0.0
    model_objects: dict = field(default_factory=dict)  # {name: trained_model}


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    use_smote: bool = True,
) -> TrainingReport:
    """
    Обучает и сравнивает 3 модели.

    Args:
        X: матрица признаков
        y: целевая переменная (0/1)
        cv_folds: количество фолдов для кросс-валидации
        use_smote: использовать ли SMOTE для балансировки

    Returns:
        TrainingReport с результатами всех моделей
    """
    start_time = time.time()
    feature_names = list(X.columns)
    X_np = X.values.astype(float)
    y_np = y.values.astype(int)

    report = TrainingReport(
        feature_names=feature_names,
        n_samples=len(X),
        n_features=len(feature_names),
        churn_rate=float(y_np.mean()),
    )

    # --- Определяем модели ---
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(1 - y_np.mean()) / y_np.mean(),
            random_state=42,
            verbosity=0,
        ),
    }

    # --- Кросс-валидация ---
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        t0 = time.time()

        if use_smote and HAS_IMBLEARN:
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=42, k_neighbors=min(5, int(y_np.sum()) - 1))),
                ("model", model),
            ])
        else:
            pipeline = model

        # Кросс-валидация
        try:
            cv_scores = cross_val_score(
                pipeline if use_smote and HAS_IMBLEARN else model,
                X_np, y_np,
                cv=skf,
                scoring="roc_auc",
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = 0.0, 0.0

        # Обучение на всех данных
        if use_smote and HAS_IMBLEARN:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_np.sum()) - 1))
            X_resampled, y_resampled = smote.fit_resample(X_np, y_np)
            model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_np, y_np)

        # Калибровка вероятностей
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated.fit(X_np, y_np)

        # Предсказания
        y_prob = calibrated.predict_proba(X_np)[:, 1]
        y_pred = calibrated.predict(X_np)

        # ROC-кривая
        fpr, tpr, thresholds = roc_curve(y_np, y_prob)

        # Метрики
        auc = roc_auc_score(y_np, y_prob)
        precision = precision_score(y_np, y_pred)
        recall = recall_score(y_np, y_pred)
        f1 = f1_score(y_np, y_pred)
        accuracy = accuracy_score(y_np, y_pred)

        # Feature importance
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                importances = np.zeros(len(feature_names))

            feat_imp = {
                name: float(imp)
                for name, imp in zip(feature_names, importances)
            }
            # Сортируем по убыванию
            feat_imp = dict(
                sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
            )
        except Exception:
            feat_imp = {}

        train_time = time.time() - t0

        result = ModelResult(
            name=name,
            auc_roc=round(auc, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            accuracy=round(accuracy, 4),
            cv_auc_mean=round(cv_mean, 4),
            cv_auc_std=round(cv_std, 4),
            train_time_sec=round(train_time, 2),
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            feature_importance=feat_imp,
        )
        report.models.append(result)
        report.model_objects[name] = calibrated

    # Выбираем лучшую модель по AUC
    best = max(report.models, key=lambda m: m.auc_roc)
    report.best_model_name = best.name
    report.best_auc = best.auc_roc
    report.total_time_sec = round(time.time() - start_time, 2)

    return report


def predict_churn(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Предсказывает вероятность оттока для новых данных.

    Args:
        model: обученная модель (из TrainingReport.model_objects)
        X: матрица признаков

    Returns:
        Массив вероятностей оттока [0, 1]
    """
    return model.predict_proba(X.values.astype(float))[:, 1]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 3.0,
) -> tuple[float, float]:
    """
    Находит оптимальный порог вероятности по бизнес-метрике.

    False Negative (пропустили уходящего) стоит дороже,
    чем False Positive (ложная тревога).

    Args:
        y_true: истинные метки
        y_prob: предсказанные вероятности
        cost_fp: стоимость ложного срабатывания
        cost_fn: стоимость пропуска уходящего клиента

    Returns:
        (optimal_threshold, min_cost)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    # Для каждого порога считаем стоимость
    costs = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)

    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
