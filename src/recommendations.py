"""
Генератор бизнес-рекомендаций для ChurnGuard.

Rule-based движок, который на основе:
- RFM-сегмента
- Вероятности оттока
- SHAP-факторов риска
- LTV/ценности клиента

...генерирует конкретное бизнес-действие с каналом, тоном и оценкой ROI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ruff: noqa: E501

# ---------------------------------------------------------------------------
# Правила по сегментам
# ---------------------------------------------------------------------------

SEGMENT_ACTIONS = {
    "Champions": {
        "priority": "low",
        "action": "Программа лояльности. Попросить рефералов и кейс-стади.",
        "channel": "email",
        "tone": "благодарность",
        "urgency": "none",
        "offer": "Ранний доступ к новым фичам",
        "retention_lift_pct": 0.95,  # 95% останутся и так
    },
    "Loyal": {
        "priority": "medium",
        "action": "Upsell до Premium. Показать неиспользуемые фичи. Пригласить на вебинар.",
        "channel": "in-app + email",
        "tone": "вовлечение",
        "urgency": "low",
        "offer": "Бесплатный месяц Premium при переходе на годовой план",
        "retention_lift_pct": 0.85,
    },
    "At Risk": {
        "priority": "critical",
        "action": "Персональный звонок менеджера. Спецпредложение. Решить все открытые тикеты.",
        "channel": "phone + email",
        "tone": "срочность",
        "urgency": "critical",
        "offer": "Промокод 20% с expiry 7 дней. Персональный план удержания.",
        "retention_lift_pct": 0.50,
    },
    "Hibernating": {
        "priority": "high",
        "action": "Реактивационная email-цепочка «Мы скучаем». Опрос NPS. Показ новых фичей.",
        "channel": "email",
        "tone": "теплота",
        "urgency": "medium",
        "offer": "Бесплатный аудит использования. Скидка 15% на возвращение.",
        "retention_lift_pct": 0.30,
    },
    "Lost": {
        "priority": "low",
        "action": "Win-back кампания. Глубокий дисконт или не тратить бюджет.",
        "channel": "email",
        "tone": "последний шанс",
        "urgency": "low",
        "offer": "Скидка 30% на первые 3 месяца при возвращении",
        "retention_lift_pct": 0.10,
    },
}

# ---------------------------------------------------------------------------
# SHAP-факторы → дополнительные рекомендации
# ---------------------------------------------------------------------------

SHAP_ADVICE = {
    "days_since_last_login": "📅 Клиент давно не заходил — отправить персонализированное push-уведомление с новым контентом.",
    "days_since_last_order": "🛒 Давно не было заказов — предложить персональную подборку на основе истории.",
    "support_tickets_last_30d": "🎫 Много открытых тикетов — эскалировать в support, решить до звонка по удержанию.",
    "calls_to_support": "📞 Частые звонки в поддержку — провести аудит качества обслуживания.",
    "subscription_tier_basic": "💳 Базовый тариф — предложить апгрейд со скидкой за лояльность.",
    "contract_type_month-to-month": "📄 Помесячный контракт — предложить годовой со скидкой 15%.",
    "login_frequency_decline": "📉 Снижение частоты логинов — триггер реактивационной кампании.",
    "nps_score": "📊 Низкий NPS — провести глубинное интервью, понять причины недовольства.",
    "has_technical_issues": "🔧 Технические проблемы — направить в техническую поддержку с высоким приоритетом.",
    "competitor_offer_seen": "🏢 Видел предложение конкурента — подготовить встречное предложение с match-скидкой.",
    "payment_delay_count": "💸 Задержки платежей — проверить платёжный метод, предложить рассрочку.",
    "return_rate": "↩️ Высокий процент возвратов — проанализировать причины, возможно проблема в товаре.",
    "email_open_rate": "📧 Не открывает письма — сменить канал на SMS/push/звонок.",
    "has_app_installed": "📱 Нет приложения — предложить установку с бонусом за первый заказ.",
    "has_dedicated_manager": "👤 Нет персонального менеджера — назначить для Enterprise-клиентов.",
    "avg_session_duration_min": "⏱️ Короткие сессии — упростить onboarding, добавить гайды.",
    "feature_adoption_count": "🔧 Мало используемых фичей — провести training, показать value.",
    "international_plan": "🌍 Международный план — проверить качество связи в роуминге.",
    "data_usage_gb": "📶 Низкое потребление данных — возможно, клиент использует конкурента для данных.",
    "discount_usage_pct": "🏷️ Высокая доля скидочных заказов — клиент чувствителен к цене, не повышать.",
    "reviews_written": "✍️ Не пишет отзывы — попросить отзыв за бонусные баллы.",
}


def get_shap_advice(shap_factors: list[str], top_n: int = 3) -> list[str]:
    """
    Возвращает советы на основе SHAP-факторов.

    Args:
        shap_factors: список названий признаков (из explain_client)
        top_n: сколько советов возвращать

    Returns:
        Список строк-советов на русском языке
    """
    advice = []
    for factor in shap_factors:
        # Ищем по частичному совпадению
        for key, text in SHAP_ADVICE.items():
            if key in factor.lower().replace(" ", "_"):
                if text not in advice:
                    advice.append(text)
                break
        if len(advice) >= top_n:
            break
    return advice[:top_n]


def generate_recommendations(
    rfm_df: pd.DataFrame,
    churn_probs: np.ndarray,
    shap_explanation: dict | None = None,
    churn_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Генерирует персональные рекомендации для каждого клиента.

    Args:
        rfm_df: DataFrame с результатами RFM-сегментации
        churn_probs: массив вероятностей оттока
        shap_explanation: результат explain_model() (опционально)
        churn_threshold: порог для пометки "высокого риска"

    Returns:
        DataFrame с колонками: client_id, segment, churn_prob, churn_risk_level,
        ltv, action, channel, tone, offer, urgency, roi_estimate
    """
    result = rfm_df[["client_id"]].copy()

    result["rfm_segment"] = rfm_df["rfm_segment"].values
    result["churn_prob"] = churn_probs

    # Уровень риска
    result["churn_risk_level"] = np.where(
        churn_probs >= churn_threshold, "high",
        np.where(churn_probs >= 0.4, "medium", "low"),
    )

    # Расчёт приблизительного LTV (если есть monetary-данные)
    monetary_col = None
    for cand in ["total_revenue", "monthly_fee", "monthly_charges", "total_charges"]:
        norm = cand.lower().strip().replace(" ", "_")
        cols_map = {c.lower().strip().replace(" ", "_"): c for c in rfm_df.columns}
        if norm in cols_map:
            monetary_col = cols_map[norm]
            break

    if monetary_col:
        result["ltv"] = rfm_df[monetary_col].fillna(0).values
    else:
        result["ltv"] = 100.0  # default

    # --- Применяем правила ---
    actions = []
    channels = []
    tones = []
    offers = []
    urgencies = []
    roi_estimates = []

    for _, row in result.iterrows():
        segment = row["rfm_segment"]
        risk = row["churn_risk_level"]
        ltv = row["ltv"]

        seg_rule = SEGMENT_ACTIONS.get(segment, SEGMENT_ACTIONS["Lost"])

        # Модифицируем действие в зависимости от уровня риска
        if risk == "high" and segment in ("At Risk", "Hibernating"):
            action = f"🔥 СРОЧНО: {seg_rule['action']}"
            channel = "phone + sms + email"
            urgency = "critical"
            offer = seg_rule["offer"]
        elif risk == "medium" and segment in ("At Risk", "Hibernating"):
            action = f"⚠ {seg_rule['action']}"
            channel = seg_rule["channel"]
            urgency = "high"
            offer = seg_rule["offer"]
        else:
            action = seg_rule["action"]
            channel = seg_rule["channel"]
            urgency = seg_rule["urgency"]
            offer = seg_rule["offer"]

        # ROI: оцениваем, сколько денег сохраним
        retention_lift = seg_rule["retention_lift_pct"]
        if risk == "high":
            retention_lift *= 0.7  # труднее удержать
        elif risk == "low":
            retention_lift *= 1.1  # легче удержать
        roi = ltv * (1 - retention_lift) * row["churn_prob"]

        actions.append(action)
        channels.append(channel)
        tones.append(seg_rule["tone"])
        offers.append(offer)
        urgencies.append(urgency)
        roi_estimates.append(round(roi, 2))

    result["action"] = actions
    result["channel"] = channels
    result["tone"] = tones
    result["offer"] = offers
    result["urgency"] = urgencies
    result["roi_estimate"] = roi_estimates

    # Сортируем: сначала критические, потом по убыванию LTV×ChurnProb
    result["_sort_score"] = result["ltv"] * result["churn_prob"]
    result = result.sort_values(
        ["urgency", "_sort_score"],
        ascending=[True, False],
    ).drop(columns=["_sort_score"])

    return result


def generate_executive_summary(
    segment_stats: pd.DataFrame,
    churn_rate: float,
    total_clients: int,
    total_money_at_risk: float,
    model_auc: float,
    top_risk_factor: str,
) -> str:
    """
    Генерирует executive summary для CEO — 1 страница текста.

    Returns:
        Markdown-строка с резюме
    """
    at_risk_count = int(
        segment_stats.loc[segment_stats["rfm_segment"] == "At Risk", "client_count"].sum()
        if "At Risk" in segment_stats["rfm_segment"].values else 0
    )
    hibernating_count = int(
        segment_stats.loc[segment_stats["rfm_segment"] == "Hibernating", "client_count"].sum()
        if "Hibernating" in segment_stats["rfm_segment"].values else 0
    )

    summary = f"""## 📊 Executive Summary: Анализ оттока клиентов

### Текущая ситуация
- **Всего клиентов:** {total_clients:,}
- **Уровень оттока:** {churn_rate:.1%}
- **Деньги под угрозой:** ${total_money_at_risk:,.0f}
- **Качество модели:** AUC-ROC = {model_auc:.2f}

### Ключевые сегменты
- 🔥 **At Risk:** {at_risk_count} клиентов — были ценными, но теряют активность. **Приоритет #1.**
- 💤 **Hibernating:** {hibernating_count} клиентов — низкая активность, требуют реактивации.

### Главный фактор оттока
**{top_risk_factor}** — основной драйвер оттока. Рекомендуется сфокусировать retention-усилия на этом направлении.

### Рекомендуемые действия
1. **Сегодня:** обзвонить топ-20 At Risk клиентов с наибольшим LTV
2. **На этой неделе:** запустить реактивационную кампанию для Hibernating
3. **В этом месяце:** внедрить предиктивный мониторинг на основе модели (AUC {model_auc:.2f})

---
*Отчёт сгенерирован ChurnGuard. Для деталей обратитесь к дашборду.*
"""
    return summary
