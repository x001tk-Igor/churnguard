"""
ChurnGuard — предиктивная аналитика оттока клиентов.
Streamlit web-приложение.

6 экранов:
1. Welcome / Upload — загрузка CSV или выбор демо-датасета
2. Data Profiling — Data Health Score и отчёт по качеству
3. RFM Analysis — RFM-сегментация и heatmap
4. Model Training — сравнение 3 моделей, SHAP
5. Main Dashboard — Money at Risk, топ клиентов, рекомендации
6. Export — PDF, CSV, модель
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ruff: noqa: E501

# Добавляем src/ в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    DEMO_DATASETS,
    load_demo_dataset,
    load_user_csv,
    validate_columns,
)
from src.data_profiler import auto_fix_dataframe, profile_dataframe
from src.rfm_analyzer import (
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    compute_rfm,
    get_segment_stats,
)
from src.feature_engineering import engineer_features
from src.churn_trainer import train_and_compare
from src.shap_explainer import explain_model, format_shap_for_display
from src.recommendations import (
    generate_executive_summary,
    generate_recommendations,
)
from src.drift_detector import detect_drift, get_drift_alerts
from src.report_generator import (
    export_model_pickle,
    export_predictions_csv,
    generate_pdf_report,
    get_csv_download_link,
    get_pickle_download_link,
)


# ---------------------------------------------------------------------------
# Настройки страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard — Аналитика оттока",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #1A1C24;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9E9E9E;
        margin-top: 4px;
    }
    .metric-value.warning { color: #FF9800; }
    .metric-value.danger { color: #F44336; }
    .stProgress > div > div {
        background: linear-gradient(90deg, #F44336, #FF9800, #4CAF50);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Инициализация сессии
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "step": 0,                       # 0=Upload, 1=Profiling, 2=RFM, 3=Training, 4=Dashboard, 5=Export
        "df_raw": None,                  # Исходный DataFrame
        "df_clean": None,                # Очищенный DataFrame
        "health_report": None,           # DataHealthReport
        "rfm_df": None,                  # DataFrame с RFM-сегментацией
        "segment_stats": None,           # Статистика по сегментам
        "X": None,                       # Матрица признаков
        "y": None,                       # Целевая переменная
        "feature_names": [],             # Названия признаков
        "training_report": None,         # TrainingReport
        "shap_explanation": None,        # SHAP-результаты
        "recommendations_df": None,      # Таблица рекомендаций
        "filename": "unknown.csv",
        "churn_threshold": 0.5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ---------------------------------------------------------------------------
# Sidebar: навигация
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("🛡️ ChurnGuard")
        st.caption("Предиктивная аналитика оттока")

        steps = [
            "📤 Загрузка данных",
            "📊 Профилирование",
            "🎯 RFM-сегментация",
            "🧠 Обучение модели",
            "📈 Дашборд",
            "📥 Экспорт",
        ]

        current = st.session_state.step
        for i, label in enumerate(steps):
            if i < current:
                st.markdown(f"✅ ~~{label}~~")
            elif i == current:
                st.markdown(f"**→ {label}**")
            else:
                st.markdown(f"⚪ {label}")

        st.divider()

        # Настройки
        st.subheader("⚙️ Настройки")
        st.session_state.churn_threshold = st.slider(
            "Порог вероятности оттока",
            min_value=0.3, max_value=0.9, value=0.5, step=0.05,
            help="Клиенты с вероятностью выше порога помечаются как 'высокий риск'",
        )

        st.divider()
        st.caption("© 2026 ChurnGuard")


# ---------------------------------------------------------------------------
# View 1: Upload
# ---------------------------------------------------------------------------
def render_upload():
    st.title("🛡️ ChurnGuard")
    st.subheader("Предиктивная аналитика оттока клиентов")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Загрузите CSV-файл")
        uploaded_file = st.file_uploader(
            "Перетащите файл или нажмите для выбора",
            type=["csv"],
            help="CSV с клиентскими данными. Минимально: client_id, activity_date, churned",
        )

        if uploaded_file:
            with st.spinner("Загрузка и анализ..."):
                df = load_user_csv(uploaded_file)
                if df is not None and not df.empty:
                    is_valid, missing_req, missing_rec = validate_columns(df)
                    if not is_valid:
                        st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_req)}")
                        st.info("Добавьте колонки и попробуйте снова. "
                                "Или выберите демо-датасет справа.")
                    else:
                        st.session_state.df_raw = df
                        st.session_state.filename = uploaded_file.name
                        st.session_state.step = 1
                        st.rerun()

    with col2:
        st.markdown("### 🎲 Или выберите демо-датасет")

        for label, filename in DEMO_DATASETS.items():
            if st.button(label, use_container_width=True):
                with st.spinner(f"Загрузка {label}..."):
                    df = load_demo_dataset(filename)
                    if df is not None:
                        st.session_state.df_raw = df
                        st.session_state.filename = filename
                        st.session_state.step = 1
                        st.rerun()
                    else:
                        st.error(f"Демо-датасет {filename} не найден. "
                                 f"Запустите `python tools/generate_demo_data.py`")

    # Как это работает
    st.divider()
    st.markdown("### 🔄 Как это работает")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊", "Профилирование", "Качество данных")
    c2.metric("🎯", "RFM", "Сегментация")
    c3.metric("🧠", "ML", "3 модели")
    c4.metric("📈", "Дашборд", "Рекомендации")


# ---------------------------------------------------------------------------
# View 2: Data Profiling
# ---------------------------------------------------------------------------
def render_profiling():
    st.title("📊 Data Health Report")

    df = st.session_state.df_raw
    st.caption(f"Файл: **{st.session_state.filename}** | {len(df):,} строк × {len(df.columns)} колонок")

    with st.spinner("Профилируем данные..."):
        report = profile_dataframe(df, st.session_state.filename)
        st.session_state.health_report = report

    # Health Score
    score = report.overall_score
    color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"

    st.markdown(f"""
    <div style="background:#1A1C24;border:1px solid #2A2D3A;border-radius:12px;padding:24px;text-align:center;margin:16px 0;">
        <div style="font-size:0.85rem;color:#9E9E9E;">DATA HEALTH SCORE</div>
        <div style="font-size:4rem;font-weight:700;color:{color};">{score}<span style="font-size:1.5rem;">/100</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Проблемы
    if report.global_issues:
        st.markdown("### ⚠ Найденные проблемы")
        for issue in report.global_issues:
            st.warning(issue)
    else:
        st.success("✅ Критических проблем не обнаружено")

    # Детали по колонкам
    st.markdown("### 📋 Детали по колонкам")

    col_data = []
    for c in report.columns:
        col_data.append({
            "Колонка": c.name,
            "Тип": c.dtype,
            "Пропуски": f"{c.missing_pct:.1f}%",
            "Выбросы": f"{c.outlier_pct:.1f}%" if c.outlier_pct > 0 else "—",
            "Статус": "✅" if c.score >= 70 else "⚠" if c.score >= 40 else "🔴",
            "Проблемы": ", ".join(c.issues) if c.issues else "—",
        })
    st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    # Действия
    st.divider()
    c1, c2, c3 = st.columns(3)

    if c1.button("🔧 Авто-исправление и продолжить", use_container_width=True, type="primary"):
        with st.spinner("Исправляем данные..."):
            df_clean = auto_fix_dataframe(df, report)
            st.session_state.df_clean = df_clean
            st.session_state.step = 2
            st.rerun()

    if c2.button("▶ Продолжить без исправлений", use_container_width=True):
        st.session_state.df_clean = df.copy()  # noqa: DF292
        st.session_state.step = 2
        st.rerun()

    if c3.button("↩ Назад к загрузке", use_container_width=True):
        st.session_state.step = 0
        st.rerun()


# ---------------------------------------------------------------------------
# View 3: RFM Analysis
# ---------------------------------------------------------------------------
def render_rfm():
    st.title("🎯 RFM-сегментация клиентов")

    df = st.session_state.df_clean

    with st.spinner("Вычисляем RFM-сегментацию..."):
        try:
            rfm_df = compute_rfm(df)
        except ValueError as e:
            st.error(str(e))
            st.info("Убедитесь, что в данных есть колонки для Recency, Frequency и Monetary. "
                    "Проверьте документацию в SPEC.md.")
            if st.button("↩ Назад"):
                st.session_state.step = 0
                st.rerun()
            return

        st.session_state.rfm_df = rfm_df
        segment_stats = get_segment_stats(rfm_df)
        st.session_state.segment_stats = segment_stats

    # Ключевые цифры
    total = len(rfm_df)

    col1, col2, col3, col4 = st.columns(4)
    at_risk = int(segment_stats.loc[segment_stats["rfm_segment"] == "At Risk", "client_count"].sum()) if "At Risk" in segment_stats["rfm_segment"].values else 0
    champions = int(segment_stats.loc[segment_stats["rfm_segment"] == "Champions", "client_count"].sum()) if "Champions" in segment_stats["rfm_segment"].values else 0

    col1.metric("👥 Всего клиентов", f"{total:,}")
    col2.metric("🏆 Champions", f"{champions:,} ({champions/total*100:.1f}%)")
    col3.metric("⚠️ At Risk", f"{at_risk:,} ({at_risk/total*100:.1f}%)")
    churn_rate = float(rfm_df["churned"].mean() * 100)
    col4.metric("📉 Отток", f"{churn_rate:.1f}%")

    # Распределение по сегментам
    st.markdown("### Распределение по сегментам")

    colors_map = {s: SEGMENT_COLORS.get(s, "#757575") for s in segment_stats["rfm_segment"]}

    fig = px.bar(
        segment_stats,
        x="rfm_segment",
        y="client_count",
        color="rfm_segment",
        color_discrete_map=colors_map,
        labels={"rfm_segment": "Сегмент", "client_count": "Клиентов"},
        text=segment_stats["client_pct"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Таблица сегментов
    st.markdown("### 📋 Детали по сегментам")
    display_stats = segment_stats.copy()
    display_stats["segment_label"] = display_stats["rfm_segment"].map(SEGMENT_LABELS)
    display_stats["avg_r"] = display_stats["avg_r_score"].round(1)
    display_stats["avg_f"] = display_stats["avg_f_score"].round(1)
    display_stats["avg_m"] = display_stats["avg_m_score"].round(1)
    st.dataframe(
        display_stats[["segment_label", "client_count", "client_pct", "avg_r", "avg_f", "avg_m"]],
        use_container_width=True,
        hide_index=True,
    )

    # Действия
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🧠 Обучить модель оттока →", use_container_width=True, type="primary"):
        st.session_state.step = 3
        st.rerun()
    if c2.button("↩ Назад", use_container_width=True):
        st.session_state.step = 1
        st.rerun()


# ---------------------------------------------------------------------------
# View 4: Model Training
# ---------------------------------------------------------------------------
def render_training():
    st.title("🧠 Обучение модели оттока")

    df = st.session_state.df_clean

    # Feature engineering
    with st.spinner("Генерируем признаки..."):
        target_col = "churned"
        y = df[target_col].copy()
        X, feature_names, feat_meta = engineer_features(
            df.drop(columns=[target_col]) if target_col in df.columns else df,
            target_col=target_col,
        )
        # Если churned был в df, инженерим без него
        if target_col in df.columns:
            X2, feature_names, feat_meta = engineer_features(df)
            # Но target не должен быть в X
            if target_col in X2.columns:
                X2 = X2.drop(columns=[target_col])
                feature_names = [n for n in feature_names if n != target_col]
            X = X2

        st.session_state.X = X
        st.session_state.y = y
        st.session_state.feature_names = feature_names

    st.caption(f"Признаков: {len(feature_names)} | Строк: {len(X):,} | Отток: {y.mean():.1%}")

    # Обучение
    with st.spinner("Обучаем и сравниваем 3 модели..."):
        report = train_and_compare(X, y)
        st.session_state.training_report = report

    st.success(f"✅ Обучение завершено за {report.total_time_sec:.1f} сек")

    # Сравнение моделей
    st.markdown("### 📊 Сравнение моделей")

    comparison = []
    for m in report.models:
        comparison.append({
            "Модель": f"🏆 {m.name}" if m.name == report.best_model_name else m.name,
            "AUC-ROC": f"{m.auc_roc:.3f}",
            "Precision": f"{m.precision:.3f}",
            "Recall": f"{m.recall:.3f}",
            "F1": f"{m.f1:.3f}",
            "CV AUC (mean ± std)": f"{m.cv_auc_mean:.3f} ± {m.cv_auc_std:.3f}",
            "Время": f"{m.train_time_sec:.1f}с",
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    # ROC-кривые
    st.markdown("### 📈 ROC-кривые")
    fig_roc = go.Figure()
    for m in report.models:
        if m.fpr is not None and m.tpr is not None:
            fig_roc.add_trace(go.Scatter(
                x=m.fpr, y=m.tpr,
                mode="lines",
                name=f"{m.name} (AUC={m.auc_roc:.3f})",
            ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random",
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Лучшая модель: обучение на всех данных
    st.divider()
    st.markdown(f"### 🏆 Лучшая модель: **{report.best_model_name}**")

    # SHAP
    with st.spinner("Вычисляем SHAP-объяснения..."):
        best_model = report.model_objects[report.best_model_name]
        shap_explanation = explain_model(best_model, X)
        st.session_state.shap_explanation = shap_explanation

    # Топ-факторы
    if shap_explanation["feature_importance"]:
        fi_display = format_shap_for_display(shap_explanation["feature_importance"])
        fi_df = pd.DataFrame(fi_display)
        fi_df = fi_df.rename(columns={"feature": "Признак", "pct": "Вклад %", "importance": "Важность"})
        fi_df["Вклад %"] = fi_df["Вклад %"].apply(lambda x: f"{x:.1f}%")

        st.markdown("#### 🔍 Топ-факторы оттока (SHAP)")
        fig_fi = px.bar(
            fi_df.head(10),
            x="Важность",
            y="Признак",
            orientation="h",
            color="Важность",
            color_continuous_scale="Reds",
        )
        fig_fi.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Действия
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("📈 Перейти к дашборду →", use_container_width=True, type="primary"):
        # Генерируем рекомендации
        with st.spinner("Генерируем рекомендации..."):
            churn_probs = best_model.predict_proba(X.values.astype(float))[:, 1]
            recs = generate_recommendations(
                st.session_state.rfm_df,
                churn_probs,
                shap_explanation,
                st.session_state.churn_threshold,
            )
            st.session_state.recommendations_df = recs
            st.session_state.step = 4
            st.rerun()

    if c2.button("↩ Назад", use_container_width=True):
        st.session_state.step = 2
        st.rerun()


# ---------------------------------------------------------------------------
# View 5: Main Dashboard
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("📈 Дашборд оттока клиентов")

    df = st.session_state.df_clean
    rfm_df = st.session_state.rfm_df
    recs = st.session_state.recommendations_df
    report = st.session_state.training_report
    segment_stats = st.session_state.segment_stats
    shap_expl = st.session_state.shap_explanation

    churn_probs = recs["churn_prob"].values
    total_clients = len(recs)
    churn_rate = float(churn_probs.mean())
    avg_ltv = float(recs["ltv"].mean())

    # Деньги под угрозой: сумма LTV клиентов с высокой вероятностью оттока
    high_risk_mask = recs["churn_risk_level"] == "high"
    money_at_risk = float(recs.loc[high_risk_mask, "ltv"].sum())

    # --- Верхняя строка: KPI ---
    st.markdown("### 🔑 Ключевые показатели")

    col1, col2, col3, col4, col5 = st.columns(5)

    money_color = "danger" if money_at_risk > avg_ltv * total_clients * 0.1 else "warning"
    cols = [
        ("💰 Money at Risk", f"${money_at_risk:,.0f}", money_color),
        ("📉 Средний риск оттока", f"{churn_rate:.1%}", "danger" if churn_rate > 0.2 else "normal"),
        ("🧠 AUC-ROC модели", f"{report.best_auc:.3f}", "normal"),
        ("👥 Всего клиентов", f"{total_clients:,}", "normal"),
        ("💵 Средний LTV", f"${avg_ltv:,.0f}", "normal"),
    ]

    for col, (label, value, color_class) in zip([col1, col2, col3, col4, col5], cols):
        with col:
            css_class = f" {color_class}" if color_class != "normal" else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value{css_class}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- График: вероятность оттока по сегментам ---
    st.markdown("### 📊 Вероятность оттока по сегментам")

    seg_prob = recs.groupby("rfm_segment")["churn_prob"].mean().reset_index()
    seg_prob["color"] = seg_prob["rfm_segment"].map(SEGMENT_COLORS)

    fig_seg = px.bar(
        seg_prob,
        x="rfm_segment",
        y="churn_prob",
        color="rfm_segment",
        color_discrete_map={s: SEGMENT_COLORS.get(s, "#757575") for s in seg_prob["rfm_segment"]},
        labels={"rfm_segment": "Сегмент", "churn_prob": "Средняя вероятность оттока"},
        text=seg_prob["churn_prob"].apply(lambda x: f"{x:.1%}"),
    )
    fig_seg.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_seg, use_container_width=True)

    # --- Топ клиентов под угрозой ---
    st.markdown("### 🔴 Топ-10 клиентов под угрозой")

    top10 = recs.head(10)
    display_top = top10[["client_id", "rfm_segment", "churn_prob", "ltv", "action"]].copy()
    display_top["churn_prob"] = display_top["churn_prob"].apply(lambda x: f"{x:.1%}")
    display_top["ltv"] = display_top["ltv"].apply(lambda x: f"${x:,.0f}")
    display_top.columns = ["Клиент", "Сегмент", "Вер. оттока", "LTV", "Действие"]

    st.dataframe(display_top, use_container_width=True, hide_index=True)

    # --- Рекомендации по сегментам ---
    st.markdown("### 💡 Рекомендации по сегментам")

    for segment in ["At Risk", "Hibernating", "Loyal", "Champions"]:
        seg_data = recs[recs["rfm_segment"] == segment]
        if seg_data.empty:
            continue

        seg_label = SEGMENT_LABELS.get(segment, segment)
        seg_color = SEGMENT_COLORS.get(segment, "#757575")
        n_clients = len(seg_data)
        money_seg = float(seg_data["ltv"].sum())
        avg_prob = float(seg_data["churn_prob"].mean())

        sample_action = seg_data.iloc[0]

        st.markdown(f"""
        <div style="background:#1A1C24;border-left:4px solid {seg_color};border-radius:0 8px 8px 0;padding:16px;margin:8px 0;">
            <strong>{seg_label}</strong> — {n_clients} клиентов, ${money_seg:,.0f}, средний риск {avg_prob:.1%}<br/>
            <span style="color:#9E9E9E;">🎯 {sample_action['action']}</span><br/>
            <span style="color:#9E9E9E;">📡 Канал: {sample_action['channel']} | 💰 Предполагаемый ROI: ${sample_action['roi_estimate']:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Executive Summary ---
    st.divider()
    st.markdown("### 📋 Executive Summary")

    top_risk_factor = "N/A"
    if shap_expl and shap_expl.get("feature_importance"):
        top_risk_factor = list(shap_expl["feature_importance"].keys())[0]

    summary = generate_executive_summary(
        segment_stats=segment_stats,
        churn_rate=churn_rate,
        total_clients=total_clients,
        total_money_at_risk=money_at_risk,
        model_auc=report.best_auc,
        top_risk_factor=top_risk_factor,
    )
    st.markdown(summary)

    # --- Дрифт данных ---
    st.markdown("### 🔬 Мониторинг дрифта данных")

    X = st.session_state.X
    with st.spinner("Проверяем дрифт..."):
        # Сравниваем первую половину со второй (симуляция)
        mid = len(X) // 2
        drift_df = detect_drift(X.iloc[:mid], X.iloc[mid:])
        alerts = get_drift_alerts(drift_df)

    for alert in alerts:
        if "Критический" in alert:
            st.error(alert)
        elif "Умеренный" in alert:
            st.warning(alert)
        else:
            st.success(alert)

    # --- Действия ---
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("📥 Перейти к экспорту →", use_container_width=True, type="primary"):
        st.session_state.step = 5
        st.rerun()
    if c2.button("↩ Назад к модели", use_container_width=True):
        st.session_state.step = 3
        st.rerun()


# ---------------------------------------------------------------------------
# View 6: Export
# ---------------------------------------------------------------------------
def render_export():
    st.title("📥 Экспорт результатов")

    recs = st.session_state.recommendations_df
    report = st.session_state.training_report

    # 1. CSV с прогнозами
    st.markdown("### 📊 CSV с прогнозами")
    csv_str = export_predictions_csv(recs)
    st.download_button(
        label="📥 Скачать predictions.csv",
        data=csv_str,
        file_name="churnguard_predictions.csv",
        mime="text/csv",
    )

    # Превью
    with st.expander("Предпросмотр CSV"):
        st.dataframe(recs.head(20), use_container_width=True, hide_index=True)

    # 2. Модель
    st.markdown("### 🧠 Модель (pickle)")
    best_model = report.model_objects[report.best_model_name]
    pickle_bytes = export_model_pickle(best_model)
    st.download_button(
        label="📥 Скачать модель (.pkl)",
        data=pickle_bytes,
        file_name="churnguard_model.pkl",
        mime="application/octet-stream",
    )
    st.caption("Модель можно загрузить через `pickle.load()` и использовать для predict_proba().")

    # 3. PDF
    st.markdown("### 📄 PDF-отчёт")
    try:
        pdf_bytes = generate_pdf_report(
            executive_summary=generate_executive_summary(
                segment_stats=st.session_state.segment_stats,
                churn_rate=float(recs["churn_prob"].mean()),
                total_clients=len(recs),
                total_money_at_risk=float(recs.loc[recs["churn_risk_level"] == "high", "ltv"].sum()),
                model_auc=report.best_auc,
                top_risk_factor=list(st.session_state.shap_explanation.get("feature_importance", {}).keys())[0] if st.session_state.shap_explanation else "N/A",
            ),
            segment_stats=st.session_state.segment_stats,
            model_results=report.models,
            recommendations_df=recs,
            churn_rate=float(recs["churn_prob"].mean()),
            total_clients=len(recs),
        )
        st.download_button(
            label="📄 Скачать отчёт (.pdf)",
            data=pdf_bytes,
            file_name="churnguard_report.pdf",
            mime="application/pdf",
        )
    except Exception:
        st.warning("PDF-отчёт недоступен. Установите `fpdf2` для этой функции.")

    # --- Действия ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    if c1.button("🔄 Начать заново", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session()
        st.rerun()
    if c2.button("📈 Вернуться к дашборду", use_container_width=True):
        st.session_state.step = 4
        st.rerun()
    if c3.button("🧠 К обучению", use_container_width=True):
        st.session_state.step = 3
        st.rerun()


# ---------------------------------------------------------------------------
# Главный роутер
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    steps = {
        0: render_upload,
        1: render_profiling,
        2: render_rfm,
        3: render_training,
        4: render_dashboard,
        5: render_export,
    }

    step = st.session_state.get("step", 0)
    render_func = steps.get(step, render_upload)
    render_func()


if __name__ == "__main__":
    main()
