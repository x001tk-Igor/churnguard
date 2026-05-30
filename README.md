# 🛡️ ChurnGuard — предиктивная аналитика оттока клиентов

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ChurnGuard** — это web-приложение, которое превращает CSV с клиентскими данными в полноценную предиктивную систему удержания клиентов за 5 шагов.

Загрузи файл → получи:
- 📊 **Data Health Report** — качество данных и авто-исправления
- 🎯 **RFM-сегментацию** — кто Champions, а кто At Risk
- 🧠 **ML-модель оттока** — сравнение 3 моделей, SHAP-объяснения
- 📈 **Интерактивный дашборд** — Money at Risk, приоритеты, рекомендации
- 📥 **Экспорт** — PDF, CSV, pickle-модель

🔗 **[Живая демка](https://churnguard-cecwmy4heiptn7vkpohfoj.streamlit.app)**

---

## 🎥 Быстрый старт

```bash
# 1. Клонируй репозиторий
git clone https://github.com/YOUR_USERNAME/churnguard.git
cd churnguard

# 2. Установи зависимости
pip install -r requirements.txt

# 3. Запусти приложение
streamlit run app.py
```

Откроется браузер на `http://localhost:8501`.

---

## 🐳 Docker

```bash
docker build -t churnguard .
docker run -p 8501:8501 churnguard
```

---

## 📊 Что внутри

### 6 экранов приложения

| # | Экран | Что делает |
|---|-------|-----------|
| 1 | **Upload** | Загрузка CSV или выбор из 3 демо-датасетов (SaaS / E-com / Телеком) |
| 2 | **Profiling** | Data Health Score, пропуски, выбросы, авто-исправления |
| 3 | **RFM** | Recency / Frequency / Monetary → 5 бизнес-сегментов |
| 4 | **Training** | Logistic Regression vs Random Forest vs XGBoost, ROC-кривые, SHAP |
| 5 | **Dashboard** | Money at Risk, топ клиентов, рекомендации, executive summary |
| 6 | **Export** | PDF-отчёт, CSV с прогнозами, pickle-модель |

### 3 встроенных демо-датасета

| Датасет | Строк | Сценарий |
|---------|-------|----------|
| SaaS-метрики | 5 000 | B2B-подписки с поддержкой, фичами и NPS |
| E-commerce | 12 450 | Розница с заказами, возвратами, email-кампаниями |
| Телеком | 8 200 | Оператор связи с контрактами, звонками, роумингом |

---

## 🏗️ Архитектура

```
churnguard/
├── app.py                     # Streamlit web-интерфейс (6 view)
├── src/                       # Бизнес-логика
│   ├── data_loader.py         # Загрузка CSV, демо-датасеты, валидация схемы
│   ├── data_profiler.py       # Data Health Score, авто-исправления
│   ├── rfm_analyzer.py        # RFM-сегментация (Champions → Lost)
│   ├── feature_engineering.py # Генерация признаков для ML
│   ├── churn_trainer.py       # Сравнение 3 моделей, auto-select
│   ├── shap_explainer.py      # SHAP-интерпретация
│   ├── recommendations.py     # Бизнес-рекомендации (rule-based engine)
│   ├── drift_detector.py      # Детектор дрифта данных (PSI, KS-тест)
│   └── report_generator.py    # Экспорт (PDF, CSV, pickle)
├── tools/
│   └── generate_demo_data.py  # Генератор синтетических данных
├── data/                      # Демо-датасеты (генерируются)
├── notebooks/                 # Исследовательские ноутбуки
├── tests/                     # Unit-тесты
├── SPEC.md                    # Полная спецификация проекта
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🎯 Какие компетенции демонстрирует проект

### Data Analyst
- Работа с реальными (грязными) данными — не Titanic/Iris
- Полный цикл: загрузка → очистка → анализ → визуализация → рекомендации
- RFM-сегментация — знание методологии
- Интерактивный дашборд в вебе

### AI/ML Engineer
- Сравнение моделей с обоснованием выбора (AUC-ROC, Precision, Recall, F1)
- SHAP-интерпретация — production requirement
- Детектор дрифта данных (PSI, KS-тест) — MLOps-зрелость
- Калибровка вероятностей + выбор порога по business-метрике
- Код в `.py`-файлах, готовый к продакшену

### Business Analyst
- Методология RFM — формальный фреймворк сегментации
- Бизнес-рекомендации с каналами, тоном и ROI
- Настраиваемые определения метрик (горизонт оттока)
- Executive summary для CEO
- Процесс: AS-IS → TO-BE → ROI-оценка

---

## 📋 Минимальные требования к CSV

```
client_id          : str — уникальный идентификатор клиента
activity_date      : date — дата последней активности
churned            : bool — целевая переменная (1 = ушёл, 0 = активен)

+ хотя бы одна метрика ценности:
total_revenue / monthly_fee / monthly_charges — для Monetary (RFM)

+ хотя бы одна метрика частоты:
transaction_count / total_orders — для Frequency (RFM)
```

---

## 🧪 Тестирование

```bash
pytest tests/ -v
```

---

## 📄 Лицензия

MIT © 2026

---

## 🤝 Связанные проекты

- **ReqForge** — AI-формализатор бизнес-требований (coming soon)
- **DataHealth** — автоматический аудит качества данных (coming soon)
