"""
Streamlit-приложение для предсказания оттока клиентов банка.
Интерактивный интерфейс с вводом данных, предсказанием, SHAP-анализом и бенчмарком моделей.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ============================================================
# Настройки страницы
# ============================================================
st.set_page_config(
    page_title="🏦 Предсказание оттока клиентов",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Загрузка модели и артефактов
# ============================================================
@st.cache_resource
def load_artifacts():
    """Загрузка модели и вспомогательных файлов."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    le_gender = joblib.load(os.path.join(models_dir, "label_encoder_gender.pkl"))

    with open(os.path.join(models_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    loaded_shap_model = None
    shap_model_path = os.path.join(models_dir, "shap_model.pkl")
    if os.path.exists(shap_model_path):
        loaded_shap_model = joblib.load(shap_model_path)

    benchmark = None
    bench_path = os.path.join(models_dir, "benchmark_results.csv")
    if os.path.exists(bench_path):
        benchmark = pd.read_csv(bench_path, index_col=0)

    return model, scaler, le_gender, metadata, benchmark, loaded_shap_model


@st.cache_data
def load_dataset():
    """Загрузка исходного датасета для EDA."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "TZ.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


try:
    model, scaler, le_gender, metadata, benchmark_df, loaded_shap_model = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f" Ошибка загрузки модели: {e}")
    st.info(" Сначала запустите ноутбук churn_analysis.ipynb для обучения модели.")
    st.stop()

feature_names = metadata["feature_names"]
cities = metadata["cities"]
needs_scaling = metadata.get("needs_scaling", False)


# ============================================================
# Функция предобработки
# ============================================================
def preprocess_input(credit_score, city, gender, age, tenure, balance,
                     num_products, has_credit_card, is_active, salary):
    """Преобразование пользовательского ввода в вектор признаков."""
    gender_encoded = le_gender.transform([gender])[0]
    city_encoded = [1 if c == city else 0 for c in cities]

    features = [
        credit_score, gender_encoded, age, tenure, balance,
        num_products, has_credit_card, is_active, salary
    ] + city_encoded

    features_array = np.array(features).reshape(1, -1)

    if needs_scaling:
        features_array = scaler.transform(features_array)

    return features_array, np.array(features).reshape(1, -1)


# ============================================================
# Sidebar — ввод данных клиента
# ============================================================
st.sidebar.title("📝 Данные клиента")
st.sidebar.markdown("---")

credit_score = st.sidebar.slider("Кредитный рейтинг", 300, 900, 650, 10)
city = st.sidebar.selectbox("Город", cities)
gender = st.sidebar.selectbox("Пол", ["Male", "Female"])
age = st.sidebar.slider("Возраст", 18, 92, 35)
tenure = st.sidebar.slider("Стаж в банке (лет)", 0, 15, 5)
balance = st.sidebar.number_input("Баланс депозита", 0.0, 500000.0, 100000.0, 1000.0)
num_products = st.sidebar.selectbox("Число продуктов", [1, 2, 3, 4])
has_credit_card = st.sidebar.selectbox("Есть кредитка?", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
is_active = st.sidebar.selectbox("Активный клиент?", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
salary = st.sidebar.number_input("Оценочная зарплата", 0.0, 300000.0, 120000.0, 1000.0)

# ============================================================
# Главная часть — вкладки
# ============================================================
st.title("🏦 Предсказание оттока клиентов банка")
st.markdown(f"**Модель:** {metadata['best_model_name']} | **ROC-AUC:** {metadata['metrics'].get('ROC-AUC', 'N/A')}")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Предсказание", "📊 EDA", "🏆 Бенчмарк моделей", "ℹ️ О проекте"])

# ============================================================
# Tab 1 — Предсказание
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1])

    features_for_model, features_raw = preprocess_input(
        credit_score, city, gender, age, tenure, balance,
        num_products, has_credit_card, is_active, salary
    )

    probability = float(model.predict_proba(features_for_model)[0][1])
    prediction = int(probability >= 0.5)

    with col1:
        st.subheader("📋 Данные клиента")
        client_info = {
            "Кредитный рейтинг": credit_score,
            "Город": city,
            "Пол": "Мужской" if gender == "Male" else "Женский",
            "Возраст": age,
            "Стаж в банке": f"{tenure} лет",
            "Баланс депозита": f"{balance:,.0f} ₸",
            "Число продуктов": num_products,
            "Кредитка": "Да" if has_credit_card else "Нет",
            "Активный": "Да" if is_active else "Нет",
            "Зарплата": f"{salary:,.0f} ₸"
        }
        for k, v in client_info.items():
            st.write(f"**{k}:** {v}")

    with col2:
        st.subheader("🎯 Результат предсказания")

        if probability < 0.3:
            risk_color = "green"
            risk_level = "🟢 Низкий риск"
            risk_emoji = "✅"
        elif probability < 0.7:
            risk_color = "orange"
            risk_level = "🟡 Средний риск"
            risk_emoji = "⚠️"
        else:
            risk_color = "red"
            risk_level = "🔴 Высокий риск"
            risk_emoji = "🚨"

        st.metric("Вероятность оттока", f"{probability:.1%}")
        st.markdown(f"### {risk_emoji} {risk_level}")
        st.markdown(f"**Прогноз:** {'Клиент уйдёт' if prediction == 1 else 'Клиент останется'}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title={"text": "Вероятность оттока, %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')

    st.markdown("---")
    st.subheader("🔍 Объяснение предсказания (SHAP)")

    try:
        if loaded_shap_model is not None:
            shap_model = loaded_shap_model
            model_type_name = metadata.get('shap_model_type', type(shap_model).__name__)
            st.info(f"ℹ️ Для SHAP используется сохранённая модель: {model_type_name}")
        elif hasattr(model, 'estimators_'):
            model_type_name = type(model).__name__
            if isinstance(model.estimators_[0], tuple):
                shap_model = model.estimators_[0][1]
            else:
                shap_model = model.estimators_[0]
            st.info(f"ℹ️ Модель: {model_type_name}. Для SHAP используется базовая модель: {type(shap_model).__name__}")
        else:
            shap_model = model
        
        explainer = shap.TreeExplainer(shap_model)
        shap_values_single = explainer.shap_values(features_raw)

        shap_df = pd.DataFrame({
            "Признак": feature_names,
            "SHAP значение": shap_values_single[0],
            "Значение признака": features_raw[0]
        }).sort_values("SHAP значение", key=abs, ascending=False)

        fig_shap = px.bar(
            shap_df, x="SHAP значение", y="Признак",
            orientation='h', color="SHAP значение",
            color_continuous_scale="RdBu_r",
            title="Вклад каждого признака в предсказание"
        )
        fig_shap.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, width='stretch')

        # Интерпретация
        top_positive = shap_df[shap_df["SHAP значение"] > 0].head(3)
        top_negative = shap_df[shap_df["SHAP значение"] < 0].head(3)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔴 Факторы, увеличивающие риск оттока:**")
            for _, row in top_positive.iterrows():
                st.write(f"- {row['Признак']} (SHAP: {row['SHAP значение']:.4f})")
        with col_b:
            st.markdown("**🟢 Факторы, уменьшающие риск оттока:**")
            for _, row in top_negative.iterrows():
                st.write(f"- {row['Признак']} (SHAP: {row['SHAP значение']:.4f})")
    except Exception as e:
        st.warning(f"SHAP анализ недоступен для данного типа модели: {e}")

# ============================================================
# Tab 2 — EDA
# ============================================================
with tab2:
    st.subheader("📊 Разведочный анализ данных")

    df = load_dataset()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего клиентов", f"{len(df):,}")
        col2.metric("Ушло", f"{int(df['ушел_из_банка'].sum()):,}")
        col3.metric("Доля оттока", f"{df['ушел_из_банка'].mean():.1%}")

        st.markdown("---")

        # Распределение оттока
        col_a, col_b = st.columns(2)

        with col_a:
            fig_target = px.pie(
                df, names='ушел_из_банка',
                title='Распределение целевой переменной',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig_target.update_traces(textinfo='percent+label+value')
            st.plotly_chart(fig_target, width='stretch')

        with col_b:
            fig_city = px.histogram(
                df, x='город', color='ушел_из_банка',
                title='Отток по городам', barmode='group',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_city, width='stretch')

        # Распределения числовых признаков
        st.subheader("📈 Распределения числовых признаков")
        num_feature = st.selectbox(
            "Выберите признак:",
            ['возраст', 'кредитный_рейтинг', 'баланс_депозита',
             'стаж_в_банке', 'оценочная_зарплата', 'число_продуктов']
        )

        fig_dist = px.histogram(
            df, x=num_feature, color='ушел_из_банка',
            marginal='box', nbins=50,
            title=f'Распределение: {num_feature}',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig_dist, width='stretch')

        # Корреляция
        st.subheader("🔗 Корреляционная матрица")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['ID', 'ID_клиента']]
        corr = df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Корреляционная матрица'
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, width='stretch')

        # Статистика
        st.subheader("📋 Описательная статистика")
        st.dataframe(df.describe().round(2), width='stretch')
    else:
        st.warning("⚠️ Файл TZ.csv не найден. Поместите его в корневую папку проекта.")

# ============================================================
# Tab 3 — Бенчмарк моделей
# ============================================================
with tab3:
    st.subheader("🏆 Бенчмарк всех моделей")

    if benchmark_df is not None:
        # Таблица
        st.dataframe(
            benchmark_df.style.highlight_max(axis=0, color='#90EE90', subset=['ROC-AUC', 'F1-score', 'Recall', 'Precision'])
                              .highlight_min(axis=0, color='#FFB6C1', subset=['Время (сек)']),
            width='stretch'
        )

        st.markdown("---")

        # Графики
        col1, col2 = st.columns(2)

        with col1:
            fig_roc = px.bar(
                benchmark_df.reset_index(),
                x='ROC-AUC', y='Модель',
                orientation='h', color='ROC-AUC',
                color_continuous_scale='Viridis',
                title='ROC-AUC по моделям'
            )
            fig_roc.update_layout(height=500)
            st.plotly_chart(fig_roc, width='stretch')

        with col2:
            metrics_cols = ['ROC-AUC', 'F1-score', 'Precision', 'Recall']
            available_metrics = [c for c in metrics_cols if c in benchmark_df.columns]
            fig_metrics = go.Figure()
            for metric in available_metrics:
                fig_metrics.add_trace(go.Bar(
                    name=metric,
                    x=benchmark_df.index,
                    y=benchmark_df[metric],
                ))
            fig_metrics.update_layout(
                barmode='group', title='Сравнение метрик',
                height=500, xaxis_tickangle=-45
            )
            st.plotly_chart(fig_metrics, width='stretch')

        # Лучшая модель
        best = benchmark_df.index[0]
        st.success(f"🏆 Лучшая модель: **{best}** (ROC-AUC: {benchmark_df.loc[best, 'ROC-AUC']:.4f})")
    else:
        st.warning("⚠️ Файл benchmark_results.csv не найден.")

# ============================================================
# Tab 4 — О проекте
# ============================================================
with tab4:
    st.subheader("ℹ️ О проекте")
    st.markdown("""
    ### 🏦 Предсказание оттока клиентов банка

    **Задача:** Бинарная классификация — предсказание вероятности ухода клиента из банка.

    **Датасет:** 15 000 строк, 14 признаков.

    **Модели:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    - CatBoost
    - Bagging, AdaBoost, Gradient Boosting
    - Voting Classifier, Stacking Classifier

    **Стек технологий:**
    `Python 3.10` • `scikit-learn` • `XGBoost` • `CatBoost` • `SHAP` •
    `FastAPI` • `Streamlit` • `Docker` • `Plotly`

    ---
    **Автор:** Хазракулов Зиёбек. Для связи Телеграм: @Khazratkulov_Z.
    """)

    # Информация о модели
    st.subheader("📦 Информация о модели")
    st.json(metadata)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏦 Bank Churn Prediction | Streamlit Dashboard v1.0"
    "</div>",
    unsafe_allow_html=True
)
