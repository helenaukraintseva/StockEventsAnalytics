import streamlit as st
import pandas as pd
import os

# Заголовок приложения
st.title("🚀 AutoML")

# 1️⃣ 📂 **Загрузка датасета**
uploaded_file = st.file_uploader("Загрузите CSV или Excel-файл с данными", type=["csv", "xlsx"])
if uploaded_file:
    # Определяем формат файла и читаем его
    file_extension = os.path.splitext(uploaded_file.name)[-1]
    if file_extension == ".csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("📊 **Загруженный датасет (первые 5 строк):**")
    st.dataframe(data.head())

    # 2️⃣ **Выбор типа ML-задачи**
    task_type = st.selectbox("Выберите тип задачи", ["Классификация", "Временной ряд", "Регрессия"])

    # 3️⃣ **Выбор целевой переменной**
    target_column = st.selectbox("Выберите целевую переменную", data.columns)

    # 4️⃣ **Выбор метрик для сравнения моделей**
    default_metrics = ["Accuracy", "AUC", "F1", "Precision", "Recall"] if task_type == "Классификация" else ["RMSE",
                                                                                                             "MAE",
                                                                                                             "R2"]
    selected_metrics = st.multiselect("Выберите метрики", default_metrics, default=default_metrics)

    # 5️⃣ **Дополнительные настройки**
    experiment_name = st.text_input("Введите название эксперимента (по желанию)", "AutoML_Exp")
    session_id = st.number_input("Введите случайный seed (для воспроизводимости)", value=123, min_value=1, step=1)

    # 6️⃣ 📄 **Загрузка файла с описанием моделей (по желанию)**
    models_file = st.file_uploader("Загрузите файл с описанием моделей (CSV, XLSX)", type=["csv", "xlsx"])
    if models_file:
        models_extension = os.path.splitext(models_file.name)[-1]
        if models_extension == ".csv":
            models_info = pd.read_csv(models_file)
        else:
            models_info = pd.read_excel(models_file)

        st.write("📌 **Загруженные описания моделей:**")
        st.dataframe(models_info)

    # 7️⃣ 🚀 **Запуск AutoML**
    if st.button("▶ Запустить AutoML"):
        st.write("⏳ **Инициализация систему...**")

        # 8️⃣ **Отображение результатов**
        st.write("🎯 **Лучшие модели:**")

