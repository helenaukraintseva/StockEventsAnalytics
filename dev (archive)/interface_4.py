import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# === ИМИТАЦИЯ ПОЛУЧЕНИЯ СИГНАЛОВ ===
def get_signals(crypto: str, model: str):
    now = datetime.now()
    signals = []
    for i in range(15):
        signal_time = now - timedelta(minutes=random.randint(1, 2880))  # до 2 суток назад
        signal_price = round(random.uniform(1000, 3000), 2)
        current_price = signal_price + round(random.uniform(-50, 50), 2)
        delta = current_price - signal_price

        duration = now - signal_time
        hours = duration.total_seconds() / 3600

        # Тип сигнала
        if hours < 1:
            signal_type = "Короткая"
        elif hours < 12:
            signal_type = "Средняя"
        else:
            signal_type = "Длительная"

        # Сигнал
        signal_action = "Покупать" if delta > 0 else "Продавать"

        # Закрыта ли сделка?
        is_closed = duration.total_seconds() > 1800  # более 30 минут

        signals.append({
            "Время сигнала": signal_time,
            "Тип сигнала": signal_type,
            "Сигнал": signal_action,
            "Цена на момент сигнала": signal_price,
            "Текущая цена": current_price,
            "Разница": delta,
            "Закрытая сделка": is_closed
        })
    return pd.DataFrame(signals)

# === НАСТРОЙКИ STREAMLIT ===
st.set_page_config(page_title="Сигналы по криптовалютам", layout="wide")
st.title("📡 Сигналы для трейдинга криптовалют")

# --- Параметры пользователя ---
cryptos = ["Bitcoin", "Ethereum", "Solana", "BNB"]
models = ["Strategy A", "Strategy B", "Strategy C"]

col1, col2 = st.columns(2)
with col1:
    selected_crypto = st.selectbox("Выберите криптовалюту", cryptos)
with col2:
    selected_model = st.selectbox("Выберите модель", models)

show_closed = st.checkbox("🔓 Отображать закрытые сделки", value=False)

# --- Получение данных ---
st.subheader(f"📋 Сигналы по {selected_crypto} с моделью {selected_model}")
df = get_signals(selected_crypto, selected_model)

# --- Фильтрация ---
if not show_closed:
    df = df[df["Закрытая сделка"] == False]

# --- Визуальная подсветка по разнице ---
def highlight_signal(row):
    # Вычисляем цвет только для колонки "Разница"
    if row["Разница"] > 0:
        return [''] * 5 + ['background-color: green; color: white']
    elif row["Разница"] < 0:
        return [''] * 5 + ['background-color: red; color: white']
    else:
        return [''] * 5 + ['background-color: white; color: black']

# --- Преобразование времени в строку (для отображения) ---
df["Время сигнала"] = df["Время сигнала"].dt.strftime("%Y-%m-%d %H:%M")

# --- Показ таблицы ---
styled_df = df[[
    "Время сигнала", "Тип сигнала", "Сигнал", "Цена на момент сигнала",
    "Текущая цена", "Разница"
]].style.apply(highlight_signal, axis=1)

st.dataframe(styled_df, use_container_width=True)
