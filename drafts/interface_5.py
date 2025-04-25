import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# === ИМИТАЦИЯ ФУНКЦИИ ПОЛУЧЕНИЯ СИГНАЛОВ ===
def get_signals(crypto: str, model: str):
    now = datetime.now()
    signals = []
    for i in range(10):
        signal_time = now - timedelta(minutes=random.randint(1, 2880))  # до 2 суток назад
        signal_price = round(random.uniform(1000, 3000), 2)
        current_price = signal_price + round(random.uniform(-50, 50), 2)
        delta = current_price - signal_price

        duration = now - signal_time
        hours = duration.total_seconds() / 3600

        if hours < 1:
            signal_type = "Короткая"
        elif hours < 12:
            signal_type = "Средняя"
        else:
            signal_type = "Длительная"

        signals.append({
            "Время сигнала": signal_time.strftime("%Y-%m-%d %H:%M"),
            "Тип сигнала": signal_type,
            "Цена на момент сигнала": signal_price,
            "Текущая цена": current_price,
            "Разница": delta
        })
    return pd.DataFrame(signals)

# === НАСТРОЙКА STREAMLIT ===
st.set_page_config(page_title="Сигналы по криптовалютам", layout="wide")
st.title("📡 Сигналы для трейдинга криптовалют")

# --- Выбор параметров ---
cryptos = ["Bitcoin", "Ethereum", "Solana", "BNB"]
models = ["Strategy A", "Strategy B", "Strategy C"]

col1, col2 = st.columns(2)
with col1:
    selected_crypto = st.selectbox("Выберите криптовалюту", cryptos)
with col2:
    selected_model = st.selectbox("Выберите модель", models)

# --- Получение и отображение сигналов ---
st.subheader(f"📋 Сигналы по {selected_crypto} с моделью {selected_model}")
df = get_signals(selected_crypto, selected_model)

# --- ОЦВЕТКА ---
def highlight_signal(row):
    if row["Разница"] > 0:
        return [''] * 4 + ['background-color: green; color: white']
    elif row["Разница"] < 0:
        return [''] * 4 + ['background-color: red; color: white']
    else:
        return [''] * 4 + ['background-color: white; color: black']

styled_df = df.style.apply(highlight_signal, axis=1)

st.dataframe(styled_df, use_container_width=True)
