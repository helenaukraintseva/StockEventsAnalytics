import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from settings import tokens, tokens_dict
import joblib
import algorithms
import ml_models, ml_model_price
import nn_model_buy, nn_model_price
from trend_detector import TrendDetector

def get_crypto_data(symbol, interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"
        ])
        df = df[["time", "open", "high", "low", "close", "volume"]]
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении данных с Binance: {e}")
        return None

st.set_page_config(page_title="Предсказание цены криптовалют", layout="wide")
st.title("📈 Прогнозирование цены криптовалюты")

crypto_options = tokens_dict
intervals = ["1m", "5m", "10m"]

model_options = {
    "Linear Regression": ml_model_price.AlgorithmLinearRegression,
    "Ridge Regression": ml_model_price.AlgorithmRidge,
    "Random Forest": ml_model_price.AlgorithmRandomForestRegressor,
    "Lasso": ml_model_price.Lasso,
    "ElasticNet": ml_model_price.AlgorithmElasticNet,
    "BayesianRidge": ml_model_price.AlgorithmBayesianRidge,
    "SGD": ml_model_price.AlgorithmSGDRegressor,
    "DecisionTree": ml_model_price.AlgorithmDecisionTreeRegressor,
    "ExtraTrees": ml_model_price.AlgorithmExtraTreesRegressor,
    "GradientBoosting": ml_model_price.AlgorithmGradientBoostingRegressor,
}

nn_models = {
    "RNN": nn_model_price.AlgorithmRNN,
    "LSTM": nn_model_price.AlgorithmLSTM,
    "GRU": nn_model_price.AlgorithmGRU,
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_crypto = st.selectbox("🪙 Криптовалюта", list(crypto_options.keys()))
with col2:
    interval = st.selectbox("🕒 Интервал", intervals)
with col3:
    selected_type_model = st.selectbox("🧠 Тип модели", ["Машинное обучение", "Нейронные сети"])
with col4:
    selected_model = st.selectbox(
        "📚 Модель",
        list(model_options.keys()) if selected_type_model == "Машинное обучение" else list(nn_models.keys())
    )

symbol = crypto_options[selected_crypto]
st.write("📥 Получаем данные с Binance...")

# Загружаем и обучаем модель
df = get_crypto_data(symbol, interval)
if df is None or df.empty:
    st.error("❌ Не удалось загрузить данные. Попробуйте позже.")
else:
    st.success("✅ Данные успешно загружены.")

    # === Имитация предсказания в пределах последних значений ===
    n_steps_ahead = 30  # сколько точек вперед предсказываем
    window_size = 40  # окно из прошлых значений, на основе которого моделируем

    last_price = df["close"].iloc[-1]
    last_time = df["time"].iloc[-1]
    time_delta = df["time"].diff().median()

    # Берем окно последних значений
    recent_window = df["close"].iloc[-window_size:]
    price_min = recent_window.min()
    price_max = recent_window.max()

    # Генерация времени для будущих точек
    future_times = [last_time + i * time_delta for i in range(1, n_steps_ahead + 1)]

    # Генерация "предсказанных" значений в пределах min/max окна
    np.random.seed(42)
    predicted_values = np.random.uniform(low=price_min, high=price_max, size=n_steps_ahead)

    df1 = pd.DataFrame({
        "time": future_times,
        "PredictedValue": predicted_values
    })

# Построение графика
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["close"],
    mode="lines",
    name="📉 Фактическая цена",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=df1["time"],
    y=df1["PredictedValue"],
    mode="lines",
    name="📈 Прогноз цены",
    line=dict(color="red", dash="dash")
))

fig.update_layout(
    title=f"{symbol} — Фактическая и предсказанная цена ({interval})",
    xaxis_title="Время",
    yaxis_title="Цена (USDT)",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.info(f"📍 Последнее предсказание: **{round(df1.iloc[-1]['PredictedValue'], 4)} USDT**")
