import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from settings import tokens, tokens_dict
from config import api_id, api_hash, phone
from dotenv import load_dotenv
import logging

load_dotenv()

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Настройка страницы ---
st.set_page_config(page_title="Крипто-график и статистика", layout="wide")
st.session_state.retriever = None

st.title("Криптовалютный мониторинг")

# --- Боковое меню с вкладками ---
tab = st.sidebar.radio("Меню:", [
    "Главное меню",
    "График криптовалют",
    "Предсказать направление",
    "Предсказать цену",
    "Покупка/продажа",
    "Сигналы",
    "Индикаторы",
    "Анализ новостей",
    "Crypto RAG",
])

# --- Хранение данных в сессии ---
if "price_history" not in st.session_state:
    st.session_state.price_history = {}

if tab == "Главное меню":
    st.markdown("# CryptoInsight")
    st.markdown("### Многофункциональная платформа для анализа и мониторинга криптовалютного рынка в реальном времени.")
    st.markdown("---")
    st.markdown("**Выберите режим работы в левом меню, чтобы начать.** Ниже краткое описание возможностей:")

    st.markdown("""
    ### Обзор режимов:

    - **График криптовалют**  
      Визуализация цен криптовалют с историей, обновляемой в реальном времени.

    - **Предсказать направление**  
      Расчёт технических индикаторов (MACD, RSI, EMA и др.) и определение трендов: рост, падение, флет или разворот.

    - **Предсказать цену**  
      Модели машинного обучения прогнозируют цену на ближайшие периоды. Используются LinearRegression, BayesianRidge и др.

    - **Покупка/продажа**  
      Автоматическое предсказание сигналов (buy/sell/hold) с визуализацией на графике. Выбор модели и криптовалюты.

    - **Сигналы**  
      Исторические сигналы трейдинга с оценкой прибыли (PnL), основанные на данных из БД. Поддержка фильтрации по модели.

    - **Индикаторы**  
      Большой набор технических индикаторов: MACD, RSI, SMA, ADX, Bollinger Bands и др. Гибкая настройка параметров.

    - **Анализ новостей**  
      Анализ телеграм-новостей и классификация по криптовалюте и настроению (positive, neutral, negative).

    - **Crypto RAG**  
      AI-помощник с системой Retrieval-Augmented Generation. Ответы на вопросы с учетом знаний из документов и новостей.

    ---
    Используйте боковую панель для выбора режима. Все данные обновляются в реальном времени и доступны без задержек.
    """)

elif tab == "График криптовалют":
    import plotly.express as px

    st.subheader("⚙ Настройки графика")
    crypto_options = tokens_dict

    def get_crypto_price(symbol):
        """
        Получение текущей цены криптовалюты через API Binance.
        :param symbol: Тикер криптовалюты (например, BTCUSDT)
        :return: float цена или None
        """
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except requests.exceptions.RequestException as e:
            logging.error("Ошибка получения текущей цены: %s", e)
            return None

    def get_historical_data(symbol, interval="1m", limit=100):
        """
        Загрузка исторических данных свечей.
        :param symbol: тикер (например, BTCUSDT)
        :param interval: интервал свечей (например, 1m)
        :param limit: количество точек данных
        :return: DataFrame с колонками ["time", "price"] или None
        """
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=["time", "Open", "High", "Low", "price", "Volume", *["_"]*6])
            df = df[["time", "price"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms") + pd.Timedelta(hours=3)
            df["price"] = df["price"].astype(float)
            return df
        except requests.exceptions.RequestException as e:
            logging.error("Ошибка загрузки исторических данных: %s", e)
            return None

    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()), key="crypto_graph")

    with col2:
        update_interval = st.selectbox("Частота обновления (мин):", [1, 5, 10, 15, 30], key="interval_graph")

    if selected_crypto not in st.session_state.price_history:
        st.session_state.price_history[selected_crypto] = []

    placeholder = st.empty()
    if len(st.session_state.price_history[selected_crypto]) < 100:
        st.write("\U0001F504 Собираем данные...")
        historical_data = get_historical_data(crypto_options[selected_crypto])
        if historical_data is not None:
            st.session_state.price_history[selected_crypto] = historical_data.to_dict("records")

    @st.cache_data(ttl=60)
    def load_historical_data(symbol):
        """
        Кэшированная загрузка исторических данных.
        :param symbol: тикер
        :return: DataFrame
        """
        return get_historical_data(symbol)

    symbol = crypto_options[selected_crypto]
    df = load_historical_data(symbol)
    current_price = get_crypto_price(symbol)

    if current_price and df is not None:
        now = pd.Timestamp.now()
        new_row = pd.DataFrame([{"time": now, "price": current_price}])
        df = pd.concat([df, new_row], ignore_index=True).tail(100)

        fig = px.line(
            df,
            x="time",
            y="price",
            title=f"График {selected_crypto}",
            labels={"time": "Время", "price": "Цена (USDT)"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Не удалось загрузить данные с Binance")

elif tab == "Предсказать направление":
    import plotly.graph_objects as go
    from ta.trend import MACD
    from ta.momentum import RSIIndicator

    st.title("Криптовалютный график и тренд-прогноз")

    # --- Настройки ---
    tokens_dict = {
        "BTC/USDT": "BTCUSDT",
        "ETH/USDT": "ETHUSDT",
        "BNB/USDT": "BNBUSDT"
    }

    selected_crypto = st.selectbox("Криптовалюта:", list(tokens_dict.keys()))
    interval = st.selectbox("Интервал:", ["1m", "5m", "10m"])
    limit = st.slider("Количество точек истории:", 50, 200, 100)

    def detect_trend_signals(df, trend_window):
        """
        Вычисляет направление тренда на основе EMA, волатильности, MACD и RSI.

        :param df: DataFrame с колонкой "price"
        :param trend_window: окно для вычисления сигнала
        :return: тип сигнала (строка) и обновлённый DataFrame
        """
        df = df.copy()
        df["ema"] = df["price"].ewm(span=trend_window, adjust=False).mean()
        df["diff"] = df["ema"].diff()
        df["volatility"] = df["price"].rolling(window=trend_window).std()
        df["signal_strength"] = df["diff"] / df["volatility"]

        recent_strength = df["signal_strength"].tail(trend_window)
        avg_strength = recent_strength.mean()

        if abs(avg_strength) < 0.1:
            signal = "flat"
        elif avg_strength > 0.1:
            signal = "trend_up"
        elif avg_strength < -0.1:
            signal = "trend_down"
        else:
            signal = "none"

        macd = MACD(close=df["price"]).macd()
        rsi = RSIIndicator(close=df["price"]).rsi()

        last_rsi = rsi.iloc[-1]

        if signal == "trend_down" and last_rsi < 30:
            signal = "reversal_up"
        elif signal == "trend_up" and last_rsi > 70:
            signal = "reversal_down"

        return signal, df

    def get_historical_data(symbol, interval="1m", limit=100):
        """
        Получает исторические данные с Binance.

        :param symbol: тикер (например BTCUSDT)
        :param interval: интервал (1m, 5m, ...)
        :param limit: количество точек
        :return: DataFrame с колонками time и price
        """
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=["time", "Open", "High", "Low", "price", "Volume", *["_"]*6])
            df = df[["time", "price"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms") + pd.Timedelta(hours=3)
            df["price"] = df["price"].astype(float)
            return df
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка загрузки данных: {e}")
            return None

    # --- Загрузка и анализ ---
    symbol = tokens_dict[selected_crypto]
    df = get_historical_data(symbol=symbol, interval=interval, limit=limit)

    if df is not None:
        trend_window = max(5, limit // 20)
        trend_signal, df = detect_trend_signals(df, trend_window)
        df["Close"] = df["price"]

        # --- Визуализация ---
        colors = {
            "trend_up": "green",
            "trend_down": "red",
            "reversal_up": "blue",
            "reversal_down": "orange",
            "flat": "gray",
            "none": "white"
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["price"],
            mode="lines",
            name="Цена",
            line=dict(color=colors.get(trend_signal, "white"))
        ))

        fig.update_layout(
            title=f"{selected_crypto} ({interval}) — тренд: {trend_signal}",
            template="plotly_dark",
            xaxis_title="Время",
            yaxis_title="Цена"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Интерпретация ---
        st.subheader("Тренд:")
        st.metric("Сигнал", trend_signal)

        st.write("Логика определения тренда:")
        st.markdown(f"""
        - Используется **EMA({trend_window})**, волатильность и нормализованное изменение
        - Используются технические индикаторы: **MACD**, **RSI**
        - **trend_up**: устойчивый рост + нормализованная производная > 0
        - **trend_down**: устойчивое падение
        - **flat**: нет ярко выраженного движения
        - **reversal_up / down**: развороты на основе RSI
        """)

elif tab == "Покупка/продажа":
    import joblib
    import matplotlib.pyplot as plt

    st.title("Прогнозирование покупки/продажи криптовалют")

    crypto_options = tokens_dict
    MODEL_DIR = "models/"

    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей ("1m", "5m" и т.п.)
        :param window_size: Размер окна для модели
        :param forecast_horizon: Шаги вперёд
        :param reserve_steps: Резерв на случай ошибок
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume"])
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df = df.astype({col: float for col in ["open", "high", "low", "close", "volume"]})
            return df
        except Exception as e:
            st.error(f"\u274C Ошибка при запросе данных с Binance: {e}")
            return pd.DataFrame()

    @st.cache_resource
    def load_ml_model(name):
        """
        Загружает модель машинного обучения из файла.

        :param name: Название модели
        :return: Объект модели
        """
        name = name.replace(" ", "")
        return joblib.load(f"{MODEL_DIR}{name}_signal.pkl")

    model_options = {
        "AdaBoost": "AdaBoost",
        "Decision Tree": "DecisionTree",
        "GaussianNB": "GaussianNB",
        "GradientBoosting": "GradientBoosting",
        "KNN": "KNN",
        "Logistic Regression": "LogisticRegression",
        "Random Forest": "RandomForest",
    }

    scaler = joblib.load(MODEL_DIR + "scaler_signal.pkl")

    interval_options = {
        "1 минута": "1m",
        "5 минут": "5m",
        "15 минут": "15m"
    }

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()))
    with col2:
        interval = st.selectbox("Выберите интервал:", ["1m", "5m", "15m", "1h", "1d"])
    with col3:
        selected_type_model = st.selectbox("Выберите модель:", ["Машинное обучение"])
    with col4:
        selected_model = st.selectbox("Выберите алгоритм:", list(model_options.keys()))

    model = load_ml_model(selected_model)
    symbol = crypto_options[selected_crypto]

    st.subheader(f"\U0001F4CA График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)
    df = df.drop("time", axis=1)

    def build_windowed_features(df, window_size=20):
        """
        Преобразует временной ряд в обучающие окна.

        :param df: DataFrame с колонками [open, high, low, close, volume]
        :param window_size: Размер окна
        :return: DataFrame окон
        """
        features = []
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i + window_size].values.flatten()
            features.append(window)
        return pd.DataFrame(features)

    if df is not None:
        df_display = df.copy()
        df = build_windowed_features(df, window_size=20)

        st.subheader("\U0001F4C8 Предсказание")
        X_scaled = df  # X_scaled = scaler.transform(df) — по желанию
        y_pred = model.predict(X_scaled)

        df_pred = df_display.iloc[-len(y_pred):].copy()
        df_pred["Signal"] = y_pred

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_pred.index, df_pred["close"], label="Цена", color="blue")
        ax.scatter(df_pred.index[df_pred["Signal"] == 1], df_pred["close"][df_pred["Signal"] == 1],
                   color="green", label="\U0001F4C8 Покупка", marker="^")
        ax.scatter(df_pred.index[df_pred["Signal"] == -1], df_pred["close"][df_pred["Signal"] == -1],
                   color="red", label="\U0001F4C9 Продажа", marker="v")
        ax.set_title(f"{symbol} ({interval}) - {selected_model}")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена (USDT)")
        ax.legend()
        st.pyplot(fig)

        last_signal = df_pred["Signal"].iloc[-1]
        if last_signal == 1:
            recommendation = "\U0001F4C8 **Покупка**"
            recommendation_color = "green"
        elif last_signal == -1:
            recommendation = "\U0001F4C9 **Продажа**"
            recommendation_color = "red"
        else:
            recommendation = "\u23F3 **Держите**"
            recommendation_color = "gray"

        st.markdown(f"<h2 style='color: {recommendation_color}; text-align: center;'>{recommendation}</h2>", unsafe_allow_html=True)
        st.subheader(f"\U0001F4C8 Прогноз модели {selected_model}: **{recommendation}**")

elif tab == "Сигналы":
    import psycopg2
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    st.title("Сигналы для трейдинга криптовалют")

    cryptos = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    models = ["MACD", "RSI", "ICHIMOKU"]

    def fetch_recent_signals(symbol: str, model: str):
        """
        Загружает сигналы из PostgreSQL за последние 6 часов.

        :param symbol: Название криптовалюты
        :param model: Название модели
        :return: DataFrame сигналов
        """
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
            )
            query = """
                SELECT *
                FROM signals
                WHERE symbol = %s
                  AND model = %s
                  AND end_time IS NOT NULL
                  AND end_time >= NOW() - INTERVAL '6 hours';
            """
            df = pd.read_sql_query(query, conn, params=(symbol, model))
            conn.close()
            logging.info("Загружено %d сигналов для %s (%s)", len(df), symbol, model)
            return df
        except Exception as e:
            logging.error("Ошибка подключения к БД: %s", e)
            return pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту", cryptos)
    with col2:
        selected_model = st.selectbox("Выберите модель", models)

    df = fetch_recent_signals(symbol=selected_crypto, model=selected_model)
    if df.empty:
        st.warning("Нет данных для отображения")
    else:
        df["delta"] = df["end_price"] - df["start_price"]
        st.subheader(f"\U0001F4CB Сигналы по {selected_crypto} ({selected_model})")

        def highlight_signal(row):
            """
            Подсвечивает строки по дельте результата.

            :param row: Строка DataFrame
            :return: Стили для строки
            """
            if row["delta"] > 0:
                return [''] * 5 + ['background-color: green; color: white']
            elif row["delta"] < 0:
                return [''] * 5 + ['background-color: red; color: white']
            else:
                return [''] * 5 + ['background-color: gray; color: black']

        styled_df = df[[
            "start_time", "end_time", "signal", "start_price",
            "end_price", "delta"
        ]].style.apply(highlight_signal, axis=1)

        st.dataframe(styled_df, use_container_width=True)

elif tab == "Индикаторы":
    import algorithms
    import matplotlib.pyplot as plt
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    st.title("Crypto Trading Predictor")

    # --- Доступные алгоритмы ---
    ALGORITHMS = {
        "RSI": algorithms.AlgorithmRSI,
        "SMA": algorithms.AlgorithmSMA,
        "EMA": algorithms.AlgorithmEMA,
        "MACD": algorithms.AlgorithmMACD,
        "ADX": algorithms.AlgorithmADX,
        "Stochastic": algorithms.AlgorithmStochastic,
        "WilliamsR": algorithms.AlgorithmWilliamsR,
        "OBV": algorithms.AlgorithmOBV,
        "VMAP": algorithms.AlgorithmVWAP,
        "BollingerBands": algorithms.AlgorithmBollingerBands,
        "ATR": algorithms.AlgorithmATR,
        "ARIMA": algorithms.AlgorithmARIMA,
    }

    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей ("1m", "5m" и т.п.)
        :param window_size: Размер окна
        :param forecast_horizon: Сколько шагов вперёд будет предсказано
        :param reserve_steps: Резервное количество свечей
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume"])
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        except Exception as e:
            logging.error("Ошибка при запросе данных с Binance: %s", e)
            return pd.DataFrame()

    symbols = tokens
    symbol = st.selectbox("Выберите криптовалюту", symbols, index=symbols.index("BTCUSDT"))
    interval = st.selectbox("Выберите интервал", ["1m", "5m", "15m", "1h", "1d"])
    selected_algorithm = st.selectbox("Выберите алгоритм", list(ALGORITHMS.keys()))

    st.subheader(f"График {symbol} ({interval})")

    df = get_crypto_data(symbol, interval)
    if df.empty:
        st.error("Не удалось загрузить данные.")
    else:
        df["Close"] = df["close"]
        algorithm = ALGORITHMS[selected_algorithm]()

        # Выбор параметров по алгоритму
        st.write("Выберите параметры для индикатора")
        if selected_algorithm in ["SMA", "EMA", "RSI", "ADX", "ATR"]:
            window_range = range(5, 30)
            window_param = st.selectbox("Размер окна", list(window_range))
            algorithm.get_param(int(window_param))
        elif selected_algorithm == "MACD":
            fastperiod = st.selectbox("Fast Period", list(range(1, 10)))
            slowperiod = st.selectbox("Slow Period", list(range(11, 20)))
            algorithm.get_param(fastperiod=int(fastperiod), slowperiod=int(slowperiod))
        elif selected_algorithm == "Stochastic":
            k_period = st.selectbox("%K период", list(range(10, 20)))
            d_period = st.selectbox("%D период", list(range(1, 10)))
            algorithm.get_param(k_period=int(k_period), d_period=int(d_period))
        elif selected_algorithm == "WilliamsR":
            period = st.selectbox("Период", list(range(5, 20)))
            algorithm.get_param(period=int(period))
        elif selected_algorithm == "BollingerBands":
            window = st.selectbox("Период SMA", list(range(5, 30)))
            nbdev_up = st.selectbox("Коэф. вверх", [round(i * 0.1, 1) for i in range(1, 50)], index=20)
            nbdev_dn = st.selectbox("Коэф. вниз", [round(i * 0.1, 1) for i in range(1, 50)], index=20)
            algorithm.get_param(window=window, nbdev_up=nbdev_up, nbdev_dn=nbdev_dn)
        elif selected_algorithm == "ARIMA":
            p = st.selectbox("AR порядок", list(range(1, 5)))
            d = st.selectbox("DIFF порядок", list(range(0, 5)))
            q = st.selectbox("MA порядок", list(range(1, 5)))
            algorithm.get_param(p=p, d=d, q=q)
        else:
            st.info("Индикатор не требует параметров.")

        df = algorithm.run(df)

        # Визуализация
        fig, (ax, ax_ind) = plt.subplots(2, 1, figsize=(12, 8))
        ax.plot(df["time"], df["close"], label="Цена", color="blue")
        ax.set_title(f"{symbol} ({interval})")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена (USDT)")

        if "Signal" in df:
            ax.scatter(df["time"][df["Signal"] == 1], df["close"][df["Signal"] == 1], color="green", label="Покупка", marker="^")
            ax.scatter(df["time"][df["Signal"] == -1], df["close"][df["Signal"] == -1], color="red", label="Продажа", marker="v")

        # Пример отображения индикатора на втором графике (если есть нужные колонки)
        if selected_algorithm == "RSI" and "RSI" in df:
            ax_ind.plot(df["time"], df["RSI"], label="RSI", color="purple")
            ax_ind.axhline(70, color='green', linestyle='--')
            ax_ind.axhline(30, color='red', linestyle='--')
        elif selected_algorithm == "MACD" and "MACD" in df:
            ax_ind.plot(df["time"], df["MACD"], label="MACD", color="purple")
            ax_ind.plot(df["time"], df["Signal_line"], label="Signal", color="red", linestyle="--")

        ax.legend()
        ax.grid(True)
        ax_ind.legend()
        ax_ind.grid(True)
        st.pyplot(fig)

        # Вывод последнего сигнала
        if "Signal" in df:
            last_signal = df["Signal"].iloc[-1]
            if last_signal == 1:
                st.success("✅ Рекомендация: Покупать!")
            elif last_signal == -1:
                st.error("❌ Рекомендация: Продавать!")
            else:
                st.info("🔍 Рекомендация: Держать (нет явного сигнала).")

elif tab == "Предсказать цену":
    import joblib
    import plotly.graph_objects as go
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    st.title("Прогнозирование цены криптовалюты")

    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей (например, "1m")
        :param window_size: Размер окна для анализа
        :param forecast_horizon: Горизонт прогноза
        :param reserve_steps: Дополнительные шаги
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume"])
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df = df.astype({col: float for col in ["open", "high", "low", "close", "volume"]})
            return df

        except Exception as e:
            logging.error("Ошибка при запросе данных с Binance: %s", e)
            return pd.DataFrame()

    def predict_future_prices(model_name, last_sequence, model_dir="models", n_steps=100):
        """
        Предсказание будущих значений модели на основе последнего окна.

        :param model_name: имя модели (файл .pkl)
        :param last_sequence: последний батч (np.array) формы (1, -1)
        :param model_dir: путь к директории моделей
        :param n_steps: количество шагов вперёд
        :return: DataFrame с колонками time и PredictedValue
        """
        model = joblib.load(f"{model_dir}/{model_name}")
        x_scaler = joblib.load(f"{model_dir}/scaler_v2.pkl")
        y_min, y_max = joblib.load(f"{model_dir}/y_norm_params.pkl")

        def unscale_y(y_scaled):
            return y_scaled * (y_max - y_min) / 0.1 + y_min

        generated = []
        for _ in range(n_steps):
            input_scaled = x_scaler.transform(last_sequence)
            pred_scaled = model.predict(input_scaled)[0]
            pred_real = unscale_y(pred_scaled)
            generated.append(pred_real)
            new_step = last_sequence[0][-1].copy()
            new_step[0] = pred_scaled
            last_sequence = np.vstack([last_sequence[:, 1:], [new_step]])

        return generated

    crypto_options = tokens_dict
    intervals = ["1m", "5m", "30m"]

    model_options = {
        "Linear Regression": "LinearRegression_v3.pkl",
        "BayesianRidge": "BayesianRidge_v3.pkl",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_crypto = st.selectbox("Криптовалюта", list(crypto_options.keys()))
    with col2:
        interval = st.selectbox("Интервал", intervals)
    with col3:
        selected_model = st.selectbox("Модель", list(model_options.keys()))

    symbol = crypto_options[selected_crypto]
    df = get_crypto_data(symbol, interval)

    if df.empty:
        st.error("❌ Не удалось загрузить данные. Попробуйте позже.")
    else:
        st.success("✅ Данные успешно загружены")

        last_price = df["close"].iloc[-1]
        last_time = df["time"].iloc[-1]
        time_delta = df["time"].diff().median()
        recent_window = df.select_dtypes(include=[np.number]).tail(20)
        last_sequence = recent_window.values.reshape(1, -1)

        model_filename = model_options[selected_model]
        predicted_raw = predict_future_prices(model_filename, last_sequence=last_sequence, n_steps=30)

        future_times = [last_time + (i + 1) * time_delta for i in range(len(predicted_raw))]
        df_pred = pd.DataFrame({"time": future_times, "PredictedValue": predicted_raw})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["close"],
            mode="lines",
            name="Фактическая цена",
            line=dict(color="blue")
        ))

        fig.add_trace(go.Scatter(
            x=df_pred["time"],
            y=df_pred["PredictedValue"],
            mode="lines",
            name="Прогноз цены",
            line=dict(color="red", dash="dash")
        ))

        fig.update_layout(
            title=f"{symbol} — Прогноз цены ({interval})",
            xaxis_title="Время",
            yaxis_title="Цена (USDT)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Последнее предсказание: **{round(df_pred.iloc[-1]['PredictedValue'], 4)} USDT**")

elif tab == "Crypto RAG":
    st.title("RAG-система для вопросов")

    if st.session_state.retriever is None:
        with st.spinner("Инициализация моделей и индекса..."):
            import google.generativeai as genai
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain.text_splitter import CharacterTextSplitter
            from langchain_community.document_loaders import TextLoader
            from config import gemini_api

            def prepare_model(text_path: str = "text_1.txt", model_name: str = "all-MiniLM-L6-v2"):
                """
                Подготавливает модель генерации и индекс ретривера.

                :param text_path: путь к текстовому файлу
                :param model_name: имя модели эмбеддингов
                :return: модель Gemini и ретривер
                """
                genai.configure(api_key=gemini_api)
                model = genai.GenerativeModel("models/gemini-2.0-flash-lite-001")

                loader = TextLoader(text_path, encoding="utf-8")
                documents = loader.load()

                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever()

                return model, retriever

            def answer_query(model, retriever, query: str) -> str:
                """
                Отвечает на вопрос с помощью модели Gemini и контекста из ретривера.

                :param model: генеративная модель
                :param retriever: объект ретривера
                :param query: строка вопроса
                :return: строка ответа
                """
                answer = f"\n❓ Вопрос: {query}\n"
                docs = retriever.invoke(query)

                if not docs:
                    return answer + "🤷 Ничего не найдено."

                answer += f"\n🔎 Найдено {len(docs)} релевантных фрагментов:\n"
                context = ""
                for i, doc in enumerate(docs, 1):
                    answer += f"\n📄 Фрагмент {i}:\n{doc.page_content}\n"
                    context += doc.page_content + "\n"

                prompt = f"Контекст:\n{context}\n\nВопрос: {query}"
                st.write("\n⏳ Отправка запроса к Gemini...")
                response = model.generate_content(prompt)
                answer += "\nОтвет от Gemini:\n\n"
                answer += response.text
                return answer

            model, retriever = prepare_model()
            st.session_state.retriever = retriever
            st.session_state.model = model

    query = st.text_input("Введите ваш вопрос:", "")

    if st.button("Получить ответ") and query.strip():
        with st.spinner("Думаю..."):
            response = answer_query(st.session_state.model, st.session_state.retriever, query)
            st.success(response)


elif tab == "Анализ новостей":
    import joblib
    from sklearn.pipeline import Pipeline
    from parsing_news.telegram_4 import parse_telegram_news

    st.title("Анализ крипто-новостей по источнику и времени")

    def sentiment_color(sentiment: str) -> str:
        """
        Возвращает цвет фона в зависимости от настроения.

        :param sentiment: строка с меткой (Positive, Negative, Neutral)
        :return: цвет фона в HEX
        """
        mapping = {
            "positive": "#d4edda",
            "negative": "#f8d7da",
            "neutral": "#e2e3e5"
        }
        return mapping.get(sentiment.lower(), "#ffffff")

    # --- Загрузка моделей ---
    crypto_pipe = joblib.load("NLP/sentiment_model/crypto_classifier_model.pkl")
    sentiment_pipe = joblib.load("NLP/sentiment_model/felt_classifier_model.pkl")
    label_encoder = joblib.load("NLP/sentiment_model/label_encoder.pkl")

    def fetch_news(source: str, days_back: int):
        """
        Загружает новости из Telegram-канала.

        :param source: название канала
        :param days_back: сколько дней назад брать новости
        :return: список словарей с новостями
        """
        return parse_telegram_news(days_back=days_back, channel_title=source,
                                   api_id=api_id, api_hash=api_hash, phone=phone)

    def process_news(news_list: list) -> list:
        """
        Обрабатывает новости: классифицирует крипто-контент и настроение.

        :param news_list: список словарей новостей
        :return: отфильтрованные и размеченные новости
        """
        results = []
        for news in news_list:
            text = news.get("text", "")
            is_crypto = crypto_pipe.predict([text])[0]
            if is_crypto:
                sentiment = sentiment_pipe.predict([text])[0]
                sentiment_label = label_encoder.inverse_transform([sentiment])[0]
                results.append({
                    "Дата": news.get("date"),
                    "Время": news.get("time"),
                    "Новость": text,
                    "Настроение": sentiment_label,
                    "Ссылка": news.get("url", "-")
                })
        return results

    source = st.selectbox("Источник новостей:", [
        "if_market_news", "web3news", "cryptodaily", "slezisatoshi"])
    days_back = st.slider("За сколько дней назад брать новости?", 1, 30, 7)

    if st.button("Анализировать"):
        with st.spinner("Загружаем и анализируем..."):
            raw_news = fetch_news(source, days_back)
            processed_news = process_news(raw_news)

            if processed_news:
                df = pd.DataFrame(processed_news)
                st.success(f"Обнаружено крипто-новостей: {len(df)}")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Крипто-новости не найдены по указанным параметрам.")
