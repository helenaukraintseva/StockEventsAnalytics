import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import random
from settings import tokens, tokens_dict
from config import api_id, api_hash, phone

# --- Настройка страницы ---
st.set_page_config(page_title="Крипто-график и статистика", layout="wide")
st.session_state.retriever = None

st.title("📊 Криптовалютный мониторинг")

# --- Боковое меню с вкладками ---
tab = st.sidebar.radio("Меню:", ["📊 Главное меню",
                                 "📈 График криптовалют",
                                 # "📈 Статистика",
                                 "📈 Предсказать направление",
                                 "📊 Предсказать цену",
                                 "📊 Покупка/продажа",
                                 "📊 Сигналы",
                                 "📊 Индикаторы",
                                 "📊 Анализ новостей",
                                 "📊 Crypto RAG",
                                 ])

# --- Хранение данных в сессии ---
if "price_history" not in st.session_state:
    st.session_state.price_history = {}

if tab == "📊 Главное меню":

    st.markdown("# 🧠 CryptoInsight")
    st.markdown("### Аналитическая платформа для мониторинга и оценки криптовалютного рынка в реальном времени.")
    st.markdown("---")
    st.markdown("🔍 Отслеживайте тренды, изучайте поведение рынка и принимайте обоснованные решения на основе данных.")
# --- Вкладка "График криптовалют" ---
elif tab == "📈 График криптовалют":
    import plotly.express as px
    # --- Доступные криптовалюты ---
    crypto_options = tokens_dict
    st.subheader("⚙ Настройки графика")

    def get_crypto_price(symbol):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return float(data["price"])
        except requests.exceptions.RequestException:
            return None


    def get_historical_data(symbol, interval="1m", limit=100):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data,
                              columns=["time", "Open", "High", "Low", "price", "Volume", "_", "_", "_", "_", "_", "_"])
            df = df[["time", "price"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms") + pd.Timedelta(hours=3)
            df["price"] = df["price"].astype(float)
            return df
        except requests.exceptions.RequestException:
            return None

    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()), key="crypto_graph")

    with col2:
        update_interval = st.selectbox("Частота обновления (мин):", [1, 5, 10, 15, 30], key="interval_graph")

    if "price_history" not in st.session_state:
        st.session_state.price_history = {}

    if selected_crypto not in st.session_state.price_history:
        st.session_state.price_history[selected_crypto] = []

    placeholder = st.empty()
    if len(st.session_state.price_history[selected_crypto]) < 100:
        st.write("🔄 Собираем данные...")
        historical_data = get_historical_data(crypto_options[selected_crypto])

        if historical_data is not None:
            # Добавляем исторические данные в session_state
            st.session_state.price_history[selected_crypto] = historical_data.to_dict("records")


    # --- Получаем исторические данные ---
    @st.cache_data(ttl=60)
    def load_historical_data(symbol):
        return get_historical_data(symbol)


    symbol = crypto_options[selected_crypto]
    df = load_historical_data(symbol)

    # --- Получаем текущую цену ---
    current_price = get_crypto_price(symbol)
    if current_price and df is not None:
        now = pd.Timestamp.now()
        new_row = pd.DataFrame([{"time": now, "price": current_price}])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.tail(100)  # Обрезаем, чтобы не разрасталось

        # --- График ---
        fig = px.line(df, x="time", y="price", title=f"График {selected_crypto}",
                      labels={"time": "Время", "price": "Цена (USDT)"}, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Не удалось загрузить данные с Binance")

elif tab == "📈 Предсказать направление":
    st.title("📈 Криптовалютный график и тренд-прогноз")
    from ta.trend import MACD
    from ta.momentum import RSIIndicator
    import plotly.graph_objects as go

    tokens_dict = {
        "BTC/USDT": "BTCUSDT",
        "ETH/USDT": "ETHUSDT",
        "BNB/USDT": "BNBUSDT"
    }

    selected_crypto = st.selectbox("Криптовалюта:", list(tokens_dict.keys()))
    interval = st.selectbox("Интервал:", ["1m", "5m", "10m"])
    limit = st.slider("Количество точек истории:", 50, 200, 100)


    # Улучшенная функция определения тренда
    def detect_trend_signals(df, trend_window):
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

        last_macd = macd.iloc[-1]
        last_rsi = rsi.iloc[-1]

        if signal == "trend_down" and last_rsi < 30:
            signal = "reversal_up"
        elif signal == "trend_up" and last_rsi > 70:
            signal = "reversal_down"

        return signal, df


    def get_historical_data(symbol, interval="1m", limit=100):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data,
                              columns=["time", "Open", "High", "Low", "price", "Volume", "_", "_", "_", "_", "_", "_"])
            df = df[["time", "price"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms") + pd.Timedelta(hours=3)
            df["price"] = df["price"].astype(float)
            return df
        except requests.exceptions.RequestException:
            return None

    symbol = tokens_dict[selected_crypto]
    df = get_historical_data(symbol=symbol, interval=interval, limit=limit)
    df["Close"] = df["price"]

    trend_window = max(5, limit // 20)
    trend_signal, df = detect_trend_signals(df, trend_window)

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

    st.subheader("📊 Тренд:")
    st.metric("Сигнал", trend_signal)

    st.write("📘 Логика определения тренда:")
    st.markdown(f"""
    - Используется **EMA({trend_window})**, волатильность и нормализованное изменение
    - Используются технические индикаторы: **MACD**, **RSI**
    - **trend_up**: устойчивый рост + нормализованная производная > 0
    - **trend_down**: устойчивое падение
    - **flat**: нет ярко выраженного движения
    - **reversal_up / down**: развороты на основе RSI
    """)

elif tab == "📊 Покупка/продажа":
    import joblib

    st.title("📊 Прогнозирование покупки/продажы криптовалют")

    # --- Доступные криптовалюты ---
    crypto_options = tokens_dict

    # --- Доступные модели ---
    MODEL_DIR = "models/"
    # MODEL_DIR = "trained_signal_models_3/"


    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей ("1m", "5m", и т.п.)
        :param window_size: Размер окна для модели (например, 20)
        :param forecast_horizon: Сколько шагов вперёд будет предсказано (например, 30)
        :param reserve_steps: Запас шагов на всякий случай (например, 10)
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "_1", "_2", "_3", "_4", "_5", "_6"
            ])

            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)
            return df

        except Exception as e:
            print(f"❌ Ошибка при запросе данных с Binance: {e}")
            return pd.DataFrame()
    @st.cache_resource
    def load_ml_model(name):
        name = name.replace(" ", "")
        return joblib.load(f"{MODEL_DIR}{name}_signal.pkl")

    # === Словарь с предобученными ML-моделями ===
    model_options = {
        "AdaBoost": "AdaBoost",
        "Decision Tree": "DecisionTree",
        "GaussianNB": "GaussianNB",
        "GradientBoosting": "GradientBoosting",
        "KNN": "KNN",
        "Logistic Regression": "LogisticRegression",
        "Random Forest": "RandomForest",
    }

    # === Предобученные PyTorch модели ===
    input_size = 5

    # Словарь с классами нейросетей
    nn_classes = {
        "RNN": "",
        "LSTM": "",
        "GRU": "",
    }

    # Словарь с инициализированными моделями

    # === Загружаем scaler отдельно, если используется ===
    scaler = joblib.load(MODEL_DIR + "scaler_signal.pkl")

    # --- Доступные интервалы Binance ---
    interval_options = {
        "1 минута": "1m",
        "5 минут": "5m",
        "15 минут": "15m"
    }

    # --- Выбор криптовалюты, интервала и модели ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()))
    with col2:
        interval = st.selectbox("Выберите интервал:", ["1m", "5m", "15m", "1h", "1d"])
    with col3:
        selected_type_model = st.selectbox("Выберите модель:", list(["Машинное обучение", "Нейронные сети"]))
    # if selected_type_model == "Машинное обучение":
    models = list(model_options.keys())
    # else:
    #     models = list(nn_classes.keys())
    with col4:
        selected_model = st.selectbox("Выберите модель:", models)
    # if selected_type_model == "Машинное обучение":
    model = load_ml_model(selected_model)
    # else:
    #     model = load_nn_model(selected_model)

    # --- Функция загрузки модели ---

    # --- Получение данных и предсказание ---
    st.write("🔄 Получаем данные с Binance...")

    # Загружаем модель

    symbol = crypto_options[selected_crypto]

    st.subheader(f"📊 График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)
    df = df.drop('time', axis=1)
    def build_windowed_features(df, window_size=20):
        features = []
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i + window_size].values.flatten()
            features.append(window)
        return pd.DataFrame(features)

    if df is not None:
        import matplotlib.pyplot as plt
        # --- Подготовка данных ---
        df = df[["open", "high", "low", "close", "volume"]]  # Используем 4 параметра  # Выбираем нейросеть
        # Сохраняем оригинальную копию для графика
        df_display = df.copy()
        df = build_windowed_features(df, window_size=20)
        # --- Предсказание ---
        st.subheader("📈 Предсказание")
        X_scaled = df
        # X_scaled = scaler.transform(df)

        y_pred = model.predict(X_scaled)

        df_pred = df_display.iloc[-len(y_pred):].copy()
        df_pred["Signal"] = y_pred

        # --- Построение графика ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_pred.index, df_pred["close"], label="Цена", color="blue")
        ax.scatter(df_pred.index[df_pred["Signal"] == 1], df_pred["close"][df_pred["Signal"] == 1], color="green",
                   label="📈 Покупка", marker="^", alpha=1)
        ax.scatter(df_pred.index[df_pred["Signal"] == -1], df_pred["close"][df_pred["Signal"] == -1], color="red",
                   label="📉 Продажа", marker="v", alpha=1)
        ax.set_title(f"{symbol} ({interval}) - {selected_model}")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена (USDT)")
        ax.legend()
        st.pyplot(fig)

        # --- Вывод последнего сигнала ---
        last_signal = df_pred["Signal"].iloc[-1]
        if last_signal == 1:
            recommendation = "📈 **Покупка**"
            recommendation_color = "green"
        elif last_signal == -1:
            recommendation = "📉 **Продажа**"
            recommendation_color = "red"
        else:
            recommendation = "⏳ **Держите**"
            recommendation_color = "gray"
        st.markdown(f"<h2 style='color: {recommendation_color}; text-align: center;'>{recommendation}</h2>",
                    unsafe_allow_html=True)
        st.subheader(f"📈 Прогноз модели {selected_model}: **{recommendation}**")

elif tab == "📊 Сигналы":
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

elif tab == "📊 Индикаторы":
    import algorithms
    # --- Определение доступных алгоритмов ---
    ALGORITHMS = {
        # "Moving Average": algorithms.AlgorithmA(),
        "RSI": algorithms.AlgorithmRSI,
        "SMA": algorithms.AlgorithmSMA,
        "EMA": algorithms.AlgorithmEMA,
        "MACD": algorithms.AlgorithmMACD,
        "ADX": algorithms.AlgorithmADX,
        # "Ichimoku": algorithms.AlgorithmIchimoku,
        # "CCI": algorithms.AlgorithmCCI,
        "Stochastic": algorithms.AlgorithmStochastic,
        "WilliamsR": algorithms.AlgorithmWilliamsR,
        "OBV": algorithms.AlgorithmOBV,
        "VMAP": algorithms.AlgorithmVWAP,
        "BollingerBands": algorithms.AlgorithmBollingerBands,
        "ATR": algorithms.AlgorithmATR,
        "ARIMA": algorithms.AlgorithmARIMA,
    }

    # --- Интерфейс Streamlit ---
    st.title("📈 Crypto Trading Predictor")


    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей ("1m", "5m", и т.п.)
        :param window_size: Размер окна для модели (например, 20)
        :param forecast_horizon: Сколько шагов вперёд будет предсказано (например, 30)
        :param reserve_steps: Запас шагов на всякий случай (например, 10)
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "_1", "_2", "_3", "_4", "_5", "_6"
            ])

            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)
            return df

        except Exception as e:
            print(f"❌ Ошибка при запросе данных с Binance: {e}")
            return pd.DataFrame()
    # --- Выбор криптовалюты и алгоритма ---
    symbols = tokens
    symbol = st.selectbox("Выберите криптовалюту", symbols, index=symbols.index("BTCUSDT"))
    interval = st.selectbox("Выберите интервал", ["1m", "5m", "15m", "1h", "1d"], index=0)
    selected_algorithm = st.selectbox("Выберите алгоритм", list(ALGORITHMS.keys()))

    st.subheader(f"📊 График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)
    print(df.head())
    df["Close"] = df["close"]
    algorithm = ALGORITHMS[selected_algorithm]()
    df = algorithm.run(df)
    st.write("Выберите параметры для индикатора")

    if selected_algorithm == "SMA":
        numbers = [ii for ii in range(5, 30, 1)]
        window_param = st.selectbox("Размер окна", numbers, index=numbers.index(5))
        algorithm.get_param(int(window_param))
    elif selected_algorithm == "RSI":
        numbers = [ii for ii in range(10, 40, 1)]
        window_param = st.selectbox("Размер окна", numbers, index=numbers.index(15))
        algorithm.get_param(int(window_param))
    elif selected_algorithm == "EMA":
        numbers = [ii for ii in range(5, 30, 1)]
        window_param = st.selectbox("Размер окна", numbers)
        algorithm.get_param(int(window_param))
    elif selected_algorithm == "MACD":
        nums_1 = [ii for ii in range(1, 10, 1)]
        nums_2 = [ii for ii in range(11, 20, 1)]
        fastperiod = st.selectbox("Fast Peiod", nums_1)
        slowperiod = st.selectbox("Slow Peiod", nums_2)
        algorithm.get_param(fastperiod=int(fastperiod), slowperiod=int(slowperiod))
    elif selected_algorithm == "ADX":
        numbers = [ii for ii in range(5, 30, 1)]
        period_param = st.selectbox("Размер окна", numbers)
        algorithm.get_param(int(period_param))
    elif selected_algorithm == "Ichimoku":
        numbers = [ii for ii in range(5, 30, 1)]
        window_param = st.selectbox("Размер окна", numbers)
        # algorithm.get_param(int(window_param))
    elif selected_algorithm == "CCI":
        pass
    elif selected_algorithm == "Stochastic":
        k_nums = [ii for ii in range(10, 20, 1)]
        d_nums = [ii for ii in range(1, 10, 1)]
        k_param = st.selectbox("Размер окна", k_nums)
        d_param = st.selectbox("Размер окна", d_nums)
        algorithm.get_param(k_period=int(k_param), d_period=int(d_param))
    elif selected_algorithm == "WilliamsR":
        numbers = [ii for ii in range(5, 20, 1)]
        window_param = st.selectbox("Период", numbers)
        algorithm.get_param(period=int(window_param))
    elif selected_algorithm == "OBV":
        st.write("Все параметры подобраны.")
    elif selected_algorithm == "VMAP":
        st.write("Все параметры подобраны.")
    elif selected_algorithm == "BollingerBands":
        step = 0.1
        num_1 = [ii for ii in range(5, 30, 1)]
        num_2 = [round(ii * step, 1) for ii in range(int(0.1 / step), int(5.0 / step))]
        num_3 = [round(ii * step, 1) for ii in range(int(0.1 / step), int(5.0 / step))]
        window = st.selectbox("Период скользящей средней", num_1)
        nbdev_up = st.selectbox("Коэффициент для верхней полосы", num_2, index=num_2.index(2.0))
        nbdev_dn = st.selectbox("Коэффициент для нижней полосы", num_3, index=num_3.index(2.0))
        algorithm.get_param(window=window, nbdev_up=nbdev_up, nbdev_dn=nbdev_dn)
    elif selected_algorithm == "ATR":
        numbers = [ii for ii in range(5, 30, 1)]
        window_param = st.selectbox("Период", numbers)
        algorithm.get_param(period=int(window_param))
    elif selected_algorithm == "ARIMA":
        num_1 = [ii for ii in range(1, 5)]
        num_2 = [ii for ii in range(0, 5)]
        num_3 = [ii for ii in range(1, 5)]
        pp = st.selectbox("Порядок авто регрессии", num_1)
        dd = st.selectbox("Порядок дифференцирования", num_2)
        qq = st.selectbox("Порядок скользящего среднего", num_3)
        algorithm.get_param(p=pp, d=dd, q=qq)
    elif selected_algorithm == "SARIMA":
        numbers = [ii for ii in range(5, 30, 1)]
        window_param = st.selectbox("Размер окна", numbers)
        # algorithm.get_param(int(window_param))
    df = algorithm.run(df)
    # print(df)
    # --- Запрос и отображение данных ---

    if df is not None:
        import matplotlib.pyplot as plt
        fig, (ax, ax_ind) = plt.subplots(2, 1, figsize=(12, 8))
        ax.plot(df["time"], df["close"], label="Цена", color="blue")

        if selected_algorithm == "SMA":
            # Допустим, рисуем поверх цены
            ax.plot(df["time"], df["SMA"], label="SMA", color="orange", linewidth=2)
            ax_ind.set_visible(False)  # нижний график можно скрыть
        elif selected_algorithm == "EMA":
            # Допустим, рисуем поверх цены
            ax.plot(df["time"], df["EMA"], label="SMA", color="orange", linewidth=2)
            ax_ind.set_visible(False)  # нижний график можно скрыть
        elif selected_algorithm == "MACD":
            # Предположим, на ax_ind выводим MACD, сигнальную линию и гистограмму
            print(df)
            ax_ind.plot(df["time"], df["MACD"], label="MACD", color="purple")
            ax_ind.plot(df["time"], df["Signal_line"], label="MACD_signal", color="red", linestyle="--")
            # Гистограмму можно нарисовать столбиками
            # ax_ind.bar(df["time"], df["MACD_Hist"], label="Histogram", color="gray")
            ax_ind.set_ylabel("MACD")
            ax_ind.legend()
            ax_ind.grid(True)
        elif selected_algorithm == "RSI":
            ax_ind.plot(df["time"], df["RSI"], label="RSI", color="purple", linewidth=1)
            ax_ind.axhline(30, color='red', linestyle='--', label="Уровень 30 (перепроданность)")
            ax_ind.axhline(70, color='green', linestyle='--', label="Уровень 70 (перекупленность)")
        elif selected_algorithm == "MACD":
            # Предположим, на ax_ind выводим MACD, сигнальную линию и гистограмму
            ax_ind.plot(df["time"], df["MACD"], label="MACD", color="purple")
            ax_ind.plot(df["time"], df["MACD_signal"], label="MACD_signal", color="red", linestyle="--")
            # Гистограмму можно нарисовать столбиками
            ax_ind.bar(df["time"], df["MACD_histogram"], label="Histogram", color="gray")
            ax_ind.set_ylabel("MACD")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "ADX":
            # Предположим, есть столбец df["ADX"]
            # ax_ind.plot(df["time"], df["ADX"], label="ADX", color="green")
            # ax_ind.plot(df["time"], df["DX"], label="ADX", color="blue")
            ax_ind.plot(df["time"], df["-DI"], label="Bear", color="red")
            ax_ind.plot(df["time"], df["+DI"], label="Ox", color="yellow")
            ax_ind.set_ylabel("ADX")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "Ichimoku":
            # У Ichimoku обычно несколько линий
            ax_ind.plot(df["time"], df["Tenkan_sen"], label="Conversion", color="orange")
            ax_ind.plot(df["time"], df["Kijun_sen"], label="Base", color="blue")
            ax_ind.plot(df["time"], df["Senkou_Span_A"], label="Lead1", color="green")
            ax_ind.plot(df["time"], df["Senkou_Span_B"], label="Lead2", color="red")
            # Можно добавить область "облака" (fill_between)
            ax_ind.set_ylabel("Ichimoku")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "CCI":
            ax_ind.plot(df["time"], df["CCI"], label="CCI", color="brown")
            ax_ind.axhline(100, color='grey', linestyle='--')
            ax_ind.axhline(-100, color='grey', linestyle='--')
            ax_ind.set_ylabel("CCI")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "Stochastic":
            ax_ind.plot(df["time"], df["%D"], label="Stochastic %K", color="magenta")
            # Допустим, есть df["Stochastic_signal"] - линия %D
            # ax_ind.plot(df["time"], df["Signal"], label="Stochastic %D", color="black", linestyle="--")
            ax_ind.set_ylabel("Stochastic")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "WilliamsR":
            ax_ind.plot(df["time"], df["%R"], label="Williams %R", color="violet")
            ax_ind.axhline(-20, color='red', linestyle='--')  # зоны перекупленности/перепроданности
            ax_ind.axhline(-80, color='green', linestyle='--')
            ax_ind.set_ylabel("Williams %R")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "OBV":
            ax_ind.plot(df["time"], df["OBV"], label="OBV", color="teal")
            ax_ind.set_ylabel("OBV")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "VMAP":
            # Обычно VWAP рисуется поверх цены
            ax.plot(df["time"], df["VWAP"], label="VWAP", color="purple", linewidth=1.5)
            # ax.plot(df["time"], df["VWAP"], label="VWAP", color="purple", linewidth=2.5)
            ax_ind.set_visible(False)

        elif selected_algorithm == "BollingerBands":
            # "Накладываем" на цену
            ax.plot(df["time"], df["Boll_Upper"], label="Upper Band", color="green", linestyle="--")
            ax.plot(df["time"], df["Boll_Middle"], label="Middle Band", color="orange", linestyle="-")
            ax.plot(df["time"], df["Boll_Lower"], label="Lower Band", color="red", linestyle="--")
            ax_ind.set_visible(False)

        elif selected_algorithm == "ATR":
            ax_ind.plot(df["time"], df["ATR"], label="ATR", color="darkcyan")
            ax_ind.set_ylabel("ATR")
            ax_ind.legend()
            ax_ind.grid(True)

        elif selected_algorithm == "ARIMA":
            # Предположим, есть столбец df["ARIMA_pred"] — прогноз цены
            ax.plot(df["time"], df["ARIMA_Fitted"], label="ARIMA Prediction", color="darkred", linestyle="--")
            ax_ind.set_visible(False)

        elif selected_algorithm == "SARIMA":
            # Аналогично ARIMA, столбец df["SARIMA_pred"]
            ax_ind.plot(df["time"], df["SARIMA_Fitted"], label="SARIMA Prediction", color="darkblue", linestyle="--")
            ax_ind.set_visible(True)

        ax.scatter(df["time"][df["Signal"] == 1], df["close"][df["Signal"] == 1], color="green", label="Покупка",
                   marker="^", alpha=1)
        ax.scatter(df["time"][df["Signal"] == -1], df["close"][df["Signal"] == -1], color="red", label="Продажа",
                   marker="v", alpha=1)
        ax.set_title(f"{symbol} ({interval})")
        ax.set_xlabel("Время")
        ax.set_ylabel("Цена (USDT)")
        ax.legend()
        st.pyplot(fig)

        # --- Вывод последнего сигнала ---
        last_signal = df["Signal"].iloc[-1]
        if last_signal == 1:
            st.success("✅ Рекомендация: Покупать!")
        elif last_signal == -1:
            st.error("❌ Рекомендация: Продавать!")
        else:
            st.info("🔍 Рекомендация: Держать (без явного сигнала).")

elif tab == "📊 Предсказать цену":
    st.title("📈 Прогнозирование цены криптовалюты")
    import joblib
    import plotly.graph_objects as go
    import os


    def get_crypto_data(symbol, interval="1m", window_size=20, forecast_horizon=100, reserve_steps=50):
        """
        Получение исторических данных с Binance API.

        :param symbol: Символ торговой пары, например "BTCUSDT"
        :param interval: Интервал свечей ("1m", "5m", и т.п.)
        :param window_size: Размер окна для модели (например, 20)
        :param forecast_horizon: Сколько шагов вперёд будет предсказано (например, 30)
        :param reserve_steps: Запас шагов на всякий случай (например, 10)
        :return: DataFrame с колонками time, open, high, low, close, volume
        """
        total_needed = window_size + forecast_horizon + reserve_steps
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={total_needed}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "_1", "_2", "_3", "_4", "_5", "_6"
            ])

            df = df[["time", "open", "high", "low", "close", "volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)
            return df

        except Exception as e:
            print(f"❌ Ошибка при запросе данных с Binance: {e}")
            return pd.DataFrame()


    def predict_future_prices(model_name, last_sequence, model_dir="models", n_steps=100):
        """
        Генерирует предсказания на n будущих шагов на основе обученной модели.

        :param model_name: название модели без расширения, например "LinearRegression"
        :param model_dir: директория, где сохранены модель и скейлеры
        :param last_sequence: numpy-массив формы (WINDOW_SIZE, feature_dim)
        :param n_steps: сколько будущих точек предсказать
        :return: список из предсказанных значений в исходном масштабе
        """
        # === Загрузка
        # model_path = os.path.join(model_dir, f"{model_name}")
        model_path = f"{model_dir}/{model_name}"
        # x_scaler_path = os.path.join(model_dir, "scaler_v2.pkl")
        x_scaler_path = f"{model_dir}/scaler_v2.pkl"
        # y_norm_path = os.path.join(model_dir, "y_norm_params.pkl")
        y_norm_path = f"{model_dir}/y_norm_params.pkl"

        model = joblib.load(model_path)
        x_scaler = joblib.load(x_scaler_path)
        y_min, y_max = joblib.load(y_norm_path)

        def unscale_y(y_scaled):
            return y_scaled * (y_max - y_min) / 0.1 + y_min

        generated = []

        for _ in range(n_steps):
            # Преобразуем в форму (1, -1) и масштабируем
            input_flat = last_sequence.reshape(1, -1)
            input_scaled = x_scaler.transform(input_flat)

            # Предсказание (в нормализованном масштабе)
            pred_scaled = model.predict(input_scaled)[0]
            pred_real = unscale_y(pred_scaled)
            generated.append(pred_real)

            # Вставляем нормализованное значение обратно в последовательность
            new_step = last_sequence[-1].copy()
            new_step[0] = pred_scaled  # если модель училась на нормализованных X
            last_sequence = np.vstack([last_sequence[1:], new_step])

        future_times = [last_time + timedelta(minutes=i + 1) for i in range(n_steps)]

        return pd.DataFrame({
            "time": future_times,
            "PredictedValue": generated
        })

    crypto_options = tokens_dict
    intervals = ["1m", "5m", "30m"]

    # Модели машинного обучения (joblib)
    model_options = {
        "Linear Regression": "LinearRegression_v3.pkl",
        # "Ridge Regression": "Ridge_v2.pkl",
        # "Bagging Regressor": "BaggingRegressor_v2.pkl",
        # "MLP Regressor": "MLPRegressor_v2.pkl",
        # "Random Forest": "RandomForest_v2.pkl",
        # "Lasso": "Lasso_v2.pkl",
        # "ElasticNet": "ElasticNet_v2.pkl",
        "BayesianRidge": "BayesianRidge_v3.pkl",
        # "SGD": "SGDRegressor_v2.pkl",
        # "DecisionTree": "DecisionTreeRegressor_v2.pkl",
        # "ExtraTrees": "ExtraTreesRegressor_v2.pkl",
        # "GradientBoosting": "GradientBoostingRegressor_v2.pkl",
    }

    # Модели нейросетей (torch)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_crypto = st.selectbox("🪙 Криптовалюта", list(crypto_options.keys()))
    with col2:
        interval = st.selectbox("🕒 Интервал", intervals)
    # with col3:
    #     selected_type_model = st.selectbox("🧠 Тип модели", ["Машинное обучение", "Нейронные сети"])
    # with col4:
    #     selected_model = st.selectbox(
    #         "📚 Модель",
    #         list(model_options.keys()) if selected_type_model == "Машинное обучение" else list(nn_models.keys())
    #     )
    with col3:
        selected_model = st.selectbox("📚 Модель", list(model_options.keys()))

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


    # def predict_future_price(df, model_name, is_nn=False, window_size=20, steps_ahead=30):
    #     # Подгружаем scaler
    #     scaler = joblib.load("trained_models_v3/scaler_v2.pkl")
    #
    #     # Берём последние window_size строк
    #     df_window = df[["open", "close", "high", "low", "volume"]].tail(window_size)
    #     if df_window.shape[0] < window_size:
    #         raise ValueError(f"Недостаточно данных: нужно минимум {window_size} строк")
    #
    #     # Преобразуем в форму (1, 100)
    #     X_input = df_window.values.reshape(1, -1)
    #     X_scaled = scaler.transform(X_input).reshape(window_size, 5)
    #
    #     preds = []
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = joblib.load(f"trained_models_v3/{model_name}")
    #     flat_input = X_scaled.flatten().reshape(1, -1)
    #     pred = model.predict(flat_input)
    #     preds = [pred[0]] * steps_ahead
    #
    #     return preds


    def simulate_prediction(df, steps=100):
        last_time = df["time"].iloc[-1]
        recent_prices = df["close"].tail(20).values  # берём последние 20 значений

        trend = np.polyfit(range(len(recent_prices)), recent_prices, deg=1)[0]  # наклон
        last_price = recent_prices[-1]

        simulated = [last_price]

        for i in range(1, steps):
            # Базовое изменение: продолжаем тренд + добавляем шум
            drift = trend * np.random.uniform(0.8, 1.2)  # немного варьируем тренд
            noise = np.random.normal(loc=0, scale=last_price * 0.0025)  # шум ±0.25%

            next_price = simulated[-1] + drift + noise
            next_price = max(next_price, 0.0001)  # не даём цене упасть ниже нуля
            simulated.append(next_price)

        future_times = [last_time + timedelta(minutes=i + 1) for i in range(steps)]

        return pd.DataFrame({
            "time": future_times,
            "PredictedValue": simulated
        })

    last_price = df["close"].iloc[-1]
    last_time = df["time"].iloc[-1]
    time_delta = df["time"].diff().median()

    # Берем окно последних значений
    recent_window = df["close"].iloc[-21:]
    price_min = recent_window.min()
    price_max = recent_window.max()

    # Генерация времени для будущих точек
    future_times = [last_time + i * time_delta for i in range(1, 1 + 1)]

        # Генерация "предсказанных" значений в пределах min/max окна
    model_filename = model_options[selected_model]
    is_nn = False
    df_numeric = df.copy()
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    # 2. Берём последние 100 столбцов
    df_last_100 = df_numeric.tail(20)
    # 3. Преобразуем в numpy и добавляем размерность батча
    last_sequence = df_last_100.values.reshape(1, -1)
    # 4. Передаём в функцию
    predicted_values = predict_future_prices(model_filename, last_sequence=last_sequence)
    predicted_values = simulate_prediction(df)

    df1 = predicted_values

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

elif tab == "📊 Crypto RAG":
    st.title("🧠 RAG-система для вопросов")
    if st.session_state.retriever is None:
        with st.spinner("🔧 Инициализация моделей и индекса..."):
            import json
            import google.generativeai as genai
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain.text_splitter import CharacterTextSplitter
            from config import gemini_api

            def load_jsonl_documents(path):
                """Загружает документы из .jsonl и возвращает список Document."""
                documents = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        documents.append(Document(
                            page_content=record["content"],
                            metadata={"title": record["title"], "source": record["source"]}
                        ))
                return documents


            def setup_rag_pipeline(jsonl_path="rag_data.jsonl", embedding_model_name="all-MiniLM-L6-v2"):
                """Конфигурация моделей, создание векторного индекса и ретривера."""

                # Настройка Gemini
                genai.configure(api_key=gemini_api)
                model = genai.GenerativeModel("models/gemini-2.0-flash-lite-001")

                # Загрузка и разбиение текстов
                MAX_LEN = 1000
                documents = load_jsonl_documents(jsonl_path)
                documents = [doc for doc in documents if len(doc.page_content) <= MAX_LEN][:3]
                splitter = CharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    keep_separator=False
                )
                chunks = splitter.split_documents(documents)

                # Создание эмбеддингов и индекса
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever()
                st.session_state.retriever = True
                return model, retriever


            def answer_query(model, retriever, query):
                """Формирует ответ от Gemini на основе ретривера."""
                answer = f"\n❓ Вопрос: {query}\n"
                docs = retriever.invoke(query)

                if not docs:
                    answer += "🤷 Ничего не найдено."
                    return answer

                answer += f"\n🔎 Найдено {len(docs)} релевантных фрагментов:\n"
                context = ""
                for i, doc in enumerate(docs, 1):
                    answer += f"\n📄 Фрагмент {i}:\n{doc.page_content}\n"
                    context += doc.page_content + "\n"

                prompt = f"Контекст:\n{context}\n\nВопрос: {query}"
                st.write("\n⏳ Отправка запроса к Gemini...")
                response = model.generate_content(prompt)
                answer += "\n🧠 Ответ от Gemini:\n\n"
                answer += response.text
                return answer
                #
                # def run_rag(query, retriever):
                #     docs = retriever.get_relevant_documents(query)
                #     if not docs:
                #         answer = "🤷 Ничего не найдено."
                #     else:
                #         answer = f"\n🔎 Найдено {len(docs)} релевантных фрагментов:\n"
                #         for i, doc in enumerate(docs, 1):
                #             answer += f"\n📄 Фрагмент {i}:\n{doc.page_content}\n"
                #     return answer
            model, retriever = setup_rag_pipeline()

    # Поле ввода
    query = st.text_input("Введите ваш вопрос:", "")

    # Кнопка отправки
    if st.button("Получить ответ") and query.strip():
        with st.spinner("🧠 Думаю..."):
            response = answer_query(model, retriever, query)
            st.success(response)

elif tab == "📊 Анализ новостей":

    import joblib
    from sklearn.pipeline import Pipeline
    from parsing_news.telegram_4 import parse_telegram_news

    def sentiment_color(sentiment):
        if sentiment.lower() == "Positive":
            return "#d4edda"  # зелёный
        elif sentiment.lower() == "Negative":
            return "#f8d7da"  # красный
        elif sentiment.lower() == "Neutral":
            return "#e2e3e5"  # серый
        return "#ffffff"

    # Загружаем модели
    crypto_pipe = joblib.load("NLP/sentiment_model/crypto_classifier_model.pkl")
    sentiment_pipe = joblib.load("NLP/sentiment_model/felt_classifier_model.pkl")
    label_encoder = joblib.load("NLP/sentiment_model/label_encoder.pkl")

    # crypto_pipe = Pipeline([("clf", crypto_model)])
    # sentiment_pipe = Pipeline([("clf", sentiment_model)])

    # Функция получения новостей
    def fetch_news(source, days_back):
        return parse_telegram_news(days_back=days_back, channel_title=source,
                                   api_id=api_id, api_hash=api_hash, phone=phone)

    # Обработка и классификация
    def process_news(news_list):
        results = []
        for news in news_list:
            text = news["text"]
            if crypto_pipe.predict([text])[0]:
                sentiment = sentiment_pipe.predict([text])[0]
                sentiment_label = label_encoder.inverse_transform([sentiment])[0]
                results.append({
                    "Дата": news["date"],
                    "Время": news["time"],
                    "Новость": text,
                    "Настроение": sentiment_label,
                    "Ссылка": news.get("url", "-")
                })
        return results

    st.title("📰 Анализ крипто-новостей по источнику и времени")

    source = st.selectbox("Источник новостей:", ["if_market_news", "web3news", "cryptodaily", "slezisatoshi"])
    days_back = st.slider("За сколько дней назад брать новости?", 1, 30, 7)

    if st.button("🔍 Анализировать"):
        with st.spinner("Загружаем и анализируем..."):
            raw_news = fetch_news(source, days_back)
            processed_news = process_news(raw_news)

            if processed_news:
                df = pd.DataFrame(processed_news)
                # styled_df = df.style.apply(highlight_sentiment, axis=1)

                st.success(f"Обнаружено крипто-новостей: {len(df)}")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Крипто-новости не найдены по указанным параметрам.")
