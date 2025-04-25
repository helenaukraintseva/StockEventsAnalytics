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

# --- Настройка страницы ---
st.set_page_config(page_title="Крипто-график и статистика", layout="wide")

st.title("📊 Криптовалютный мониторинг")

# --- Боковое меню с вкладками ---
tab = st.sidebar.radio("Меню:", ["📈 График криптовалют",
                                 # "📈 Статистика",
                                 "📈 Предсказать направление",
                                 "📊 Предсказать цену",
                                 "📊 Предсказать покупку/продажу",
                                 "📊 Индикаторы"])


# --- Функция получения данных с Binance ---
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

        df = pd.DataFrame(data, columns=["time", "Open", "High", "Low", "price", "Volume", "_", "_", "_", "_", "_", "_"])
        df = df[["time", "price"]]
        df["time"] = pd.to_datetime(df["time"], unit="ms") + pd.Timedelta(hours=3)
        df["price"] = df["price"].astype(float)
        return df
    except requests.exceptions.RequestException:
        return None


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


# --- Получение списка всех криптовалют с Binance ---
# def get_all_binance_symbols():
#     url = "https://api.binance.com/api/v3/exchangeInfo"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         return [symbol["symbol"] for symbol in data["symbols"] if symbol["symbol"].endswith("USDT")]
#     except requests.exceptions.RequestException as e:
#         st.error(f"Ошибка при получении списка монет: {e}")
#         return []


# --- Хранение данных в сессии ---
if "price_history" not in st.session_state:
    st.session_state.price_history = {}

# --- Вкладка "График криптовалют" ---
if tab == "📈 График криптовалют":
    # --- Доступные криптовалюты ---
    crypto_options = tokens_dict
    st.subheader("⚙ Настройки графика")

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

    while True:
        price = get_crypto_price(crypto_options[selected_crypto])
        st.write()

        if price:
            # Добавляем новую цену в историю
            st.session_state.price_history[selected_crypto].append({"time": pd.Timestamp.now(), "price": price})

            # Оставляем последние 100 записей
            st.session_state.price_history[selected_crypto] = st.session_state.price_history[selected_crypto][-100:]

            # Создаем DataFrame
            df = pd.DataFrame(st.session_state.price_history[selected_crypto])

            # --- График ---
            fig = px.line(df, x="time", y="price", title=f"График {selected_crypto}",
                          labels={"time": "Время", "price": "Цена (USDT)"}, template="plotly_dark")
            # placeholder.plotly_chart(fig, use_container_width=True)

            # Обновляем только график
            with placeholder:
                placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(int(update_interval * 60))

# --- Вкладка "Статистика" ---
# elif tab == "📊 Статистика":
#     crypto_options = tokens_dict
#     st.subheader("⚙ Настройки статистики")
#
#     selected_crypto_stat = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()), key="crypto_stat")
#
#     st.subheader(f"📊 Последние 10 цен {selected_crypto_stat}")
#
#     if selected_crypto_stat in st.session_state.price_history and st.session_state.price_history[selected_crypto_stat]:
#         df = pd.DataFrame(st.session_state.price_history[selected_crypto_stat]).tail(10)
#         st.dataframe(df)
#     else:
#         st.warning("Данных пока нет. Переключитесь на вкладку 'График криптовалют', чтобы они появились.")

elif tab == "📈 Предсказать направление":
    st.title("Криптовалютный график и тренд-прогноз")

    # Список криптовалют
    tokens_dict = tokens_dict

    selected_crypto = st.selectbox("Криптовалюта:", list(tokens_dict.keys()))
    interval = st.selectbox("Интервал:", ["1m", "5m", "10m"])
    limit = st.slider("Количество точек истории:", 50, 200, 100)


    # Получаем (или генерируем) данные
    symbol = tokens_dict[selected_crypto]
    df = get_historical_data(symbol=symbol, interval=interval, limit=limit)
    df["Close"] = df["price"]

    # Отрисовываем график
    fig = px.line(df, x="time", y="Close", title=f"{selected_crypto} ({interval})", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Детектируем тренд
    trend = limit // 20
    detector = TrendDetector(period=trend, target_col="price")
    last_signal = detector.detect_last_signal(df)
    st.subheader("Определение тренда и разворота")
    st.write(f"Текущий сигнал: **{last_signal}**")

    st.write("Описание логики:")
    st.markdown(f"""
        - **trend_up**: последние несколько ({trend}) приращений цены положительные  
        - **trend_down**: последние несколько ({trend}) приращений цены отрицательные  
        - **reversal_up**: был тренд down, теперь up  
        - **reversal_down**: был тренд up, теперь down  
        - **flat/none**: нет явного тренда или недостаточно данных  
        """)

elif tab == "📊 Предсказать покупку/продажу":

    st.title("📊 Прогнозирование покупки/продажы криптовалют")

    # --- Доступные криптовалюты ---
    crypto_options = tokens_dict

    # --- Доступные модели ---
    model_options = {
        "Linear Regression": ml_models.AlgorithmLinearRegression,
        "Ridge Regression": ml_models.AlgorithmRidge,
        "Random Forest Regressor": ml_models.AlgorithmRandomForestRegressor,
        "Lasso": ml_models.Lasso,
        "ElasticNet": ml_models.AlgorithmElasticNet,
        "BayesianRidge": ml_models.AlgorithmBayesianRidge,
        "SGD": ml_models.AlgorithmSGDRegressor,
        "DecisionTree": ml_models.AlgorithmDecisionTreeRegressor,
        "RandomForest": ml_models.AlgorithmRandomForestRegressor,
        "ExtraTrees": ml_models.AlgorithmExtraTreesRegressor,
        "GradientBoosting": ml_models.AlgorithmGradientBoostingRegressor
    }

    nn_models = {
        "RNN": nn_model_buy.AlgorithmRNN,
        "LSTM": nn_model_buy.AlgorithmLSTM,
        "GRU": nn_model_buy.AlgorithmGRU,
    }

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
    step_pred = st.selectbox("Выберите шаг предсказания:", [1, 5, 10], key="step_pred")
    window_pred = st.selectbox("Выберите окно предсказания:", [5, 10, 20], key="window_pred")
    if selected_type_model == "Машинное обучение":
        models = list(model_options.keys())
    else:
        models = list(nn_models.keys())
    with col4:
        selected_model = st.selectbox("Выберите модель:", models)


    # --- Функция загрузки модели ---

    # --- Получение данных и предсказание ---
    st.write("🔄 Получаем данные с Binance...")

    # Загружаем модель
    if selected_type_model == "Машинное обучение":
        model = model_options[selected_model]()
    else:
        model = nn_models[selected_model]()

    symbol = crypto_options[selected_crypto]

    st.subheader(f"📊 График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)

    if df is not None:
        # --- Подготовка данных ---
        df = df[["open", "high", "low", "close", "volume"]]  # Используем 4 параметра  # Выбираем нейросеть

        # --- Предсказание ---
        st.subheader("📈 Предсказание")
        # pred_data = df.drop(["volume"], axis=1)
        df_pred = model.predict(df)

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
    # ohlc_data = get_crypto_ohlc(symbol, interval)
    #
    # if ohlc_data:
    #     st.write(f"**Последние данные {selected_crypto} ({selected_interval}):**")
    #     st.write(pd.DataFrame([ohlc_data]))
    #
    #
    #
    #     if model:
    #         # Создаем DataFrame с OHLC данными
    #         X = pd.DataFrame(ohlc_data)
    #         y_pred = model.predict(X)
    #         if y_pred == 1:
    #             recommendation = "📈 **Покупка**"
    #             recommendation_color = "green"
    #         elif y_pred == 0:
    #             recommendation = "📉 **Продажа**"
    #             recommendation_color = "red"
    #         else:
    #             recommendation = "⏳ **Держите**"
    #             recommendation_color = "gray"
    #
    #         # with recommendation_container:
    #         st.markdown(f"<h2 style='color: {recommendation_color}; text-align: center;'>{recommendation}</h2>",
    #                     unsafe_allow_html=True)
    #
    #         # Вывод предсказанной цены
    #         st.subheader(f"📈 Прогноз модели {selected_model}: **{recommendation}**")
    # else:
    #     st.error("Ошибка при получении данных с Binance API. Попробуйте позже.")

elif tab == "📊 Индикаторы":
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
        # "SARIMA": algorithms.AlgorithmSARIMA
    }

    # --- Интерфейс Streamlit ---
    st.title("📈 Crypto Trading Predictor")

    # --- Выбор криптовалюты и алгоритма ---
    symbols = tokens
    symbol = st.selectbox("Выберите криптовалюту", symbols, index=symbols.index("BTCUSDT"))
    interval = st.selectbox("Выберите интервал", ["1m", "5m", "15m", "1h", "1d"], index=0)
    selected_algorithm = st.selectbox("Выберите алгоритм", list(ALGORITHMS.keys()))

    st.subheader(f"📊 График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)
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
        num_2 = [round(ii*step, 1) for ii in range(int(0.1/step), int(5.0/step))]
        num_3 = [round(ii*step, 1) for ii in range(int(0.1/step), int(5.0/step))]
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
        pp = st.selectbox("Порядок авторегрессии", num_1)
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
        # --- Применение алгоритма ---
        # --- Построение графика ---
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
    st.title("Криптовалютный график с предсказанием")

    # Пример списка криптовалют и интервалов
    crypto_options = tokens_dict
    intervals = ["1m", "5m", "10m"]

    model_options = {
        "Linear Regression": ml_model_price.AlgorithmLinearRegression,
        "Ridge Regression": ml_model_price.AlgorithmRidge,
        "Random Forest Regressor": ml_model_price.AlgorithmRandomForestRegressor,
        "Lasso": ml_model_price.Lasso,
        "ElasticNet": ml_model_price.AlgorithmElasticNet,
        "BayesianRidge": ml_model_price.AlgorithmBayesianRidge,
        "SGD": ml_model_price.AlgorithmSGDRegressor,
        "DecisionTree": ml_model_price.AlgorithmDecisionTreeRegressor,
        "RandomForest": ml_model_price.AlgorithmRandomForestRegressor,
        "ExtraTrees": ml_model_price.AlgorithmExtraTreesRegressor,
        "GradientBoosting": ml_model_price.AlgorithmGradientBoostingRegressor
    }

    nn_models = {
        "RNN": nn_model_price.AlgorithmRNN,
        "LSTM": nn_model_price.AlgorithmLSTM,
        "GRU": nn_model_price.AlgorithmGRU,
    }

    # Выбор
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_crypto = st.selectbox("Выберите криптовалюту:", list(crypto_options.keys()))
    with col2:
        interval = st.selectbox("Выберите интервал:", intervals)
    with col3:
        selected_type_model = st.selectbox("Выберите модель:", list(["Машинное обучение", "Нейронные сети"]))
    step_pred = st.selectbox("Выберите шаг предсказания:", [1, 5, 10], key="step_pred")
    window_pred = st.selectbox("Выберите окно предсказания:", [5, 10, 20], key="window_pred")
    if selected_type_model == "Машинное обучение":
        models = list(model_options.keys())
    else:
        models = list(nn_models.keys())
    with col4:
        selected_model = st.selectbox("Выберите модель:", models)

    st.write("🔄 Получаем данные с Binance...")

    # Загружаем модель
    if selected_type_model == "Машинное обучение":
        model = model_options[selected_model]()
    else:
        model = nn_models[selected_model]()

    symbol = crypto_options[selected_crypto]

    st.subheader(f"📊 График {symbol} ({interval})")
    df = get_crypto_data(symbol, interval)

    st.subheader(f"График {selected_crypto} ({interval}): предсказане с шагом {step_pred}")
    df1 = model.predict(data=df)

    # Построение графика на одном рисунке
    fig = go.Figure()

    st.subheader(f"Последнее предсказание: {round(df1.iloc[-1]['PredictedValue'], 4)}")

    # Фактическая цена (линия)
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["close"],
        mode="lines",
        name="Close (Фактическая)",
        line=dict(color="blue")
    ))

    # Предсказанная цена (линия)
    fig.add_trace(go.Scatter(
        x=df1["time"],
        y=df1["PredictedValue"],
        mode="lines",
        name="PredictedPrice",
        line=dict(color="red")
    ))

    fig.update_layout(
        title=f"{crypto_options[selected_crypto]} Price vs Predicted (interval: {interval})",
        xaxis_title="Время",
        yaxis_title="Цена (USDT)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)
    # NEURAL_NETWORKS = {
    #     "RNN": AlgorithmRNN(),
    #     "LSTM": AlgorithmLSTM(),
    #     "GRU": AlgorithmGRU(),
    # }
    # NEURAL_NETWORKS = {
    #     "RNN": nn_model_buy.AlgorithmRNN(),
    #     "LSTM": nn_model_buy.AlgorithmLSTM(),
    #     "GRU": nn_model_buy.AlgorithmGRU(),
    # }
    #
    # # --- Интерфейс Streamlit ---
    # st.title("🤖 AI Crypto Trading Predictor (Neural Networks)")
    #
    # # --- Выбор криптовалюты, интервала и модели ---
    # symbols = tokens
    # selected_symbol = st.selectbox("Выберите криптовалюту", symbols, index=symbols.index("BTCUSDT"))
    # selected_nn = st.selectbox("Выберите нейронную сеть", list(NEURAL_NETWORKS.keys()))
    # selected_interval = st.selectbox("Выберите интервал", ["1m", "5m", "15m", "1h", "1d"], index=0)
    #
    # # --- Запрос данных ---
    # st.subheader(f"📊 График {selected_symbol} ({selected_interval})")
    # df = get_crypto_data(selected_symbol, selected_interval)
    # print(df.head())
    #
    # if df is not None:
    #     # --- Подготовка данных ---
    #     df = df[["open", "high", "low", "close", "volume"]]  # Используем 4 параметра
    #     model = NEURAL_NETWORKS[selected_nn]  # Выбираем нейросеть
    #
    #     # --- Предсказание ---
    #     st.subheader("📈 Предсказание")
    #     # pred_data = df.drop(["volume"], axis=1)
    #     df_pred = model.predict(df)
    #
    #     # --- Построение графика ---
    #     fig, ax = plt.subplots(figsize=(12, 5))
    #     ax.plot(df_pred.index, df_pred["close"], label="Цена", color="blue")
    #     ax.scatter(df_pred.index[df_pred["Signal"] == 1], df_pred["close"][df_pred["Signal"] == 1], color="green",
    #                label="📈 Покупка", marker="^", alpha=1)
    #     ax.scatter(df_pred.index[df_pred["Signal"] == -1], df_pred["close"][df_pred["Signal"] == -1], color="red",
    #                label="📉 Продажа", marker="v", alpha=1)
    #     ax.set_title(f"{selected_symbol} ({selected_interval}) - {selected_nn}")
    #     ax.set_xlabel("Время")
    #     ax.set_ylabel("Цена (USDT)")
    #     ax.legend()
    #     st.pyplot(fig)
    #
    #     # --- Вывод последнего сигнала ---
    #     last_signal = df_pred["Signal"].iloc[-1]
    #     if last_signal == 1:
    #         st.success("✅ Рекомендация: Покупать!")
    #     elif last_signal == -1:
    #         st.error("❌ Рекомендация: Продавать!")
    #     else:
    #         st.info("🔍 Рекомендация: Держать (без явного сигнала).")
