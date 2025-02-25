import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class AlgorithmA:
    def __init__(self):
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['Signal'] = 0
        df.loc[df['Close'] > df['MA_5'], 'Signal'] = 1
        df.loc[df['Close'] < df['MA_5'], 'Signal'] = -1
        return df


class AlgorithmRSI:
    def __init__(self):
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['Delta'] = df['Close'].diff()
        df['Gain'] = df['Delta'].clip(lower=0).rolling(14).mean()
        df['Loss'] = -df['Delta'].clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - 100 / (1 + df['Gain']/(df['Loss']+1e-9))
        df['Signal'] = 0
        df.loc[df['RSI'] < 30, 'Signal'] = 1
        df.loc[df['RSI'] > 70, 'Signal'] = -1
        return df


class AlgorithmSMA:
    def __init__(self, window: int = 5):
        self.window = window

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df[f'SMA_{self.window}'] = df['Close'].rolling(self.window).mean()
        df['Signal'] = 0
        df.loc[df['Close'] > df[f'SMA_{self.window}'], 'Signal'] = 1
        df.loc[df['Close'] < df[f'SMA_{self.window}'], 'Signal'] = -1
        return df


class AlgorithmEMA:
    def __init__(self, window: int = 5):
        self.window = window

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df[f'EMA_{self.window}'] = df['Close'].ewm(span=self.window, adjust=False).mean()
        df['Signal'] = 0
        df.loc[df['Close'] > df[f'EMA_{self.window}'], 'Signal'] = 1
        df.loc[df['Close'] < df[f'EMA_{self.window}'], 'Signal'] = -1
        return df


class AlgorithmMACD:
    def __init__(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['EMA_fast'] = df['Close'].ewm(span=self.fastperiod, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=self.slowperiod, adjust=False).mean()
        df['MACD_line'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal_line'] = df['MACD_line'].ewm(span=self.signalperiod, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_line'] - df['Signal_line']
        df['Signal'] = 0
        df.loc[df['MACD_line'] > df['Signal_line'], 'Signal'] = 1
        df.loc[df['MACD_line'] < df['Signal_line'], 'Signal'] = -1
        return df


class AlgorithmADX:
    def __init__(self, period: int = 14):
        """
        Алгоритм на основе ADX (Average Directional Index).
        :param period: Период для расчёта ADX (по умолчанию 14).
        """
        self.period = period

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает ADX и формирует столбцы:
          - 'TR' (True Range),
          - '+DM' (Positive Directional Movement),
          - '-DM' (Negative Directional Movement),
          - '+DI' (Positive Directional Index),
          - '-DI' (Negative Directional Index),
          - 'DX' (Directional Index),
          - 'ADX'
        А также простой столбец 'Signal': 1, если +DI > -DI, -1 — иначе.

        :param data: DataFrame, содержащий столбцы 'High', 'Low', 'Close'.
        :return: Копия DataFrame с добавленными столбцами для ADX.
        """
        df = data.copy()
        df['PrevHigh'] = df['High'].shift(1)
        df['PrevLow'] = df['Low'].shift(1)
        df['PrevClose'] = df['Close'].shift(1)
        df['TR'] = df[['High', 'PrevClose']].max(axis=1) - df[['Low', 'PrevClose']].min(axis=1)
        df['+DM'] = np.where((df['High'] - df['PrevHigh']) > (df['PrevLow'] - df['Low']),
                             np.maximum(df['High'] - df['PrevHigh'], 0), 0)
        df['-DM'] = np.where((df['PrevLow'] - df['Low']) > (df['High'] - df['PrevHigh']),
                             np.maximum(df['PrevLow'] - df['Low'], 0), 0)
        df['TR_ema'] = df['TR'].ewm(alpha=1 / self.period, adjust=False).mean()
        df['+DM_ema'] = df['+DM'].ewm(alpha=1 / self.period, adjust=False).mean()
        df['-DM_ema'] = df['-DM'].ewm(alpha=1 / self.period, adjust=False).mean()
        df['+DI'] = 100 * (df['+DM_ema'] / df['TR_ema'])
        df['-DI'] = 100 * (df['-DM_ema'] / df['TR_ema'])
        df['DX'] = ((df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'])) * 100
        df['ADX'] = df['DX'].ewm(alpha=1 / self.period, adjust=False).mean()
        df['Signal'] = 0
        df.loc[df['+DI'] > df['-DI'], 'Signal'] = 1
        df.loc[df['+DI'] < df['-DI'], 'Signal'] = -1
        df.drop(['PrevHigh', 'PrevLow', 'PrevClose'], axis=1, inplace=True)
        return df


class AlgorithmIchimoku:
    def __init__(self,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_span_b_period: int = 52,
                 shift: int = 26):
        """
        Алгоритм расчёта Ichimoku Cloud.
        :param tenkan_period: Период для Tenkan-sen (Conversion Line).
        :param kijun_period: Период для Kijun-sen (Base Line).
        :param senkou_span_b_period: Период для Senkou Span B.
        :param shift: Сдвиг облака и Chikou Span (обычно 26).
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.shift = shift

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает линии Ichimoku:
          - Tenkan-sen (Conversion Line)
          - Kijun-sen (Base Line)
          - Senkou Span A
          - Senkou Span B
          - Chikou Span
        И простой сигнал: 1, если Close > Senkou Span A и Close > Senkou Span B, иначе -1 (упрощённо).
        :param data: DataFrame со столбцами 'High', 'Low', 'Close'.
        :return: Копия DataFrame с добавленными столбцами Ichimoku.
        """
        df = data.copy()
        df['Tenkan_sen'] = (
                                   df['High'].rolling(window=self.tenkan_period).max() +
                                   df['Low'].rolling(window=self.tenkan_period).min()
                           ) / 2
        df['Kijun_sen'] = (
                                  df['High'].rolling(window=self.kijun_period).max() +
                                  df['Low'].rolling(window=self.kijun_period).min()
                          ) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(self.shift)
        rolling_high = df['High'].rolling(window=self.senkou_span_b_period).max()
        rolling_low = df['Low'].rolling(window=self.senkou_span_b_period).min()
        df['Senkou_Span_B'] = ((rolling_high + rolling_low) / 2).shift(self.shift)
        df['Chikou_Span'] = df['Close'].shift(-self.shift)
        df['Signal'] = 0
        df['Cloud_Upper'] = df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
        df['Cloud_Lower'] = df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)
        df.loc[df['Close'] > df['Cloud_Upper'], 'Signal'] = 1
        df.loc[df['Close'] < df['Cloud_Lower'], 'Signal'] = -1
        return df


class AlgorithmCCI:
    def __init__(self, period: int = 20, constant: float = 0.015):
        """
        Алгоритм расчёта CCI (Commodity Channel Index).
        :param period: Период расчёта CCI (по умолчанию 20).
        :param constant: Константа в формуле CCI (обычно 0.015).
        """
        self.period = period
        self.constant = constant

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает CCI по формуле:
          Typical Price = (High + Low + Close)/3
          SMA_TP = скользящая средняя от Typical Price
          Mean Deviation = среднее абсолютное отклонение от SMA_TP
          CCI = (TypicalPrice - SMA_TP)/(constant * MeanDeviation)
        Добавляет столбец 'CCI' и простой 'Signal':
          - 1, если CCI > 100
          - -1, если CCI < -100
          - 0 иначе
        :param data: DataFrame со столбцами 'High', 'Low', 'Close'.
        :return: Копия DataFrame с добавленным столбцом 'CCI' и 'Signal'.
        """
        df = data.copy()
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['SMA_TP'] = df['TP'].rolling(self.period).mean()

        def mean_abs_dev(x):
            return (x - x.mean()).abs().mean()
        df['Mean_Dev'] = df['TP'].rolling(self.period).apply(mean_abs_dev, raw=False)
        df['CCI'] = (df['TP'] - df['SMA_TP']) / (self.constant * df['Mean_Dev'])
        df['Signal'] = 0
        df.loc[df['CCI'] > 100, 'Signal'] = 1
        df.loc[df['CCI'] < -100, 'Signal'] = -1
        return df


class AlgorithmStochastic:
    def __init__(self, k_period: int = 14, d_period: int = 3):
        """
        Алгоритм расчёта Stochastic Oscillator (Стохастик).
        :param k_period: Период для %K (по умолчанию 14).
        :param d_period: Период сглаживания %K для получения %D (по умолчанию 3).
        """
        self.k_period = k_period
        self.d_period = d_period

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает Stochastic Oscillator:
          %K = 100 * (Close - minLow(k_period)) / (maxHigh(k_period) - minLow(k_period))
          %D = скользящая средняя %K (d_period)
        Добавляет столбцы '%K', '%D' и 'Signal', где:
          Signal = 1, если %K > %D,
                  -1, если %K < %D,
                   0 в противном случае.
        :param data: DataFrame со столбцами 'High', 'Low', 'Close'.
        :return: Копия DataFrame с добавленными столбцами '%K', '%D', 'Signal'.
        """
        df = data.copy()
        df['Lowest_Low'] = df['Low'].rolling(self.k_period).min()
        df['Highest_High'] = df['High'].rolling(self.k_period).max()
        df['%K'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])
        df['%D'] = df['%K'].rolling(self.d_period).mean()
        df['Signal'] = 0
        df.loc[df['%K'] > df['%D'], 'Signal'] = 1
        df.loc[df['%K'] < df['%D'], 'Signal'] = -1
        return df


class AlgorithmWilliamsR:
    def __init__(self, period: int = 14):
        """
        Алгоритм расчёта Williams %R.
        :param period: Период расчёта (по умолчанию 14).
        """
        self.period = period

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает Williams %R:
          %R = -100 * (HighestHigh(period) - Close) / (HighestHigh(period) - LowestLow(period))
        Добавляет столбец '%R' и упрощённый сигнал:
          - 1, если %R < -80 (перепроданность)
          - -1, если %R > -20 (перекупленность)
          - 0 иначе
        :param data: DataFrame со столбцами 'High','Low','Close'.
        :return: Копия DataFrame с '%R' и 'Signal'.
        """
        df = data.copy()
        df['HH'] = df['High'].rolling(self.period).max()
        df['LL'] = df['Low'].rolling(self.period).min()
        df['%R'] = -100 * (df['HH'] - df['Close']) / (df['HH'] - df['LL'])
        df['Signal'] = 0
        df.loc[df['%R'] < -80, 'Signal'] = 1
        df.loc[df['%R'] > -20, 'Signal'] = -1
        return df


class AlgorithmOBV:
    def __init__(self):
        """
        Алгоритм на основе On-Balance Volume (OBV).
        """
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает OBV:
          Если Close[t] > Close[t-1], OBV[t] = OBV[t-1] + Volume[t]
          Если Close[t] < Close[t-1], OBV[t] = OBV[t-1] - Volume[t]
        Добавляет столбец 'OBV' и простой сигнал:
          - 1, если OBV растёт (OBV[t] > OBV[t-1])
          - -1, если OBV падает (OBV[t] < OBV[t-1])
          - 0 в остальных случаях (или на первом шаге).
        :param data: DataFrame со столбцами 'Close' и 'Volume'.
        :return: Копия DataFrame со столбцами 'OBV' и 'Signal'.
        """
        df = data.copy()
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] - df['Volume'].iloc[i]
            else:
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV']
        df['Signal'] = 0
        for i in range(1, len(df)):
            if df['OBV'].iloc[i] > df['OBV'].iloc[i - 1]:
                df.loc[df.index[i], 'Signal'] = 1
            elif df['OBV'].iloc[i] < df['OBV'].iloc[i - 1]:
                df.loc[df.index[i], 'Signal'] = -1
            else:
                df.loc[df.index[i], 'Signal'] = 0
        return df


class AlgorithmVWAP:
    def __init__(self):
        """
        Алгоритм расчёта VWAP (Volume Weighted Average Price).
        """
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает VWAP = (сумма(Цена * Объём)) / (сумма(Объём)).
        Часто берётся средняя цена = (High+Low+Close)/3.
        Для упрощения берём также (High+Low+Close)/3 как "Typical Price".
        Добавляет столбец 'VWAP' и простой сигнал:
          - 1, если Close > VWAP
          - -1, если Close < VWAP
        :param data: DataFrame, требующий столбцов 'High','Low','Close','Volume'.
        :return: Копия DataFrame с добавленными столбцами 'TypicalPrice', 'Cumul_PV', 'Cumul_Volume', 'VWAP', 'Signal'.
        """
        df = data.copy()
        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['PV'] = df['TypicalPrice'] * df['Volume']
        df['Cumul_PV'] = df['PV'].cumsum()
        df['Cumul_Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumul_PV'] / df['Cumul_Volume']
        df['Signal'] = 0
        df.loc[df['Close'] > df['VWAP'], 'Signal'] = 1
        df.loc[df['Close'] < df['VWAP'], 'Signal'] = -1
        return df


class AlgorithmBollingerBands:
    def __init__(self, window: int = 20, nbdev_up: float = 2.0, nbdev_dn: float = 2.0):
        """
        Алгоритм расчёта полос Боллинджера (Bollinger Bands).
        :param window: Период скользящей средней (по умолчанию 20).
        :param nbdev_up: Коэффициент для верхней полосы (по умолчанию 2.0).
        :param nbdev_dn: Коэффициент для нижней полосы (по умолчанию 2.0).
        """
        self.window = window
        self.nbdev_up = nbdev_up
        self.nbdev_dn = nbdev_dn

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает полосы Боллинджера на основе цены 'Close':
          - Средняя линия (Bollinger Middle) = SMA(Close, window)
          - Верхняя полоса (Bollinger Upper) = Middle + nbdev_up * Std(Close, window)
          - Нижняя полоса (Bollinger Lower) = Middle - nbdev_dn * Std(Close, window)
        Добавляет столбцы:
          - 'Boll_Middle'
          - 'Boll_Upper'
          - 'Boll_Lower'
          - 'Signal': упрощённо:
              +1, если Close пробил верхнюю полосу
              -1, если Close пробил нижнюю полосу
               0 — в остальных случаях
        :param data: DataFrame, в котором должен быть столбец 'Close'.
        :return: Копия DataFrame с добавленными столбцами Bollinger Bands.
        """
        df = data.copy()
        df['Boll_Middle'] = df['Close'].rolling(self.window).mean()
        df['Boll_Std'] = df['Close'].rolling(self.window).std()
        df['Boll_Upper'] = df['Boll_Middle'] + self.nbdev_up * df['Boll_Std']
        df['Boll_Lower'] = df['Boll_Middle'] - self.nbdev_dn * df['Boll_Std']
        df['Signal'] = 0
        df.loc[df['Close'] > df['Boll_Upper'], 'Signal'] = 1
        df.loc[df['Close'] < df['Boll_Lower'], 'Signal'] = -1
        return df


class AlgorithmATR:
    def __init__(self, period: int = 14):
        """
        Алгоритм расчёта ATR (Average True Range).
        :param period: Период для расчёта ATR (по умолчанию 14).
        """
        self.period = period

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает ATR по стандартной методике:
          TR (True Range) = max( High - Low, abs(High - PrevClose), abs(Low - PrevClose) )
          ATR = скользящее среднее (обычно RMA/EMA) от TR за 'period' периодов
        Добавляет столбцы:
          - 'TR'
          - 'ATR'
          - 'Signal': упрощённо:
               +1, если TR > ATR (сильная волатильность)
               -1, если TR < ATR
                0 — иначе (или на первом ряде)
        :param data: DataFrame, содержащий столбцы 'High', 'Low', 'Close'.
        :return: Копия DataFrame с добавленными столбцами 'TR', 'ATR', 'Signal'.
        """
        df = data.copy()
        df['PrevClose'] = df['Close'].shift(1)
        df['TR'] = df[['High', 'PrevClose']].max(axis=1) - df[['Low', 'PrevClose']].min(axis=1)
        df['ATR'] = df['TR'].ewm(span=self.period, adjust=False).mean()
        df['Signal'] = 0
        df.loc[df['TR'] > df['ATR'], 'Signal'] = 1
        df.loc[df['TR'] < df['ATR'], 'Signal'] = -1
        df.drop(columns=['PrevClose'], inplace=True)
        return df


class AlgorithmARIMA:
    def __init__(self, p: int = 1, d: int = 0, q: int = 1):
        """
        Алгоритм на основе ARIMA (AutoRegressive Integrated Moving Average).
        :param p: Параметр p (порядок авторегрессии).
        :param d: Параметр d (порядок дифференцирования).
        :param q: Параметр q (порядок скользящего среднего).
        """
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.results = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Строит модель ARIMA на серии 'Close' и добавляет столбец 'ARIMA_Fitted',
        содержащий подгонку (in-sample) к историческим данным.
        Также формирует упрощённый сигнал 'Signal':
          - 1, если ARIMA_Fitted < фактическое значение Close (ожидание роста)
          - -1, если ARIMA_Fitted > фактическое значение Close (ожидание снижения)
          - 0 — иначе (в случае, если значения равны или при недоступных данных)
        В реальном использовании для прогноза будущих значений рекомендуется:
          - Делить данные на train/test;
          - Делать walk-forward прогноз на 1 (или несколько) шагов вперёд;
          - Подбирать (p, d, q) оптимальным образом.
        :param data: DataFrame, в котором должен быть столбец 'Close'.
        :return: Копия DataFrame с добавленными столбцами 'ARIMA_Fitted' и 'Signal'.
        """
        df = data.copy()
        self.model = ARIMA(df['Close'], order=(self.p, self.d, self.q))
        self.results = self.model.fit()
        df['ARIMA_Fitted'] = self.results.fittedvalues
        df['Signal'] = 0
        mask_up = df['ARIMA_Fitted'] < df['Close']
        mask_dn = df['ARIMA_Fitted'] > df['Close']
        df.loc[mask_up, 'Signal'] = 1
        df.loc[mask_dn, 'Signal'] = -1
        return df


class AlgorithmSARIMA:
    def __init__(self,
                 order=(1, 0, 1),
                 seasonal_order=(0, 0, 0, 0)):
        """
        Алгоритм на основе SARIMA (Seasonal ARIMA).
        :param order: (p, d, q) – порядок не-сезонной части модели.
        :param seasonal_order: (P, D, Q, m) – порядок сезонной части модели
                               и m – длина сезонности (например, 12 для месяцев).
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Строит модель SARIMA на серии 'Close' и добавляет столбец 'SARIMA_Fitted',
        содержащий подгонку (in-sample) к историческим данным.
        Также формирует упрощённый сигнал 'Signal':
          - 1, если SARIMA_Fitted < фактическое значение Close
          - -1, если SARIMA_Fitted > фактическое значение Close
          - 0, иначе (или при NaN)
        :param data: DataFrame, ожидается столбец 'Close'.
        :return: Копия DataFrame с добавленными столбцами 'SARIMA_Fitted' и 'Signal'.
        """
        df = data.copy()
        self.model = SARIMAX(
            df['Close'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.results = self.model.fit(disp=False)
        df['SARIMA_Fitted'] = self.results.fittedvalues
        df['Signal'] = 0
        mask_up = df['SARIMA_Fitted'] < df['Close']
        mask_dn = df['SARIMA_Fitted'] > df['Close']
        df.loc[mask_up, 'Signal'] = 1
        df.loc[mask_dn, 'Signal'] = -1
        return df


