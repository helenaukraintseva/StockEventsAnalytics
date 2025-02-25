import pandas as pd
import numpy as np


class TrendDetector:
    """
    Класс для определения тренда по ценам: вверх/вниз и детектирования разворота.

    - 'trend_up': цена растёт несколько баров подряд.
    - 'trend_down': цена падает несколько баров подряд.
    - 'reversal_up': был тренд вниз, теперь движение вверх.
    - 'reversal_down': был тренд вверх, теперь движение вниз.

    Наивная логика:
    1. Если последние 'period' изменений цены положительны -> 'trend_up'
    2. Если последние 'period' изменений цены отрицательны -> 'trend_down'
    3. Если предыдущий тренд был down, а теперь up -> 'reversal_up'
    4. Если предыдущий тренд был up, а теперь down -> 'reversal_down'

    """

    def __init__(self, target_col="Close", period=3):
        """
        :param target_col: Название столбца с ценами, по умолчанию Close.
        :param period: Число последних баров для определения тренда.
        """
        self.target_col = target_col
        self.period = period

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет столбец 'TrendSignal' с одним из значений:
        ['trend_up', 'trend_down', 'reversal_up', 'reversal_down', 'flat/none'].

        :param data: DataFrame, где есть хотя бы столбец self.target_col (Close).
        :return: Копия DataFrame с добавленным столбцом 'TrendSignal'.
        """
        df = data.copy()
        if self.target_col not in df.columns:
            raise ValueError(f"Столбец {self.target_col} не найден в DataFrame.")

        # Вычисляем изменения цены (diff)
        df['diff'] = df[self.target_col].diff()

        # Для удобства создадим массив сигналов
        signals = []

        # Предыдущий тренд (None / 'up' / 'down')
        prev_trend = None

        for i in range(len(df)):
            if i < self.period:
                signals.append("flat/none")
                continue

            # Берем последние 'period' изменений
            recent_diffs = df['diff'].iloc[i - self.period + 1: i + 1]

            # Проверяем, все ли > 0 (тренд вверх) или < 0 (тренд вниз)
            all_up = np.all(recent_diffs > 0)
            all_down = np.all(recent_diffs < 0)

            if all_up:
                current_trend = 'up'
            elif all_down:
                current_trend = 'down'
            else:
                current_trend = 'flat'

            # Логика разворота
            if prev_trend == 'down' and current_trend == 'up':
                signal = 'reversal_up'
            elif prev_trend == 'up' and current_trend == 'down':
                signal = 'reversal_down'
            else:
                # Просто тренд
                if current_trend == 'up':
                    signal = 'trend_up'
                elif current_trend == 'down':
                    signal = 'trend_down'
                else:
                    signal = 'flat/none'

            signals.append(signal)
            prev_trend = current_trend if current_trend in ['up', 'down'] else prev_trend

        df['TrendSignal'] = signals
        return df

    def detect_last_signal(self, data: pd.DataFrame) -> str:
        """
        Возвращает последнее значение ('TrendSignal') из detect(...).
        Удобно, если нужно просто узнать состояние сейчас.

        :param data: DataFrame со столбцом self.target_col.
        :return: Последняя строка 'TrendSignal' или 'flat/none'.
        """
        result_df = self.detect(data)
        return result_df['TrendSignal'].iloc[-1]


if __name__ == "__main__":
    # Допустим, у вас есть DataFrame с ценами
    data = pd.DataFrame({
        "Close": [100, 101, 102, 101, 100, 98, 97, 99, 100, 101],
        "time": [100, 101, 102, 101, 100, 98, 97, 99, 100, 101],

    })

    # Инициализация детектора трендов
    trend_detector = TrendDetector(target_col="Close", period=3)

    # Получаем DataFrame с 'TrendSignal'
    result_df = trend_detector.detect(data)
    print(result_df)

    # Получаем последний сигнал
    last_signal = trend_detector.detect_last_signal(data)
    print("Последний сигнал тренда:", last_signal)
