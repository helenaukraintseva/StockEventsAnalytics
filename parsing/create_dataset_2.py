import os
import pandas as pd
import numpy as np


class CSVBatchBuilder:
    def __init__(self, interval="1m", time_col="open_time", close_col="close", features=None):
        """
        :param csv_files: список путей к csv-файлам
        :param time_col: Название столбца со временем (по умолчанию 'time')
        :param close_col: Название столбца с ценой закрытия (по умолчанию 'close')
        :param features: Список фич, которые брать в X. Если None, берём ['open','high','low','close','volume']
        """
        self.interval = interval
        self.csv_files = self.list_files_in_directory(directory="crypto_data", interval=interval)[:5]
        self.time_col = time_col
        self.close_col = close_col
        self.dataset_directory = "datasets"
        if features is None:
            self.features = ["open", "high", "low", "close", "volume"]
        else:
            self.features = features

    @staticmethod
    def _test_exist_file(file_path):
        if os.path.exists(file_path):
            return True
        else:
            return False


    def _save_dataset_to_csv(self, X: np.ndarray, y: np.ndarray, out_csv: str):
        """
        Приватный метод для сохранения X, y в CSV.
        X имеет форму (N, window_size, num_features).
        Для записи разворачиваем (flatten) по оси window_size × num_features => (N, W*F).
        Затем добавляем столбец 'target' (или 'label').
        """
        if X.size == 0:
            print("[WARNING] Нечего сохранять, X пустой.")
            return
        N, W, F = X.shape  # (num_samples, window_size, num_features)
        new_X = list()
        for elem in X:
            batch = list()
            for el in elem:
                line = [str(ee) for ee in el]
                batch.append("|".join(line))
            new_X.append(batch)

        # Разворачиваем X в 2D
        # X_2d = X.reshape(N, W * F)
        X_2d = np.array(new_X)
        # Генерируем названия колонок для X
        col_names = []
        for i in range(W):
            col_names.append(f"step{i}")
        df_out = pd.DataFrame(X_2d, columns=col_names)
        df_out["target"] = y
        df_out.to_csv(out_csv, index=False)
        print(f"[INFO] Сохранён датасет (N={N}, W={W}, F={F}) в {out_csv}")

    @staticmethod
    def list_files_in_directory(directory, interval):
        try:
            files = list()
            # Получаем список файлов и папок в указанной директории
            items = os.listdir(directory)

            # Проходим по всем элементам и проверяем, являются ли они файлами
            for item in items:
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    if interval in item_path:
                        files.append(item_path)
            return files
        except Exception as e:
            print(f"Ошибка: {e}")

    def build_regression_batches(self, window_size=10, predict_step=1):
        """
        Формирует X, y для задачи регрессии (предсказание цены).
        X будет иметь форму (N, window_size, len(features)),
        а y будет содержать целевую цену через predict_step.

        :param window_size: количество «тиков» (строк) на вход
        :param predict_step: на сколько шагов вперёд делаем прогноз
        :return: (X, y) — numpy-массивы
        """
        all_X = []
        all_y = []

        out_csv = f"datasets/regress_dataset_i{self.interval}_w{window_size}_s{predict_step}.csv"
        if self._test_exist_file(out_csv):
            print(f"Файл {out_csv} уже существует.")
            return

        for file_path in self.csv_files:
            if not os.path.exists(file_path):
                print(f"[WARNING] Файл {file_path} не найден — пропускаем.")
                continue

            df = pd.read_csv(file_path)
            # Проверяем наличие нужных колонок
            needed_cols = set(self.features + [self.close_col])
            missing = needed_cols - set(df.columns)
            if missing:
                print(f"[WARNING] Файл {file_path} не содержит столбцы {missing} — пропускаем.")
                continue

            # Сортируем по времени (на всякий случай)
            if self.time_col in df.columns:
                df.sort_values(by=self.time_col, inplace=True, ascending=True)

            # Преобразуем в numpy
            arr = df[self.features].values  # 2D (num_rows, num_features)
            close_prices = df[self.close_col].values

            # Будем формировать «окна» длиной window_size,
            # а в y — значение (close[i + window_size + predict_step - 1])
            # Последние (window_size + predict_step - 1) строк не смогут сформировать окно
            limit = len(df) - window_size - predict_step + 1
            if limit <= 0:
                continue  # слишком мало данных

            for start_i in range(limit):
                # X.shape -> (window_size, num_features)
                x_window = arr[start_i: start_i + window_size]
                # y — это будущая цена (через predict_step)
                y_value = close_prices[start_i + window_size + predict_step - 1]

                all_X.append(x_window)
                all_y.append(y_value)

        if not all_X:
            print("Нет данных для формирования регрессионных батчей!")
            # Возвращаем пустые массивы
            return np.array([]), np.array([])

        X = np.array(all_X)  # shape (N, window_size, num_features)
        y = np.array(all_y)  # shape (N,)

        self._save_dataset_to_csv(X, y, out_csv)
        print("REGRESSION SHAPES:", X.shape, y.shape)

    def build_classification_batches(self, window_size=10, predict_step=1, threshold=0.0):
        """
        Формирует X, y для задачи классификации (покупать/продавать).
        По умолчанию:
          - если future_close > current_close => класс 1 (покупать)
          - иначе => класс 0 (продавать)
        Можно добавить порог threshold, если нужно учитывать незначительные изменения.

        :param window_size: количество тиков истории
        :param predict_step: через сколько тиков смотреть future_close
        :param threshold: можно задать минимальный «разрыв» цены для класса 1 (рост).
        :return: (X, y) — numpy-массивы, где y ∈ {0,1}
        """
        all_X = []
        all_y = []

        out_csv = f"datasets/class_dataset_i{self.interval}_w{window_size}_s{predict_step}.csv"
        if self._test_exist_file(out_csv):
            print(f"Файл {out_csv} уже существует.")
            return

        for file_path in self.csv_files:
            if not os.path.exists(file_path):
                print(f"[WARNING] Файл {file_path} не найден — пропускаем.")
                continue

            df = pd.read_csv(file_path)
            # Проверяем наличие нужных колонок
            needed_cols = set(self.features + [self.close_col])
            missing = needed_cols - set(df.columns)
            if missing:
                print(f"[WARNING] Файл {file_path} не содержит столбцы {missing} — пропускаем.")
                continue

            if self.time_col in df.columns:
                df.sort_values(by=self.time_col, inplace=True, ascending=True)

            arr = df[self.features].values
            close_prices = df[self.close_col].values

            limit = len(df) - window_size - predict_step + 1
            if limit <= 0:
                continue

            for start_i in range(limit):
                x_window = arr[start_i: start_i + window_size]
                current_close = close_prices[start_i + window_size - 1]
                future_close = close_prices[start_i + window_size + predict_step - 1]

                # Простая логика:
                # if future_close - current_close > threshold => 1
                # else => 0
                delta = abs(future_close - current_close) / current_close
                if delta > threshold and future_close > current_close:
                    label = 1  # BUY
                elif delta > threshold and future_close < current_close:
                    label = -1  # SELL
                else:
                    label = 0  # NEITRAL

                all_X.append(x_window)
                all_y.append(label)

        if not all_X:
            print("Нет данных для формирования классификационных батчей!")
            return np.array([]), np.array([])

        X = np.array(all_X)  # shape (N, window_size, num_features)
        y = np.array(all_y)  # shape (N,)
        self._save_dataset_to_csv(X, y, out_csv)

        print("CLASSIFICATION SHAPES:", X.shape, y.shape)


# ----------------------- Пример использования -----------------------

if __name__ == "__main__":
    # csv_files = ["BTC.csv", "ETH.csv", "LTC.csv"]  # Список ваших файлов
    intervals = ["1m", "5m", "30m", "1h"]
    windows = [5, 10, 20]
    steps = [1, 5, 10]
    counter = 0
    for interval in intervals:

        builder = CSVBatchBuilder(interval=interval, time_col="time", close_col="close")
        print(len(builder.csv_files))

        for ww in windows:
            for ss in steps:
                # 1) Для регрессии
                builder.build_regression_batches(window_size=ww, predict_step=ss)

                # # 2) Для классификации
                builder.build_classification_batches(window_size=ww, predict_step=ss, threshold=0.05)

                counter += 1
                print()
                print(f"Step: {counter}\nInterval: {interval}\nWindow size: {ww}, step size: {ss}")
                print()
