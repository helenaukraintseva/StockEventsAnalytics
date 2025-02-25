import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import requests
import time


class BaseAlgorithm:
    def __init__(self, model_class, weights_file, target_col, future_steps=5, **model_kwargs):
        """
        Класс, который детектирует (предсказывает) цену на несколько (future_steps) свечей вперёд.

        :param model_class: Класс модели (например, RNN, LSTM, GRU и т.д.)
        :param weights_file: Файл для сохранения / загрузки весов модели.
        :param target_col: Название столбца-цели (например, 'Close').
        :param future_steps: На сколько шагов вперёд хотим сделать предсказание (по умолч. 5).
        :param model_kwargs: Доп. параметры для инициализации модели.
        """
        self.ensure_directory_exists("models")
        self.weights_file = f"models/{weights_file}"
        self.title_model = weights_file.split("_")[0]
        self.target_col = target_col
        self.future_steps = future_steps
        self.model = model_class(**model_kwargs)

    @staticmethod
    def ensure_directory_exists(directory_path="models"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Директория '{directory_path}' создана.")
        else:
            print(f"Директория '{directory_path}' уже существует.")

    def train(self, data: pd.DataFrame, epochs=5, lr=1e-3):
        """
        Обучает модель на предоставленных данных, сдвигая целевую переменную (target)
        на self.future_steps баров вперёд, и сохраняет веса.

        :param data: DataFrame со столбцами признаков и target_col.
        :param epochs: Количество эпох обучения.
        :param lr: Скорость обучения.
        """
        start_time = time.time()
        df = data.copy().reset_index(drop=True)

        # Формируем X и y c учётом сдвига на future_steps
        # Например, y[i] будет = price[i + future_steps].
        # Последние future_steps строк не будут иметь y => их удаляем.
        if len(df) <= self.future_steps:
            raise ValueError("Данных слишком мало для сдвига на future_steps.")

        X = df.iloc[:-self.future_steps].drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].shift(-self.future_steps).dropna().values.astype('float32')

        # Приведём X к формату (samples, 1, features), если RNN
        X = np.expand_dims(X, axis=1)

        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Обучение
        last_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
            last_loss = loss.item()

        # Сохраняем веса
        torch.save(self.model.state_dict(), self.weights_file)
        print(f"Модель сохранена в {self.weights_file}")
        return [self.title_model, round(time.time() - start_time, 2), round(last_loss, 2)]

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Загружает модель, затем предсказывает будущую цену на (future_steps) шагов вперёд.
        Здесь есть несколько вариантов логики.
        Для упрощения: предполагаем, что хотим спрогнозировать именно 1 точку
        (цена через future_steps баров), и у нас есть "последняя известная строка" как вход.

        :param data: DataFrame (признаки + target_col), последние строки (окно)
                     для которых хотим сделать прогноз на future_steps вперёд.
        :return: DataFrame с 'PredictedValue' (цена через self.future_steps баров).
        """
        df = data.copy().reset_index(drop=True)

        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"Файл с весами {self.weights_file} не найден!")

        # Загрузить веса
        self.model.load_state_dict(torch.load(self.weights_file))
        self.model.eval()

        # Подготовка X
        # Допустим, берём всю выборку df (или последнюю строку),
        # но при этом target_col удаляем, т.к. он не нужен для входа
        print(df)
        X = df.drop(["time", "low"], axis=1).values.astype('float32')
        # Для RNN формат (samples, 1, features)
        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)
        print("FUCKKK")

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        # Создаём столбец 'PredictedValue' — это цена через future_steps
        df['PredictedValue'] = preds

        return df

class RNNModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, features)
        out, hidden = self.rnn(x)
        out = out[:, -1, :]  # берем последнее значение по seq_len
        out = self.fc(out)
        return out

# Пример LSTM-модели
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Пример GRU-модели
class GRUModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Пример CNN 1D-модели
class CNNModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, output_dim=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.squeeze(-1)  # убираем последнюю размерность
        x = self.fc(x)
        return x

class AlgorithmRNN(BaseAlgorithm):
    def __init__(self, weights_file="rnn_weights_price.pth", target_col="close", **model_kwargs):
        super().__init__(model_class=RNNModel,
                         weights_file=weights_file,
                         target_col=target_col,
                         **model_kwargs)

class AlgorithmLSTM(BaseAlgorithm):
    def __init__(self, weights_file="lstm_weights_price.pth", target_col="close", **model_kwargs):
        super().__init__(model_class=LSTMModel,
                         weights_file=weights_file,
                         target_col=target_col,
                         **model_kwargs)

class AlgorithmGRU(BaseAlgorithm):
    def __init__(self, weights_file="gru_weights_price.pth", target_col="close", **model_kwargs):
        super().__init__(model_class=GRUModel,
                         weights_file=weights_file,
                         target_col=target_col,
                         **model_kwargs)

class AlgorithmCNN(BaseAlgorithm):
    def __init__(self, weights_file="cnn_weights_price.pth", target_col="close", **model_kwargs):
        super().__init__(model_class=CNNModel,
                         weights_file=weights_file,
                         target_col=target_col,
                         **model_kwargs)


def train_models():
    df = pd.read_csv("new_dataset_price.csv")
    df = df.drop(["Unnamed: 0", "TargetPrice"], axis=1)
    rnn = AlgorithmRNN()
    lstm = AlgorithmLSTM()
    gru = AlgorithmGRU()
    cnn = AlgorithmCNN()

    train_time = list()

    train_time.append(rnn.train(data=df))
    train_time.append(lstm.train(data=df))
    train_time.append(gru.train(data=df))
    train_time.append(cnn.train(data=df))
    train_time = np.array(train_time)
    stat = pd.DataFrame(train_time, columns=["Model", "Train Time (s)", "Loss MSE"])
    stat.to_csv("Stat_train_nn_price.csv")


if __name__ == "__main__":
    train_models()



    token = "8038576871:AAFAQ60J11zxws3FlhFMZ4lgzSq8Rdn8iMw"
    chat_id = "961023982"
    message = "Обучение моделей закончилось: НС Цена"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }

    response = requests.post(url, json=payload)