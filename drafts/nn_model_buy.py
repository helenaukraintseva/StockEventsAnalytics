import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import requests
import math
import time

# ------------------------ Базовый класс для всех алгоритмов ------------------------ #
class BaseAlgorithm:
    def __init__(self, model_class, weights_file, target_col, **model_kwargs):
        """
        Базовый класс для всех алгоритмов машинного обучения.

        :param model_class: Класс модели (RNN, LSTM, GRU и т. д.)
        :param weights_file: Файл для сохранения/загрузки весов модели.
        :param target_col: Название столбца-цели.
        :param model_kwargs: Дополнительные параметры для модели.
        """
        directory = "modelsdfasfasfd"
        self.ensure_directory_exists(directory)
        self.weights_file = f"modelsdfasfasfd/{weights_file}"
        self.title_model = weights_file.split("_")[0]
        self.target_col = target_col
        self.model = model_class(**model_kwargs)

    @staticmethod
    def ensure_directory_exists(directory_path="modelsdfasfasfd"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Директория '{directory_path}' создана.")
        else:
            print(f"Директория '{directory_path}' уже существует.")

    def train(self, data: pd.DataFrame, epochs=5, lr=1e-3):
        """
        Обучает модель на предоставленных данных и сохраняет веса.

        :param data: Датафрейм с входными признаками и целевой переменной.
        :param epochs: Количество эпох обучения.
        :param lr: Скорость обучения.
        """
        start_time = time.time()
        df = data.copy()
        X = df.drop(columns=[self.target_col]).values.astype('float32')
        # X = df.values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)  # Преобразуем для RNN-совместимого формата
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Обучение модели
        last_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            last_loss = loss.item()

        # Сохраняем веса
        torch.save(self.model.state_dict(), self.weights_file)
        print(f"Модель сохранена в {self.weights_file}")
        return [self.title_model, round(time.time() - start_time, 2), round(last_loss, 2)]

    def predict(self, data: pd.DataFrame, pas: float = 0.2):
        """
        Загружает модель и делает предсказание.

        :param data: Датафрейм с входными признаками.
        :return: Датафрейм с предсказаниями и торговыми сигналами.
        """
        df = data.copy()
        X = df.values.astype('float32')
        # X = df.drop(columns=[self.target_col]).values.astype('float32')
        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)

        # Загружаем веса модели
        if os.path.exists(self.weights_file):
            self.model.load_state_dict(torch.load(self.weights_file))
            print(f"Загружены веса из {self.weights_file}")
        else:
            raise FileNotFoundError(f"Файл с весами {self.weights_file} не найден!")

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        print(df)

        df['Signal'] = np.where(preds > df["close"], 1, -1)
        return df


class RNNNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, features)
        out, hidden = self.rnn(x)
        out = out[:, -1, :]  # берём последнее значение по временной оси
        out = self.fc(out)
        return out

# --- Пример LSTM ---
class LSTMNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8, output_dim=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Пример GRU ---
class GRUNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=1):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Пример CNN 1D ---
class CNNNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, output_dim=1):
        super(CNNNet, self).__init__()
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
    def __init__(self, weights_file="rnn_weights_buy.pth", target_col="class", **model_kwargs):
        # Передаём RNNNet в BaseAlgorithm
        super().__init__(model_class=RNNNet, weights_file=weights_file, target_col=target_col, **model_kwargs)

class AlgorithmLSTM(BaseAlgorithm):
    def __init__(self, weights_file="lstm_weights_buy.pth", target_col="class", **model_kwargs):
        super().__init__(model_class=LSTMNet, weights_file=weights_file, target_col=target_col, **model_kwargs)

class AlgorithmGRU(BaseAlgorithm):
    def __init__(self, weights_file="gru_weights_buy.pth", target_col="class", **model_kwargs):
        super().__init__(model_class=GRUNet, weights_file=weights_file, target_col=target_col, **model_kwargs)

class AlgorithmCNN(BaseAlgorithm):
    def __init__(self, weights_file="cnn_weights_buy.pth", target_col="class", **model_kwargs):
        super().__init__(model_class=CNNNet, weights_file=weights_file, target_col=target_col, **model_kwargs)


def train_models():
    df = pd.read_csv("new_dataset_buy.csv")

    df = df.drop(["Unnamed: 0"], axis=1)
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
    stat.to_csv("Stat_train_nn_buy.csv")


if __name__ == "__main__":
    train_models()

    token = "8038576871:AAFAQ60J11zxws3FlhFMZ4lgzSq8Rdn8iMw"
    chat_id = "961023982"
    message = "Обучение моделей закончилось: НС Покупка"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }

    response = requests.post(url, json=payload)