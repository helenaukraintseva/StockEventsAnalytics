import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# === Генерация данных ===
def generate_data(seq_length=300):
    t = np.linspace(0, 100, seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(seq_length)
    return data.reshape(-1, 1)


# === Подготовка данных ===
def prepare_data(series, window_size=50):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


# === Модели ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1])


class CNNModel(nn.Module):
    def __init__(self, input_size=1, out_channels=16, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pool(self.relu(self.conv1(x))).squeeze(-1)
        return self.linear(x)


# === Обучение моделей ===
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return model


# === Прогноз ===
def forecast_future(model, last_window, future_steps, scaler):
    model.eval()
    generated = []
    for _ in range(future_steps):
        with torch.no_grad():
            next_val = model(last_window)
        generated.append(next_val.item())
        next_val_tensor = next_val.view(1, 1, 1)
        last_window = torch.cat((last_window[:, 1:], next_val_tensor), dim=1)
    return scaler.inverse_transform(np.array(generated).reshape(-1, 1))


# === Модель линейной регрессии ===
def forecast_linear_regression(X_train, y_train, X_test_last, scaler, future_steps):
    # Преобразуем в numpy для sklearn
    X_np = X_train.squeeze(-1).numpy()
    y_np = y_train.numpy()
    model = LinearRegression()
    model.fit(X_np, y_np)

    last_window = X_test_last.squeeze(0).squeeze(-1).numpy()
    generated = []
    for _ in range(future_steps):
        next_val = model.predict(last_window.reshape(1, -1))[0]
        generated.append(next_val)
        last_window = np.append(last_window[1:], next_val)
    return scaler.inverse_transform(np.array(generated).reshape(-1, 1))


# === График ===
def plot_forecasts(original_series, predictions_dict, future_steps):
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(original_series)), original_series, label='Реальные данные')
    for label, pred in predictions_dict.items():
        plt.plot(np.arange(len(original_series), len(original_series) + future_steps), pred, label=label)
    plt.title('Сравнение моделей временного ряда')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Основной процесс ===
def main():
    window_size = 50
    future_steps = 100
    data = generate_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = prepare_data(data_scaled, window_size)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]
    X_test_last = X_test[-1].unsqueeze(0)

    # Инициализация моделей
    models = {
        # "LSTM": LSTMModel(),
        # "GRU": GRUModel(),
        # "CNN": CNNModel()
    }

    predictions = {}
    for name, model in models.items():
        trained = train_model(model, X_train, y_train)
        pred = forecast_future(trained, X_test_last, future_steps, scaler)
        predictions[name] = pred

    # Добавляем МО модель
    lr_pred = forecast_linear_regression(X_train, y_train, X_test_last, scaler, future_steps)
    predictions["LinearRegression"] = lr_pred

    # График
    plot_forecasts(data, predictions, future_steps)

# Запуск
main()
