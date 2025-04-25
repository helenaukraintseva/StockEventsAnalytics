import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# === Генерация синтетических данных ===
def generate_data(seq_length=1000):
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


# === Определение LSTM-модели ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1])
        return out


# === Обучение модели ===
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return model


# === Прогноз и продолжение временного ряда ===
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


# === Визуализация прогноза ===
def plot_forecasts(original_series, predictions_dict, future_steps):
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(original_series)), original_series, label='Реальные данные')
    for label, pred in predictions_dict.items():
        plt.plot(np.arange(len(original_series), len(original_series) + future_steps), pred, label=label)
    plt.title('Продолжение временного ряда разными моделями')
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

    models = {
        "LSTM-50": LSTMModel(hidden_size=50),
        "LSTM-100": LSTMModel(hidden_size=100),
    }

    predictions = {}
    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        last_window = X_test[-1].unsqueeze(0)  # [1, window_size, 1]
        prediction = forecast_future(trained_model, last_window, future_steps, scaler)
        predictions[name] = prediction

    plot_forecasts(data, predictions, future_steps)


# Запуск
main()
