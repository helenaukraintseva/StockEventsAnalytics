import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# Генерация синтетических данных
def generate_data(seq_length=1000):
    t = np.linspace(0, 100, seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(seq_length)
    return data.reshape(-1, 1)


# Подготовка данных
def prepare_data(series, window_size=50):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


# Параметры
window_size = 50
data = generate_data()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = prepare_data(data_scaled, window_size)
X = torch.tensor(X, dtype=torch.float32) # [samples, timesteps, features]
y = torch.tensor(y, dtype=torch.float32)

# Разделение на train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Модель
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Предсказание
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

predicted_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# Построение графика
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Реальные значения')
plt.plot(predicted_rescaled, label='Предсказания')
plt.title('Сравнение реальных и предсказанных значений временного ряда (PyTorch)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
