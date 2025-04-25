import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# === Генерация данных ===
def generate_data(seq_length=1000):
    t = np.linspace(0, 100, seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(seq_length)
    return data.reshape(-1, 1)

# === Подготовка ===
def prepare_data(series, window_size=50):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# === Параметры ===
window_size = 50
future_steps = 100
data = generate_data()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_np, y_np = prepare_data(data_scaled, window_size)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Модель ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Обучение ===
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.6f}")

# === Предсказания на тесте ===
model.eval()
with torch.no_grad():
    test_preds = model(X_test).numpy()
    test_preds_rescaled = scaler.inverse_transform(test_preds)
    y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# === Продолжение ряда ===
last_window = X_test[-1].unsqueeze(0)  # [1, window_size, 1]
generated = []

for _ in range(future_steps):
    with torch.no_grad():
        next_val = model(last_window)  # [1, 1]
    generated.append(next_val.item())

    # Обновить окно входа
    next_val_tensor = next_val.view(1, 1, 1)
    last_window = torch.cat((last_window[:, 1:], next_val_tensor), dim=1)

generated_rescaled = scaler.inverse_transform(np.array(generated).reshape(-1, 1))

# === График ===
plt.figure(figsize=(14, 6))
plt.plot(np.arange(len(data)), data, label='Реальные данные')
plt.plot(np.arange(len(data), len(data) + future_steps), generated_rescaled, label='Продолжение (прогноз)')
plt.title('Продолжение временного ряда моделью LSTM')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
