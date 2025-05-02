import pandas as pd
import numpy as np
import os
import joblib
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn

# === Настройки ===
DATA_PATH = "train_price.csv"
MODEL_SAVE_DIR = "trained_models_v4"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
WINDOW_SIZE = 20

# === PyTorch модели ===
class RNNModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

# === Загрузка и подготовка данных ===
def parse_step(step_str):
    return list(map(float, step_str.split('|')))

def load_data(filepath):
    df = pd.read_csv(filepath)
    X, y = [], []

    for _, row in df.iterrows():
        steps = [parse_step(row[f"step{i}"]) for i in range(WINDOW_SIZE)]
        X.append(steps)
        y.append(float(row["target"]))
    X = np.array(X)
    print(X.shape)
    print("X.shape")
    return np.array(X), np.array(y)

# === Обучение torch-моделей ===
def train_torch_model(model, X_train, y_train, X_test, y_test, name, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    start_train = time.time()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_train

    start_pred = time.time()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
    pred_time = time.time() - start_pred

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"{name}_v2.pth"))

    return {
        "model": name,
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "train_time": train_time,
        "pred_time": pred_time
    }, (name, y_true, y_pred)

# === Обучение sklearn-моделей ===
def train_multiple_models(X, y, models_dict):
    flat_X = X.reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(flat_X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler_v2.pkl"))

    results, predictions = [], []

    for name, model in models_dict.items():
        instance = model()
        start_train = time.time()
        instance.fit(X_train_scaled, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = instance.predict(X_test_scaled)
        pred_time = time.time() - start_pred

        results.append({
            "model": name,
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "train_time": train_time,
            "pred_time": pred_time
        })

        joblib.dump(instance, os.path.join(MODEL_SAVE_DIR, f"{name}_v2.pkl"))
        predictions.append((name, y_test, y_pred))

    return results, predictions

# === Основной запуск ===
if __name__ == "__main__":
    X, y = load_data(DATA_PATH)

    models = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "RandomForest": RandomForestRegressor,
        "BayesianRidge": BayesianRidge,
        "BaggingRegressor": BaggingRegressor,
        "MLPRegressor": MLPRegressor
    }

    results_sklearn, preds_sklearn = train_multiple_models(X, y, models)

    # Обучение torch-моделей
    results_torch, preds_torch = [], []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, cls in {
        "RNN": RNNModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel
    }.items():
        model = cls(input_size=5)
        result, pred = train_torch_model(model, X_train, y_train, X_test, y_test, name)
        results_torch.append(result)
        preds_torch.append(pred)

    all_results = results_sklearn + results_torch
    all_preds = preds_sklearn + preds_torch

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(MODEL_SAVE_DIR, "metrics_summary_v2.csv"), index=False)

    # === График
    plt.figure(figsize=(12, 6))
    plt.plot(y, label="📈 Реальные значения", linewidth=2, color='black')
    for name, y_true, y_pred in all_preds:
        plt.plot(np.linspace(0, len(y), len(y_true)), y_pred, label=f"🔮 {name}")
    plt.title("Сравнение предсказаний моделей с реальными значениями")
    plt.xlabel("Индекс")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "prediction_plot_v2.png"))
    plt.show()
