import pandas as pd
import numpy as np
import os
import joblib
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import BaggingRegressor

# === Настройки ===
DATA_PATH = "datasets/re_0a1_i1h_w20_s1_p1.csv"
MODEL_SAVE_DIR = "trained_models_v2"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
WINDOW_SIZE = 20

# === Загрузка и подготовка данных ===
def parse_step(step_str):
    return list(map(float, step_str.split('|')))

def load_data(filepath, window_size=WINDOW_SIZE):
    df = pd.read_csv(filepath)
    X, y = [], []

    for _, row in df.iterrows():
        features = []
        for i in range(window_size):
            features.extend(parse_step(row[f"step{i}"]))
        X.append(features)
        y.append(float(row["target"]))
    return np.array(X), np.array(y)

# === Обучение нескольких моделей ===
def train_multiple_models(X, y, models_dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler_v2.pkl"))

    results = []
    predictions = []

    for name, model in models_dict.items():
        model_instance = model()
        start_time = time.time()
        model_instance.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = model_instance.predict(X_test_scaled)
        pred_time = time.time() - start_time

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "model": name,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "train_time": train_time,
            "pred_time": pred_time
        })

        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}_v2.pkl")
        joblib.dump(model_instance, model_path)

        predictions.append((name, y_test, y_pred))

    return results, predictions

# === Основной блок запуска ===
models = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForest": RandomForestRegressor
}

X, y = load_data(DATA_PATH)
results, predictions = train_multiple_models(X, y, models)

# === Вывод результатов ===
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(MODEL_SAVE_DIR, "metrics_summary_v2.csv")
results_df.to_csv(results_csv_path, index=False)

# === График реальных и предсказанных значений ===
plt.figure(figsize=(12, 6))
plt.plot(y, label="📈 Реальные значения", linewidth=2, color='black')

for name, y_true, y_pred in predictions:
    plt.plot(np.linspace(0, len(y), len(y_true)), y_pred, label=f"🔮 {name}")

plt.title("Сравнение предсказаний моделей с реальными значениями")
plt.xlabel("Индекс")
plt.ylabel("Цена")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt_path = os.path.join(MODEL_SAVE_DIR, "prediction_plot_v2.png")
plt.savefig(plt_path)
plt.show()
