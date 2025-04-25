import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# === Настройки ===
DATA_PATH = "datasets/re_0a1_i1h_w20_s1_p1.csv"  # путь к CSV-файлу
MODEL_SAVE_PATH = "trained_models/model_price_1.pkl"
WINDOW_SIZE = 20  # количество шагов (step0 ... step19)


# === Загрузка и подготовка данных ===
def parse_step(step_str):
    return list(map(float, step_str.split('|')))


def load_data(filepath):
    df = pd.read_csv(filepath)[:100]

    X = []
    y = []

    for _, row in df.iterrows():
        features = []
        for i in range(WINDOW_SIZE):
            step_col = f"step{i}"
            features.extend(parse_step(row[step_col]))
        X.append(features)
        y.append(float(row["target"]))

    return np.array(X), np.array(y)


# === Обучение ===
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("📊 Метрики модели:")
    print(f"▶ MSE:  {mean_squared_error(y_test, y_pred):.6f}")
    print(f"▶ RMSE: {mean_squared_error(y_test, y_pred):.6f}")
    print(f"▶ R²:   {r2_score(y_test, y_pred):.4f}")
    print(f"▶ MAE:  {mean_absolute_error(y_test, y_pred):.6f}")

    return model


# === Основной код ===
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Файл не найден: {DATA_PATH}")
    else:
        X, y = load_data(DATA_PATH)
        model = train_model(X, y)

        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"✅ Модель сохранена в: {MODEL_SAVE_PATH}")
