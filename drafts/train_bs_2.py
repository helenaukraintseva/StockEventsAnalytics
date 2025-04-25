import pandas as pd
import numpy as np
import os
import time
import joblib
from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from test_forex_2 import TimeSeriesSimulator

# === Пути ===
DATA_PATH = "datasets/cl_0a1_i1h_w20_s10_p1.csv"
SAVE_DIR = "trained_signal_models_2"
TEST_DATA = "crypto_data/1000BONKUSDC_1m.csv"  # ← сюда подставь свой реальный CSV с колонками time, open, close, high, low, volume

os.makedirs(SAVE_DIR, exist_ok=True)

# === Загрузка и подготовка ===
def parse_step(step_str):
    return list(map(float, step_str.split('|')))

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    step_columns = [col for col in df.columns if col.startswith("step")]
    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        features = []
        for col in step_columns:
            features.extend(parse_step(row[col]))
        X.append(features)
        y.append(int(row["target"]))
    return np.array(X), np.array(y)

X, y = load_and_prepare_data(DATA_PATH)

# === Нормализация ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_signal.pkl"))

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === Модели ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# === Обучение моделей и тестирование ===
class ClassifierSignalModel:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def analyze(self, history_df):
        if len(history_df) < 20:
            return {"signal": "hold"}
        features = []
        for _, row in history_df.iterrows():
            values = [row["open"], row["close"], row["high"], row["low"], row["volume"]]
            features.extend(values)
        X = self.scaler.transform([features])
        pred = self.model.predict(X)[0]
        return {"signal": {0: "hold", 1: "buy", 2: "sell"}[pred]}

# === Обучение и симуляция ===
for name, model in models.items():
    print(f"\n🔧 Обучаем {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SAVE_DIR, f"{name}_signal.pkl"))

    y_pred = model.predict(X_test)
    print(f"\n📊 Классификация ({name}):")
    print(classification_report(y_test, y_pred, target_names=["hold", "buy", "sell"]))

    # Тестирование через симулятор
    if os.path.exists(TEST_DATA):
        print(f"\n🚦 Запускаем симуляцию с {name}...")
        test_df = pd.read_csv(TEST_DATA)
        test_df.columns = test_df.columns.str.strip().str.lower()

        model_wrapper = ClassifierSignalModel(
            model_path=os.path.join(SAVE_DIR, f"{name}_signal.pkl"),
            scaler_path=os.path.join(SAVE_DIR, "scaler_signal.pkl")
        )

        simulator = TimeSeriesSimulator(test_df, time_column="open_time")
        simulator.run([model_wrapper])
        result = simulator.evaluate()
        print(f"\n📈 Оценка симуляции для {name}:")
        print(result)
    else:
        print(f"⚠️ Тестовые данные не найдены по пути: {TEST_DATA}")
