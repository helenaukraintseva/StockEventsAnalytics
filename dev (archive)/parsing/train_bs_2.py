import os
import joblib
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from crypto_monitoring.src.test_forex_2 import TimeSeriesSimulator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = os.getenv("SIGNAL_TRAIN_CSV", "datasets/cl_0a1_i1h_w20_s10_p1.csv")
TEST_DATA = os.getenv("SIGNAL_TEST_CSV", "crypto_data/1000BONKUSDC_1m.csv")
SAVE_DIR = os.getenv("SIGNAL_MODEL_DIR", "trained_signal_models_2")
os.makedirs(SAVE_DIR, exist_ok=True)


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


if __name__ == "__main__":
    X, y = load_and_prepare_data(DATA_PATH)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_signal.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "AdaBoost": AdaBoostClassifier(),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    for name, model in models.items():
        logging.info(f"🔧 Обучаем модель: {name}")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(SAVE_DIR, f"{name}_signal.pkl"))

        y_pred = model.predict(X_test)
        logging.info("📊 Результаты на тесте (%s):\n%s", name, classification_report(y_test, y_pred, target_names=["hold", "buy", "sell"]))

        if os.path.exists(TEST_DATA):
            test_df = pd.read_csv(TEST_DATA)
            test_df.columns = test_df.columns.str.strip().str.lower()
            wrapper = ClassifierSignalModel(
                model_path=os.path.join(SAVE_DIR, f"{name}_signal.pkl"),
                scaler_path=os.path.join(SAVE_DIR, "scaler_signal.pkl")
            )
            simulator = TimeSeriesSimulator(test_df, time_column="open_time")
            simulator.run([wrapper])
            result = simulator.evaluate()
            logging.info("📈 Симуляция для %s:\n%s", name, result)
        else:
            logging.warning("⚠️ Тестовые данные не найдены: %s", TEST_DATA)
