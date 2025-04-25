import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_PATH = "datasets/re_0a1_i1h_w20_s1_p1.csv"
SAVE_DIR = "signal_models_1"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW = 20
LOOKAHEAD = 5
THRESHOLD = 0.003  # 0.3% вверх/вниз

def parse_step(step_str):
    return list(map(float, step_str.split('|')))

def create_dataset(df):
    X, y = [], []
    for i in tqdm(range(len(df) - WINDOW - LOOKAHEAD)):
        features = []
        for j in range(WINDOW):
            step = parse_step(df.iloc[i + j])
            features.extend(step)

        current_close = parse_step(df.iloc[i + WINDOW - 1])[1]
        future_closes = [parse_step(df.iloc[i + WINDOW + k])[1] for k in range(LOOKAHEAD)]
        future_mean = np.mean(future_closes)
        pct_change = (future_mean - current_close) / current_close

        if pct_change > THRESHOLD:
            label = 1  # buy
        elif pct_change < -THRESHOLD:
            label = 2  # sell
        else:
            label = 0  # hold

        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# === Загрузка и подготовка ===
df = pd.read_csv(DATA_PATH).filter(regex='step\d+$')  # только stepN
df_flat = df.apply(lambda row: ','.join(row.dropna().values.astype(str)), axis=1)

X, y = create_dataset(df_flat)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Обучение моделей ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SAVE_DIR, f"{name}.pkl"))
    y_pred = model.predict(X_test)
    print(f"📊 {name}:\n", classification_report(y_test, y_pred, target_names=["hold", "buy", "sell"]))
