import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from crypto_monitoring.src.test_forex_2 import TimeSeriesSimulator

DATA_PATH = "datasets/cl_0a1_i1m_w20_s5_p1.csv"
SAVE_DIR = "trained_signal_models_3"
TEST_DATA = "crypto_data/1000BONKUSDC_1m.csv"  # <- замените на путь к тестовому набору данных

os.makedirs(SAVE_DIR, exist_ok=True)

def parse_step(step_str):
    return list(map(float, step_str.split('|')))

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)[:50000]
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

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_signal.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === sklearn-модели ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

class ClassifierSignalModel:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def analyze(self, history_df):
        if len(history_df) < 20:
            return {"signal": "hold"}
        features = []
        for _, row in history_df.iloc[-20:].iterrows():
            values = [row["open"], row["close"], row["high"], row["low"], row["volume"]]
            features.extend(values)
        X = self.scaler.transform([features])
        pred = self.model.predict(X)[0]
        signal_map = {0: "hold", 1: "buy", 2: "sell"}
        return {"signal": signal_map.get(int(pred), "hold")}

for name, model in models.items():
    print(f"\\n🔧 Обучаем {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SAVE_DIR, f"{name}_signal.pkl"))

    y_pred = model.predict(X_test)
    print(f"\\n📊 Классификация ({name}):")
    print(classification_report(y_test, y_pred, target_names=["hold", "buy", "sell"]))

    if os.path.exists(TEST_DATA):
        print(f"\\n🚦 Симуляция {name}...")
        test_df = pd.read_csv(TEST_DATA)
        model_wrapper = ClassifierSignalModel(
            model_path=os.path.join(SAVE_DIR, f"{name}_signal.pkl"),
            scaler_path=os.path.join(SAVE_DIR, "scaler_signal.pkl")
        )
        simulator = TimeSeriesSimulator(test_df, time_column="open_time")
        simulator.run([model_wrapper])
        result = simulator.evaluate()
        df_save = pd.DataFrame(result)
        df_save.to_csv(f"{name}.csv")
        print(f"\\n📈 Оценка симуляции для {name}:")
        print(result)

# === PyTorch ===
class TorchRNNClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_classes=3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class TorchLSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TorchGRUClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_torch_model(model_class, name, X, y, window=20, input_size=5, epochs=20):
    print(f"\\n🧠 Обучаем {name} (PyTorch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X.reshape(-1, window, input_size), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = model_class(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_signal.pt"))

class TorchClassifierSignalModel:
    def __init__(self, model_path, model_class, scaler, input_size=5, window=20):
        self.model = model_class(input_size=input_size)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.scaler = scaler
        self.window = window
        self.input_size = input_size

    def analyze(self, history_df):
        if len(history_df) < self.window:
            return {"signal": "hold"}
        features = []
        for _, row in history_df.iloc[-self.window:].iterrows():
            features.extend([row["open"], row["close"], row["high"], row["low"], row["volume"]])
        x = np.array(features).reshape(1, self.window * self.input_size)
        x_scaled = self.scaler.transform(x).reshape(1, self.window, self.input_size)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)
            pred = torch.argmax(logits, dim=1).item()
        return {"signal": {0: "hold", 1: "buy", 2: "sell"}[pred]}

torch_models = {
    "TorchRNN": TorchRNNClassifier,
    "TorchLSTM": TorchLSTMClassifier,
    "TorchGRU": TorchGRUClassifier
}

for name, model_class in torch_models.items():
    train_torch_model(model_class, name, X_scaled, y)

    if os.path.exists(TEST_DATA):
        print(f"\\n🚦 Симуляция {name}...")
        test_df = pd.read_csv(TEST_DATA)
        model_wrapper = TorchClassifierSignalModel(
            model_path=os.path.join(SAVE_DIR, f"{name}_signal.pt"),
            model_class=model_class,
            scaler=scaler
        )
        simulator = TimeSeriesSimulator(test_df, time_column="open_time")
        simulator.run([model_wrapper])
        result = simulator.evaluate()
        df_save = pd.DataFrame(result)
        df_save.to_csv(f"{name}.csv")
        print(f"\\n📈 Оценка симуляции для {name}:")
        print(result)