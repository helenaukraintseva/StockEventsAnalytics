import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# === Логирование ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Настройки ===
WINDOW_SIZE = 20
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
TARGET_SHIFT = 1

# === Загрузка данных ===
data = pd.read_csv("parsing/1000BONKUSDC_1m.csv")
features = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol']

# === Создание целевой переменной ===
data['target'] = (data['close'].shift(-TARGET_SHIFT) > data['close']).astype(int)
data.dropna(inplace=True)

# === Масштабирование ===
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# === Формирование окон ===
def create_sequences(data, target_col, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[features].iloc[i:i+window].values)
        y.append(data[target_col].iloc[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(data, 'target', WINDOW_SIZE)

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Torch Dataset ===
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Модели ===
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, window_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size * window_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# === Выбор модели ===
model_type = "gru"  # можно выбрать: 'gru' или 'mlp'
if model_type == "gru":
    model = GRUClassifier(input_size=X.shape[2])
elif model_type == "mlp":
    model = MLPClassifier(input_size=X.shape[2], window_size=WINDOW_SIZE)
else:
    raise ValueError("Неподдерживаемый тип модели")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === MLflow Logging ===
mlflow.set_experiment(f"Crypto_{model_type.upper()}")

with mlflow.start_run():
    mlflow.log_param("window_size", WINDOW_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("model_type", model_type)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_loss += loss.item()
                predicted = (preds > 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        logger.info(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={acc:.4f}")
        mlflow.log_metric("val_accuracy", acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    mlflow.pytorch.log_model(model, "model")
    logger.info("✅ Обучение завершено и залогировано в MLflow")