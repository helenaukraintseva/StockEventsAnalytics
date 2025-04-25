import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Загружаем данные
file_path = "data_news.csv"
df = pd.read_csv(file_path)

# Удаляем ненужные столбцы
df = df[['title', 'summary', 'score']]
df.dropna(inplace=True)

# Преобразуем score в числовой формат
df['score'] = df['score'].astype(float)

# Объединяем заголовок и текст новости в одну строку
df['text'] = df['title'] + " " + df['summary']

# Векторизация текста с помощью TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['score'].values

# Разделяем данные на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Преобразуем данные в формат PyTorch
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Определяем нейросеть
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Линейный выход (регрессия)
        return x


# Инициализация модели
input_dim = X_train.shape[1]
model = RegressionModel(input_dim)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()  # Для задачи регрессии
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Оценка модели
model.eval()
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)

        y_pred_list.extend(y_pred.numpy().flatten())
        y_true_list.extend(y_batch.numpy().flatten())
    print(vectorizer.inverse_transform([X_test[0]]))
    print(y_pred[0])

# Выводим среднюю ошибку модели
mse = np.mean((np.array(y_pred_list) - np.array(y_true_list)) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.4f}")
