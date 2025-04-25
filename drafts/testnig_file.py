# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Для сохранения и загрузки модели

# Загрузка данных
data = pd.read_csv('data.csv')  # Замените 'data.csv' на путь к вашим данным

# Предполагаем, что у вас есть целевая переменная 'target' и признаки 'features'
X = data.drop('target', axis=1)  # Признаки
y = data['target']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

x_real, y_real = func()
model.fit(x_real, y_train)


# Сохранение обученной модели
joblib.dump(model, 'random_forest_model.pkl')

# Загрузка обученной модели
loaded_model = joblib.load('random_forest_model.pkl')

# Использование загруженной модели для предсказания
new_predictions = loaded_model.predict(X_test)
