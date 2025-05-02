import pandas as pd

# --- 1. Загрузка CSV-файла ---
df = pd.read_csv("crypto_news_total.csv")  # укажи свой путь к файлу

# --- 2. Укажи имя колонки для подсчёта уникальных значений ---
column_name = "felt"  # замени на нужную колонку

# --- 3. Подсчёт повторений ---
value_counts = df[column_name].value_counts()

# --- 4. Вывод результата ---
print(f"\n🔍 Частота значений в колонке '{column_name}':\n")
print(value_counts)

