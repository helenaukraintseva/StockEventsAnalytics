import pandas as pd

# Путь к исходному файлу CSV
input_csv = "input.csv"
# Путь к новому файлу
output_csv = "output.csv"

# Считываем данные
df = pd.read_csv(input_csv)

# Меняем значения в столбце target
df["target"] = df["target"].map({-1: 1, 0: 2, 1: 3})

# Сохраняем в новый файл
df.to_csv(output_csv, index=False)

print(f"Файл '{output_csv}' успешно создан.")
