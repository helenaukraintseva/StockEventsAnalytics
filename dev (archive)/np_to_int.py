import os
import pandas as pd
import ast
import re
import numpy as np

# Папка с файлами
folder_path = 'datasets'
target_column = 'target'

# Функция для преобразования значений в target
def clean_target(val):
    try:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            # Попробуем извлечь число через регулярку
            match = re.search(r'\d+', val)
            if match:
                return int(match.group(0))
        if isinstance(val, (list, tuple)):
            return int(val[0])
        # Пробуем распарсить строку как Python объект
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return int(parsed[0])
        return int(parsed)
    except Exception as e:
        print(f"⚠ Ошибка при обработке значения '{val}': {e}")
        return None

# Проход по всем CSV
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"🔍 Обработка: {filename}")

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            print(f"⚠ Пропущено: Нет колонки '{target_column}'")
            continue

        # Преобразование значений
        df[target_column] = df[target_column].apply(clean_target)

        # Проверка результатов
        if df[target_column].isnull().any():
            print(f"⚠ Обнаружены пустые значения после конверсии в файле {filename}")
            continue

        # Сохранение
        df.to_csv(file_path, index=False)
        print(f"✅ Сохранено: {filename}\n")
