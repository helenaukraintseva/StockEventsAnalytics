import os
import pandas as pd

# Папка с CSV-файлами (укажи путь к своей)
folder_path = 'datasets'  # например, './data' или '.' для текущей директории

# Имя столбца с целевыми метками (можно изменить, если отличается)
target_column = 'target'

# Максимум элементов на класс
max_per_class = 20000

# Обрабатываем все файлы с расширением .csv
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"🔍 Обработка файла: {filename}")

        # Загружаем CSV
        df = pd.read_csv(file_path)
        class_counts = df[target_column].value_counts()

        # Проверяем наличие нужного столбца
        if target_column not in df.columns:
            print(f"⚠ Пропущено: Нет столбца '{target_column}'")
            continue

        # Проверим, что все значения — из нужных классов
        unique_classes = set(df[target_column].unique())
        expected_classes = {1, 0, 2}

        if not unique_classes.issubset(expected_classes):
            print(f"⚠ Пропущено: Найдены неожиданные значения классов — {unique_classes}")
            continue

        # Обрезаем каждый класс до 1000 элементов (если нужно)
        balanced_dfs = []
        for cls in sorted(expected_classes):
            cls_df = df[df[target_column] == cls]
            if len(cls_df) > max_per_class:
                cls_df = cls_df.sample(n=max_per_class, random_state=42)
            balanced_dfs.append(cls_df)

        # Собираем сбалансированный датасет
        balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)  # перемешиваем
        class_counts = balanced_df[target_column].value_counts()

        # Сохраняем обратно в тот же файл
        balanced_df.to_csv(file_path, index=False)
        print(f"✅ Сохранено: {filename} (размер: {len(balanced_df)} строк)\n")
