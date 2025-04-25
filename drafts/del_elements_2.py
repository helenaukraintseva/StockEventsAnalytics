import os
import pandas as pd
import re

# Папка с CSV-файлами
folder_path = 'datasets'

# Имя столбца с целевыми метками
target_column = 'target'

# Пределы на количество элементов в каждом классе
max_per_class = 20000
min_per_class = 8000

# Обрабатываем все файлы в папке
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"🔍 Обработка файла: {filename}")

        df = pd.read_csv(file_path)


        # Пример для конкретного столбца
        df['some_column'] = df['some_column'].astype(str).apply(
            lambda x: float(re.findall(r'[\d\.E+-]+', x)[0]) if 'np.float64' in x else float(x))
        print("📏 Размер:", df.shape)
        print("📊 Колонки:", df.columns.tolist())
        print("🔎 Типы данных:")
        print(df.dtypes)
        print("🔍 Пример данных:")
        print(df.head(3))

        #
        # # Проверка на наличие целевого столбца
        # if target_column not in df.columns:
        #     print(f"⚠ Пропущено: Нет столбца '{target_column}'")
        #     continue
        #
        # # Уникальные классы и их количества
        # class_counts = df[target_column].value_counts()
        # unique_classes = set(df[target_column].unique())
        # expected_classes = {0, 1, 2}
        #
        # if not unique_classes.issubset(expected_classes):
        #     print(f"⚠ Пропущено: Найдены неожиданные значения классов — {unique_classes}")
        #     continue
        #
        # # Удостоверимся, что все нужные классы присутствуют
        # if not expected_classes.issubset(class_counts.index):
        #     print(f"⚠ Пропущено: Не все нужные классы есть в данных — {class_counts.to_dict()}")
        #     continue
        #
        # # Определим максимально возможное количество для каждого класса в пределах лимита
        # possible_counts = [class_counts[cls] for cls in expected_classes]
        # min_class_count = min(possible_counts)
        # final_count = min(max(min_class_count, min_per_class), max_per_class)
        #
        # if final_count < min_per_class:
        #     print(f"⚠ Пропущено: Недостаточно данных для сбалансировки — минимум {min_class_count} < {min_per_class}")
        #     continue
        #
        # # Сбалансируем
        # balanced_dfs = []
        # for cls in sorted(expected_classes):
        #     cls_df = df[df[target_column] == cls]
        #     if len(cls_df) >= final_count:
        #         cls_df = cls_df.sample(n=final_count, random_state=42)
        #         balanced_dfs.append(cls_df)
        #     else:
        #         print(f"⚠ Класс {cls} имеет только {len(cls_df)} записей, требуется {final_count}. Пропускаем файл.")
        #         break
        # else:
        #     # Если все классы прошли, то сохраняем
        #     balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
        #     balanced_df.to_csv(file_path, index=False)
        #     print(f"✅ Сохранено: {filename} (по {final_count} строк на класс, всего {len(balanced_df)} строк)\n")
