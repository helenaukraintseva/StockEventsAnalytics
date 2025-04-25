import pandas as pd
import os
import glob

# Папка, где находятся CSV файлы
directory = 'datasets'  # Замените на свой путь

# Находим все CSV файлы, начинающиеся с 'l'
csv_files = glob.glob(os.path.join(directory, 'cl*.csv'))

# Обрабатываем каждый файл
for file in csv_files:
    # Читаем CSV файл
    df = pd.read_csv(file)

    # Проверяем, есть ли столбец 'target'
    if 'target' in df.columns:
        # Меняем значения в столбце 'target'
        df['target'] = df['target'].replace({-1: 0, 0: 1, 1: 2})

        # Сохраняем изменения в тот же файл
        df.to_csv(file, index=False)
        print(f'Обработан файл: {file}')
    else:
        print(f'Столбец "target" не найден в файле: {file}')
