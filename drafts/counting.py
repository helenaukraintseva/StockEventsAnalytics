import os
import pandas as pd


def count_csv_elements(directory):
    # Проходим по всем файлам в указанной директории
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                # Загружаем CSV файл в DataFrame
                df = pd.read_csv(file_path)

                # Получаем количество элементов (строк) в DataFrame
                element_count = len(df)

                print(f'Файл: {filename}, Количество элементов: {element_count}')
            except Exception as e:
                print(f'Ошибка при обработке файла {filename}: {e}')


# Замените 'your_directory_path' на путь к вашей директории с файлами .csv
count_csv_elements('datasets')
