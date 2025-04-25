import pandas as pd
import os

# Укажите директорию, в которой находятся ваши CSV файлы
directory = 'crypto_data'  # Замените на путь к вашей директории

# Словарь для хранения групп файлов
file_groups = {
    '0-1': [],
    '1-100':  [],
    '100-1000': [],
    '1000+': []
}

# Обработка всех CSV файлов в указанной директории
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)

        # Загружаем данные из CSV файла
        data = pd.read_csv(file_path)

        # Проверяем наличие колонки 'Close'
        if 'close' in data.columns:
            # Вычисляем среднюю цену закрытия
            average_close = data['close'].mean()
            print(f'Файл: {filename}, Средняя цена закрытия: {average_close}')

            # Определяем группу на основе средней цены закрытия
            if average_close < 1:
                file_groups['0-1'].append(filename)
            elif average_close < 10:
                file_groups['1-10'].append(filename)
            elif average_close < 100:
                file_groups['10-100'].append(filename)
            else:
                file_groups['1000+'].append(filename)
        else:
            print(f'В файле {filename} отсутствует колонка "Close".')

# Выводим результаты группировки
for group, files in file_groups.items():
    print(f'Группа {group}: {files}')
    print(len(files))

# Если нужно, можно сохранить результаты в отдельный файл
with open('grouped_files.txt', 'w') as f:
    for group, files in file_groups.items():
        f.write(f'Группа {group}: {files}\n')
