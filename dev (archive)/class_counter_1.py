import pandas as pd

# Замените 'your_file.csv' на путь к вашему CSV файлу
file_path = 'datasets/cl_0a1_i1m_w20_s5_p1.csv'

# Читаем CSV файл
data = pd.read_csv(file_path)

# Проверяем, существует ли колонка 'target'
if 'target' in data.columns:
    # Считаем количество повторений каждого уникального значения в колонке 'target'
    value_counts = data['target'].value_counts()

    # Выводим результаты
    print(value_counts)
else:
    print("Колонка 'target' не найдена в файле.")
