import pandas as pd

# Укажите путь к исходному CSV файлу
input_file_path = 'test_data_2.csv'  # замените на ваш файл
# Укажите путь к выходному CSV файлу
output_file_path = 'test_data_2.csv'  # замените на желаемый файл

data = pd.read_csv(input_file_path)[:5]
print(data.head())
data.to_csv(output_file_path, index=False)
