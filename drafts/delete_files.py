import os

# Укажите директорию, в которой находятся ваши CSV файлы
directory = 'crypto_data'  # Замените на путь к вашей директории

# Список элементов для проверки в названии файла
elements_to_remove = ["BCHUSDC"]  # Замените на ваши элементы

# Проходим по всем файлам в директории
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Проверяем, содержит ли имя файла любой элемент из списка
        if any(element in filename for element in elements_to_remove):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)  # Удаляем файл
            print(f'Файл {filename} был удален.')

print('Удаление завершено.')
