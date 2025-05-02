import pandas as pd


def add_column_to_csv(file_path, new_column_name, new_value):
    # Загружаем существующий CSV файл
    try:
        existing_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Файл не найден. Создаем новый DataFrame.")
        existing_df = pd.DataFrame()

    # Добавляем новую колонку с одинаковыми значениями
    existing_df[new_column_name] = [new_value] * len(existing_df)

    # Сохраняем обновленный DataFrame обратно в CSV файл
    existing_df.to_csv(file_path.split("/")[1], index=False)
    print(
        f"Файл '{file_path}' успешно обновлен. Добавлена колонка '{new_column_name}' с одинаковыми значениями '{new_value}'.")


# Пример использования функции
file_path = "channels_content_2024_1_1/crypnews247.csv"
new_column_name = 'title'
new_value = 'ConstantValue'  # Значение для новой колонки

add_column_to_csv(file_path, new_column_name, new_value)
