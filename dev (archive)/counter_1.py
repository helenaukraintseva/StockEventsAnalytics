import pandas as pd


def count_rows_in_csv(file_path):
    try:
        # Загружаем CSV файл в DataFrame
        df = pd.read_csv(file_path)

        # Получаем количество строк
        row_count = len(df)
        count_1 = len(df[df["is_crypto"] == 1])
        count_2 = len(df[df["is_crypto"] == 0])

        print(f"Количество строк в файле '{file_path}': {row_count}")
        print(f"Крипта: {count_1}")
        print(f"Не крипта: {count_2}")
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
    except pd.errors.EmptyDataError:
        print(f"Файл '{file_path}' пуст.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Пример использования функции
file_path = 'crypto_news_total.csv'  # Укажите путь к вашему CSV файлу
count_rows_in_csv(file_path)
