import pandas as pd
import os

# Укажите путь к директории с файлами .csv
directory_path = 'datasets'


# Словарь для хранения результатов
def balance_classes(df,
                    class_column,
                    file_path,
                    limit_value=2000):
    # Подсчет количества элементов каждого класса
    total_text = ""
    class_counts = df[class_column].value_counts()
    print(class_counts)

    # Наименьшее количество элементов среди классов
    min_count = class_counts.min()

    # Создание списка для хранения сбалансированных данных
    balanced_data = []

    # Для каждого класса выбираем min_count случайных элементов
    if len(class_counts.index) != 3 or min_count < limit_value:
        os.remove(file_path)
        text1, text2 = "", ""
        if len(class_counts.index) != 3:
            text1 = "\nHaven't classes enough"
        if min_count < limit_value:
            text2 = "\nHaven't values enough (<2000)"
        total_text += f"\n{file_path}\n{class_counts}{text1}{text2}\n"
        append_to_file(filename=file_name, additional_info=total_text)
        print(f"File {file_path} was deleted because: {text1} {text2}")
    for class_label in class_counts.index:
        class_data = df[df[class_column] == class_label]
        balanced_data.append(class_data.sample(min_count, random_state=42))

    # Объединяем все сбалансированные данные в один DataFrame
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    x_full = balanced_df.drop(["target"], axis=1)
    y_full = balanced_df["target"]
    class_counts = balanced_df[class_column].value_counts()
    # print(class_counts)
    print(file_path)
    return x_full, y_full


def create_file(filename):
    """Создает текстовый файл и записывает начальную информацию."""
    with open(filename, 'w') as file:
        file.write('Это начальная информация.\n')
    print(f'Файл "{filename}" создан с начальной информацией.')


def append_to_file(filename, additional_info):
    """Дополняет текстовый файл новой информацией."""
    with open(filename, 'a') as file:
        file.write(additional_info + '\n')
    print(f'Дополнена информация: "{additional_info}"')


if __name__ == "__main__":
    file_name = "result_check_datasets.txt"
    # create_file(file_name)
    # Проходим по всем файлам в указанной директории
    for filename in os.listdir(directory_path):
        target_counts = {}
        if filename.endswith('.csv') and filename.startswith("cl"):
            file_path = os.path.join(directory_path, filename)
            # Читаем CSV файл
            df = pd.read_csv(file_path)
            balance_classes(df, class_column="target", file_path=file_path)

            # Проверяем, существует ли столбец 'target'
            if 'target' in df.columns:
                # Считаем количество каждого уникального значения в столбце 'target'
                counts = df['target'].value_counts()

                # Обновляем общий словарь с результатами
                for value, count in counts.items():
                    if value in target_counts:
                        target_counts[value] += count
                    else:
                        target_counts[value] = count
        print("_________________________________")
        # print(filename)
        # for value, count in target_counts.items():
        #     print(f"Значение: {value}, Количество: {count}")

    # Выводим результаты
