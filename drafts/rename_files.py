import os


def rename_files_in_directory(directory):
    # Проверяем, существует ли указанная директория
    if not os.path.exists(directory):
        print("Указанная директория не существует.")
        return

    # Проходим по всем файлам и папкам в директории
    for filename in os.listdir(directory):
        # Полный путь к файлу
        old_file_path = os.path.join(directory, filename)

        # Проверяем, что это файл (а не папка)
        if os.path.isfile(old_file_path):
            # Создаем новое имя файла
            new_filename = f"{filename.split('.')[0]}.csv"
            new_file_path = os.path.join(directory, new_filename)

            # Переименовываем файл
            os.rename(old_file_path, new_file_path)
            print(f"Переименован: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    rename_files_in_directory("crypto_data")