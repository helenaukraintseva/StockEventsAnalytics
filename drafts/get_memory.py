import os


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Проверяем, что файл существует, чтобы избежать ошибок
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


folders_path = ["parsing_news/channels_content_2024_1_1/",
                "parse_article/articles",
                "parsing/crypto_data/"]
for path in folders_path:
    size = get_folder_size(path)
    print(f"Объем памяти, занимаемый папкой: {size} байт")