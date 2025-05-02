import os
import csv
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_csv_and_write_to_lines(input_csv_file: str, output_text_file: str):
    """
    Читает CSV-файл и записывает каждый элемент в новую строку текстового файла.

    :param input_csv_file: Путь к входному CSV
    :param output_text_file: Путь к выходному TXT
    """
    try:
        with open(input_csv_file, encoding="utf-8") as csvfile, \
             open(output_text_file, "w", encoding="utf-8") as txtfile:

            csvreader = csv.reader(csvfile)
            headers = next(csvreader, None)  # пропускаем заголовки

            for row in csvreader:
                for cell in row:
                    txtfile.write(f"{cell.strip()}\n")

        logging.info("Данные из '%s' успешно записаны в '%s'.", input_csv_file, output_text_file)
    except Exception as e:
        logging.error("Произошла ошибка: %s", e)


if __name__ == "__main__":
    input_file = os.getenv("READER_INPUT_CSV", "channels_content_2024_1_1/if_crypto_ru.csv")
    output_file = os.getenv("READER_OUTPUT_TXT", "output.txt")
    read_csv_and_write_to_lines(input_file, output_file)
