import csv

def read_csv_and_write_to_lines(input_csv_file, output_text_file):
    """
    Reads a CSV file and writes each element to a new line in a text file.

    Parameters:
    input_csv_file (str): The path to the input CSV file.
    output_text_file (str): The path to the output text file.
    """
    try:
        new_rows = list()
        with open(input_csv_file, encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            print(list(csvreader)[1:])
            for row in csvreader:
                new_rows.append(row)
        print(new_rows)
        print(f"Data from '{input_csv_file}' has been written to '{output_text_file}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
read_csv_and_write_to_lines("channels_content_2024_1_1/if_crypto_ru.csv", 'output.txt')

