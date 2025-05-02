import pandas as pd
import re

# Регулярное выражение:
# [^a-zA-Zа-яА-Я\s]+
# означает «вырезать всё, кроме английских букв (a-zA-Z), русских букв (а-яА-Я) и пробелов (\s)».
# Если пробелы не нужны, уберите \s.
pattern = re.compile(r'[^a-zA-Zа-яА-Я0-9\s]+')

# Чтение исходного CSV
input_csv = 'channel_posts.csv'        # <-- имя входного файла
output_csv = 'channel_posts_clean.csv' # <-- имя выходного файла

df = pd.read_csv(input_csv, encoding='utf-8')

# 1) Преобразуем столбец date в формат datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 2) Переводим дату в Unix timestamp (целое число).
# Если дата не распарсилась (NaT), можно оставить пустым или 0
df['date'] = df['date'].apply(lambda x: int(x.timestamp()) if not pd.isnull(x) else '')

# 3) Очищаем столбец text, оставляя только русские/английские буквы и пробелы
def clean_text(text):
    if pd.isnull(text):
        return ''
    return pattern.sub('', text)

df['text'] = df['text'].apply(clean_text)

# 4) Сохраняем результат
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Файл успешно сохранён как {output_csv}")
