import requests
from bs4 import BeautifulSoup
import logging
from time import sleep

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Заголовки для маскировки под браузер
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive'
}

url = 'https://vc.ru/crypto/1545327-150-telegram-kanalov-i-grupp-pro-kriptovalyuty-i-kripto-treiding'

def fetch_page(url):
    logging.info(f'Запрос к странице: {url}')
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logging.info('Страница успешно получена')
        return response.text
    except requests.RequestException as e:
        logging.error(f'Ошибка при запросе: {e}')
        return None

def parse_channels(html):
    soup = BeautifulSoup(html, 'html.parser')
    print(html)
    channels = soup.find_all('li')
    print(channels)
    logging.info(f'Найдено {len(channels)} пунктов в списке')

    parsed = []

    for idx, channel in enumerate(channels, start=1):
        name_tag = channel.find('strong')
        if name_tag:
            name = name_tag.text.strip()
            description = channel.text.replace(name, '').strip(' –')
            parsed.append((name, description))
            logging.info(f'[{idx}] Извлечён канал: {name}')
        else:
            logging.debug(f'[{idx}] Пропущен элемент без тега <strong>')

    return parsed


def main():
    logging.info("start of the program")
    html = fetch_page(url)
    if html:
        data = parse_channels(html)

        print('\n=== Результаты ===\n')
        for name, desc in data:
            print(f'Название: {name}')
            print(f'Описание: {desc}')


if __name__ == "__main__":
    main()
