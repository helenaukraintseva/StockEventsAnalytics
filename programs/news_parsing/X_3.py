import os
import time
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def scrape_x_posts(keyword: str, max_posts: int = 20) -> pd.DataFrame:
    """
    Парсит посты в X (Twitter) по ключевому слову.

    :param keyword: Ключевое слово или фраза для поиска
    :param max_posts: Максимум постов для извлечения
    :return: DataFrame с текстами постов
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    search_url = f"https://twitter.com/search?q={keyword}&src=typed_query&f=live"
    driver.get(search_url)

    time.sleep(5)
    posts = set()
    data = []

    while len(data) < max_posts:
        elements = driver.find_elements(By.XPATH, '//div[@data-testid="cellInnerDiv"]')

        for el in elements:
            try:
                content = el.text
                if content and content not in posts:
                    posts.add(content)
                    data.append({'text': content})
                    if len(data) >= max_posts:
                        break
            except Exception as e:
                logging.warning("Ошибка при парсинге поста: %s", e)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    driver.quit()
    return pd.DataFrame(data)


if __name__ == "__main__":
    keyword = os.getenv("X_SEARCH_KEYWORD", "искусственный интеллект")
    max_posts = int(os.getenv("X_MAX_POSTS", 10))

    df = scrape_x_posts(keyword, max_posts)
    df.to_csv("x_parsed_custom.csv", index=False)
    logging.info("Сохранено %d постов в 'x_parsed_custom.csv'", len(df))
