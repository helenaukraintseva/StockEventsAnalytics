from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import pandas as pd


def scrape_x_posts(keyword, max_posts=20):
    # Настройка браузера
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Открываем X
    search_url = f"https://twitter.com/search?q={keyword}&src=typed_query&f=live"
    driver.get(search_url)

    time.sleep(5)  # Даем странице прогрузиться

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
                print("Ошибка при парсинге поста:", e)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    driver.quit()
    return pd.DataFrame(data)


# 🔍 Пример запуска
if __name__ == "__main__":
    df = scrape_x_posts("искусственный интеллект", max_posts=10)
    print(df)
    df.to_csv("x_parsed_custom.csv", index=False)
