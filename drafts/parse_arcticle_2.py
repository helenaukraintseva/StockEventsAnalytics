from bs4 import BeautifulSoup
import requests
import os
import time

# url = 'https://vc.ru/u/1510590-python-idea/687695-parser-vcru'
# url = 'https://vc.ru/crypto/1545327-150-telegram-kanalov-i-grupp-pro-kriptovalyuty-i-kripto-treiding'
# url = 'https://vc.ru/crypto/1903157-gromkie-sluchai-moshennichestva-s-mem-tokenami'


def parse(url: str):
    response = requests.get(url)
    status = response.status_code
    if status == 200:
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            # print(soup)
            title = str(soup.find('h1', class_='content-title'))
            title = title.split(">")[1].split("<")[0].strip()
            codes = soup.find_all('figure')
            text = ""
            for code in codes:
                text += f"{code.get_text(strip=True)}\n"
            return title, text
        except Exception as ex:
            print(ex)
            return False, False
    else:
        return False, False


def open_del_articles():
    with open("del_articles.txt", "r", encoding="utf-8") as file:
        data = [int(elem) for elem in file.read().split(", ") if len(elem) > 0]
        return data


def write_del_article(num):
    with open("del_articles.txt", "a", encoding="utf-8") as file:
        file.write(f"{num}, ")


def parse_arcticles():
    del_articles = open_del_articles()
    print(del_articles)
    base_url = "https://vc.ru/crypto/"
    counter = 100000
    while True:
        counter += 1
        if counter == 5_000_000:
            break
        if counter in del_articles:
            print("This is deleted article.")
            continue
        filename = f"articles/article_{counter}.txt"
        if os.path.exists(filename):
            print(f"File {filename} is ALREADY done.")
            continue
        url = f"{base_url}{counter}"
        title, text = parse(url=url)
        time.sleep(3)
        if title == "Статья удалена":
            print(title)
            write_del_article(counter)
            continue
        if title:
            text = f"<<<{title}>>>\n{text}"
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"File {filename} is done.")
        else:
            print("some error")
            write_del_article(counter)


if __name__ == "__main__":
    parse_arcticles()