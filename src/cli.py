import click

@click.command()
@click.option("--parse-news", is_flag=True, help="Запуск парсинга новостей.")
@click.option("--train-model", is_flag=True, help="Запуск обучения модели.")
def main(parse_news, train_model):
    if parse_news:
        from parsing_news.telegram_4 import parse_telegram_news
        parse_telegram_news()
    if train_model:
        # запустить обучение
        pass

if __name__ == "__main__":
    main()