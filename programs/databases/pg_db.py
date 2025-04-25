import os
import logging
import psycopg2
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class PostgresClient:
    """
    Клиент для работы с PostgreSQL.
    Поддерживает подключение, выполнение запросов и автоматическое закрытие соединения.
    """

    def __init__(self, db_name: str, user: str, password: str, host: str, port: str):
        """
        :param db_name: Название базы данных
        :param user: Имя пользователя
        :param password: Пароль
        :param host: Адрес хоста базы данных
        :param port: Порт подключения к базе данных
        """
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = self.connect()

    def connect(self):
        """
        Устанавливает соединение с базой данных.

        :return: Объект подключения к базе данных или None при ошибке
        """
        try:
            connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            logging.info("Успешное подключение к базе данных.")
            return connection
        except Exception as ex:
            logging.error("Ошибка подключения к базе данных: %s", ex)

    def get_data(self, query: str):
        """
        Выполняет SELECT-запрос и возвращает результаты.

        :param query: SQL-запрос SELECT
        :return: Список строк с результатами запроса или None
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            logging.info("Запрос выполнен успешно: %s", query)
            return result
        except Exception as ex:
            logging.error("Ошибка при выполнении запроса: %s | %s", query, ex)
        finally:
            self.close_db()

    def set_data(self, query: str):
        """
        Выполняет INSERT, UPDATE или DELETE-запрос.

        :param query: SQL-запрос на изменение данных
        :return: None
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
            cursor.close()
            logging.info("Изменения успешно применены: %s", query)
        except Exception as ex:
            logging.error("Ошибка при выполнении запроса: %s | %s", query, ex)
        finally:
            self.close_db()

    def close_db(self):
        """
        Закрывает текущее соединение с базой данных.

        :return: None
        """
        if self.conn and not self.conn.closed:
            self.conn.close()
            logging.info("Соединение с базой данных закрыто.")


# Получение параметров подключения из .env
if __name__ == "__main__":
    db_client = PostgresClient(
        db_name=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    # Пример использования:
    # result = db_client.get_data("SELECT * FROM table_name;")
    # print(result)
