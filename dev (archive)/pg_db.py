import psycopg2
from config import DB_PAR_1

# Замените следующие параметры на свои


class PostgresClient:
    def __init__(self, db_name: str, user: str, password: str, host: str, port: str):
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = self.connect()

    def connect(self):
        try:
            connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Подключено успешно.")
            return connection
        except Exception:
            print("Произошла ошибка.")

    def get_data(self, request: str):
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute(request)
            result = cursor.fetchall()
            cursor.close()
            self.close_db()
            return result
        except Exception as ex:
            print(f"Some error: {ex}")

    def set_data(self, request: str):
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(request)
            self.conn.commit()
            cursor.close()
            self.close_db()
            print("ok")
        except Exception as ex:
            print(f"Some error: {ex}")

    def close_db(self):
        self.conn.close()
