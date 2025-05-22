from locust import HttpUser, task, between

class CryptoAppUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_main(self):
        self.client.get("/")

    @task
    def load_graph(self):
        self.client.get("/?tab=📈%20График%20криптовалют")

    @task
    def load_prediction(self):
        self.client.get("/?tab=📈%20Предсказать%20направление")

    @task
    def load_signals(self):
        self.client.get("/?tab=📊%20Сигналы")

    @task
    def load_news(self):
        self.client.get("/?tab=📊%20Анализ%20новостей")
