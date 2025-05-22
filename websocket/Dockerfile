FROM python:3.10-slim

# Установка зависимостей
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "websocket_test_2.py"]
