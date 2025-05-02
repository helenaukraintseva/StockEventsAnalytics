import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GEMINI_API = os.getenv("GEMINI_API_KEY")

def list_gemini_models(api_key: str):
    """
    Получает список доступных моделей Gemini API.

    :param api_key: API ключ Google Generative AI
    :return: Список имён моделей
    """
    genai.configure(api_key=api_key)
    models = genai.list_models()
    model_names = [m.name for m in models]
    return model_names


if __name__ == "__main__":
    if not GEMINI_API:
        logging.error("❌ Переменная окружения GEMINI_API_KEY не найдена.")
    else:
        models = list_gemini_models(GEMINI_API)
        logging.info("📦 Доступные модели Gemini:")
        for name in models:
            print(f" - {name}")
