from google.generativeai import list_models
import google.generativeai as genai
from config import gemini_api

genai.configure(api_key=gemini_api)
models = list_models()
for m in models:
    print(m.name)