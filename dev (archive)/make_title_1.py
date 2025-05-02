from transformers import pipeline
import time

time_start = time.time()

# Используем модель для генерации заголовков
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def generate_title(text):
    summary = summarizer(text, max_length=15, min_length=5, do_sample=False)
    return summary[0]['summary_text']

print(f"Total time: {round(time.time() - time_start, 2)}")
text = "Биткоин снова преодолел отметку в $70,000, побив все предыдущие рекорды. Эксперты связывают это с ростом IT сферы."
print("Предложенный заголовок:", generate_title(text))
