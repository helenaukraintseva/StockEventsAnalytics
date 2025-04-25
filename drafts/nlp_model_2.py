from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import joblib
# nltk.download("stopwords")
#
# russian_stopwords = stopwords.words("russian")
pred_model = joblib.load('crypto_classifier_model.pkl')

model = Pipeline([
    ("clf", pred_model),
])

ex_1 = ["""азработчик блокчейна Картик Патель представил Anon World, социальную платформу, основанную на протоколе Farcaster, которая гарантирует анонимность и приватность пользователей. Платформа предлагает интерфейс в стиле Reddit, способствующий удобному общению и защите данных.

🔒 Децентрализованный Farcaster в основе безопасности  
Farcaster — это протокол, обеспечивающий:  
- Децентрализацию данных, исключающую контроль одной компании.  
- Защиту от взломов и манипуляций через распределенное хранение.  
- Отсутствие цензуры и внешнего контроля над контентом.  

✨ AnonCast: от подкаста к социальной платформе  
Запуск Anon World стал следующим шагом в развитии проекта AnonCast, известного своей приверженностью анонимности и свободе слова. Платформа станет убежищем для пользователей, ценящих свою приватность.
"""]

ex_2 = ["""Кремль надеется завершить переговоры по Украине до 9 мая, когда будет отмечаться 80-я годовщина победы Советского Союза над нацистской Германией. Цель — устроить двойной праздник, — корреспондент Sky News в России"""]

print("Пример 1:", model.predict(ex_1)[0])
print("Пример 2:", model.predict(ex_2)[0])
