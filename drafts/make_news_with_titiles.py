import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import glob
import csv
import re
import torch
import numpy as np

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

CRYPTO_KEYWORDS = [
    # Английские термины
    "crypto", "cryptocurrency", "digital currency", "virtual currency", "digital asset", "blockchain",
    "decentralized", "distributed ledger", "web3", "defi", "nft", "token", "altcoin", "stablecoin",
    "smart contract", "layer 1", "layer 2", "cross-chain", "sidechain", "rollup", "zk-rollup",

    "mining", "staking", "farming", "yield farming", "airdrop", "ico", "ido", "ieo", "burn", "mint",
    "swap", "bridge", "tokenomics", "liquidity pool", "hashrate", "consensus", "proof of stake",
    "proof of work", "delegation", "slashing", "governance", "gas fee", "wallet", "hot wallet", "cold wallet",

    "binance", "coinbase", "kraken", "kucoin", "okx", "bitfinex", "bybit", "gemini", "bitstamp",
    "pancakeswap", "uniswap", "sushiswap", "1inch", "dex", "cex", "exchange",

    "metamask", "trust wallet", "ledger", "trezor", "phantom", "keplr", "safe", "rainbow", "argent",

    "ethereum", "eth", "bitcoin", "btc", "solana", "sol", "polygon", "matic", "cardano", "ada",
    "polkadot", "dot", "avalanche", "avax", "arbitrum", "optimism", "near", "fantom", "ftm", "cosmos", "atom",
    "tron", "trx", "algorand", "algo", "hedera", "hbar", "internet computer", "icp", "filecoin", "fil",

    "dogecoin", "doge", "shiba", "shib", "pepe", "floki", "baby doge", "bonk", "wif",

    "usdt", "tether", "usdc", "dai", "busd", "frax", "tusd", "gusd",

    "satoshi", "halving", "moon", "pump", "dump", "rugpull", "whale", "hodl", "fomo", "fud",
    "rekt", "airdrops", "multisig", "dex aggregator", "liquidity mining", "gas", "gnosis", "zk-sync",
    "ens", "domain name", "naming service", "staking rewards", "on-chain", "off-chain",

    # Русские термины
    "криптовалюта", "крипта", "биткоин", "эфириум", "блокчейн", "токен", "смарт-контракт", "смартконтракт",
    "альткоин", "стейблкоин", "децентрализация", "дефи", "веб3", "цифровой актив", "цифровая валюта",
    "облачный майнинг", "майнинг", "стейкинг", "вознаграждение", "транзакция", "кошелёк", "кошелек",
    "холодный кошелёк", "горячий кошелёк", "биржа", "обменник", "перевод на кошелёк", "трейдинг", "сигналы",
    "аирдроп", "ICO", "IEO", "IDO", "листинг", "листинг на бирже", "сжигание токенов", "фарминг", "доходность",
    "сеть эфириум", "цепочка блоков", "децентрализованный", "гивевей", "раздача токенов", "новая монета",
    "альтсезон", "мемкоин", "шатдаун", "памп", "дамп", "взлом криптобиржи", "регулирование криптовалют",
    "санкции против криптобирж", "криптотрейдер", "портфель токенов", "газа комиссия", "блок", "цепочка",
    "сеть", "хардфорк", "софтфорк", "форк", "протокол", "децентрализованное финансирование",
    "приватный ключ", "публичный адрес", "блокчейн проект"
]


# def generate_title(text):
#     try:
#         if len(text.split()) < 5:  # или len(text) < 40
#             return "Без названия"  # слишком короткий текст
#         summary = summarizer(text, max_length=10, min_length=5, do_sample=False)
#         if len(summary) > 0:
#             return summary[0]['summary_text']
#         else:
#             return "Без названия"
#     except Exception as e:
#         print(f"⚠️ Ошибока при генерации заголовка: {e} | Текст: {text[:100]}...")
#         print(text)
#         return "Без названия"

def is_crypto_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in CRYPTO_KEYWORDS)


def generate_title(text):
    try:
        # Токенизируем и обрезаем по 1024 токена
        inputs = summarizer.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]
        decoded_text = summarizer.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if len(decoded_text.strip()) < 40:
            return "Без названия"

        result = summarizer(decoded_text, max_length=20, min_length=5, do_sample=False)
        if not result or "summary_text" not in result[0]:
            return "Без названия"

        return result[0]["summary_text"]

    except Exception as e:
        print(f"⚠️ Ошибка при генерации заголовка: {e} | Текст: {text[:100]}...")
        return "Без названия"


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    labels = ['Negative', 'Neutral', 'Positive']
    return labels[
        np.argmax(scores)], f"{round(float(scores[0]), 3)}_{round(float(scores[1]), 3)}_{round(float(scores[2]), 3)}"


def clean_text(text: str) -> str:
    # Удаляем всё, кроме русских/английских букв и цифр
    cleaned = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    # Удаляем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def process_multiple_news_csv(input_folder_pattern, output_path):
    # Список всех CSV файлов по шаблону (например, 'data/*.csv')
    all_files = glob.glob(input_folder_pattern)

    # combined_df = pd.DataFrame()
    new_docs = {"text": list(),
                "felt": list(),
                "felt_scores": list(),
                "is_crypto": list()}

    for file in all_files:
        print(f"Now working with {file}")
        with open(file, encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile))[1:]
        count = len(rows)
        counter = 1
        for elem in rows:
            print(f"{counter}/{count}")
            counter += 1
            if len(elem) > 2 and elem[2].strip():  # Проверим и длину, и наличие текста
                text = clean_text(elem[2])
                if len(text) > 100:
                    # title = generate_title(text)
                    is_crypto = int(is_crypto_related(text))
                    felt, felt_score = get_sentiment(text)
                    new_docs["is_crypto"].append(is_crypto)
                    new_docs["text"].append(text)
                    new_docs["felt"].append(felt)
                    new_docs["felt_scores"].append(felt_score)

        # Сохраняем объединённый результат
        df = pd.DataFrame(new_docs)
        try:
            existing_df = pd.read_csv(output_path)
            # Объединяем существующий DataFrame с новым
            df = pd.DataFrame(new_docs)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # Если файл не найден, создаем новый DataFrame
            updated_df = df
        # Объединяем существующий DataFrame с новым

        updated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"File {output_path} is ready.")


# Пример использования:
process_multiple_news_csv('channels_content_2024_1_1/*.csv', 'crypto_news_total.csv')
