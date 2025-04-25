import pandas as pd


class BaseModel:
    def analyze(self, history: pd.DataFrame) -> dict:
        """
        Получает историю (DataFrame) и возвращает решение.
        Возвращаемый словарь может содержать, например:
        {
            "signal": "buy" / "sell" / "hold",
            "confidence": float (0-1)
        }
        """
        raise NotImplementedError


class SimpleMovingAverageModel(BaseModel):
    def __init__(self, window=5):
        self.window = window

    def analyze(self, history: pd.DataFrame) -> dict:
        if len(history) < self.window:
            return {"signal": "hold", "confidence": 0.0}

        prices = history["price"]
        ma = prices.rolling(self.window).mean().iloc[-1]
        price = prices.iloc[-1]

        if price > ma:
            return {"signal": "buy", "confidence": 0.7}
        elif price < ma:
            return {"signal": "sell", "confidence": 0.7}
        else:
            return {"signal": "hold", "confidence": 0.5}

