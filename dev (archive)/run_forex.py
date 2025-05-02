from test_forex.test_forex_2 import TimeSeriesSimulator
from test_forex.models import SimpleMovingAverageModel

import pandas as pd

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
    "price": [100 + (i * 0.3) + (i % 7 - 3)*2 for i in range(100)]
})

models = [SimpleMovingAverageModel(window=10)]
sim = TimeSeriesSimulator(df, time_column="timestamp", value_column="price")

log = sim.run(models=models, window_size=15)
metrics = sim.evaluate()

from pprint import pprint
pprint(metrics)