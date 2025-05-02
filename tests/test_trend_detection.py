import pandas as pd
from main_interface import detect_trend_signals

def test_trend_detection():
    # подделываем данные
    prices = [100 + i for i in range(100)]
    df = pd.DataFrame({"price": prices})
    signal, result_df = detect_trend_signals(df, trend_window=10)

    assert signal in ["trend_up", "trend_down", "flat", "reversal_up", "reversal_down"]
    assert "ema" in result_df.columns
    assert "signal_strength" in result_df.columns