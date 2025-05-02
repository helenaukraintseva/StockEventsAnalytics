from main_interface import build_windowed_features
import pandas as pd

def test_build_windowed_features():
    df = pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [2, 3, 4, 5, 6],
        "low": [0, 1, 2, 3, 4],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [100, 110, 120, 130, 140],
    })
    result = build_windowed_features(df, window_size=3)
    assert result.shape[0] == 3  # 5 - 3 + 1 окон
    assert result.shape[1] == 15  # 3 окна * 5 признаков
