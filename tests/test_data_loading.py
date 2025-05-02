from main_interface import get_crypto_price, get_historical_data

def test_crypto_price():
    price = get_crypto_price("BTCUSDT")
    assert price is not None
    assert price > 0

def test_historical_data():
    df = get_historical_data("BTCUSDT", interval="1m", limit=50)
    assert df is not None
    assert not df.empty
    assert "time" in df.columns
    assert "price" in df.columns
