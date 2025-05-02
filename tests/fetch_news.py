from main_interface import fetch_news

def test_fetch_news(mocker):
    mocker.patch("main_interface.parse_telegram_news", return_value=[{"text": "Crypto news"}])
    result = fetch_news(source="if_market_news", days_back=3)
    assert isinstance(result, list)
    assert "text" in result[0]
