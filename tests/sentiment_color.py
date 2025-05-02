from main_interface import sentiment_color

def test_sentiment_color():
    assert sentiment_color("positive") == "#d4edda"
    assert sentiment_color("negative") == "#f8d7da"
    assert sentiment_color("neutral") == "#e2e3e5"
    assert sentiment_color("unknown") == "#ffffff"  # fallback цвет
