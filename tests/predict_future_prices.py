import numpy as np
from main_interface import predict_future_prices


def test_predict_future_prices(mocker):
    # Подменяем загрузку модели
    mocker.patch("joblib.load", return_value=lambda x: np.zeros(x.shape[0]))

    last_sequence = np.random.rand(1, 40)
    preds = predict_future_prices("dummy_model.pkl", last_sequence, n_steps=10)
    assert len(preds) == 10
