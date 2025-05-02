from main_interface import load_ml_model
import joblib
import os


def test_load_ml_model(tmp_path):
    model_path = tmp_path / "test_model_signal.pkl"
    joblib.dump({"test": "model"}, model_path)

    loaded_model = load_ml_model(str(model_path.stem))  # без .pkl
    assert isinstance(loaded_model, dict)
    assert "test" in loaded_model
