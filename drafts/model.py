
import torch
import torch.nn as nn
from config import BIN_APIKEY, BIN_SECRETKEY
from parsing.binance_parser import BinanceParser


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


pars_bin = BinanceParser(api_key=BIN_APIKEY, secret_key=BIN_SECRETKEY)

model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
model.load_state_dict(torch.load('model.pth'))

x_test = pars_bin.get_data_model(symbol="BTCUSDT", interval="1m", keys=["close"])
X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)

print(X_test.shape)
    result = model(X_test)
    print(float(result))