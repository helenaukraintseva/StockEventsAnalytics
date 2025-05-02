import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class AlgorithmRNN:
    def __init__(self, target_col='Close', hidden_dim=32, epochs=5, lr=1e-3):
        """
        :param target_col: Название столбца-цели (регрессия).
        :param hidden_dim: Число скрытых нейронов в RNN.
        :param epochs: Эпохи обучения.
        :param lr: Learning rate для оптимизатора.

        (Для простоты batch_size игнорируем, тренируем "за один прогон".)
        """
        self.target_col = target_col
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')
        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)
        input_dim = X.shape[2]
        self.model = RNNModel(input_dim, self.hidden_dim, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()
        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1
        return df


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class AlgorithmLSTM:
    def __init__(self, target_col='Close', hidden_dim=32, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        input_dim = X.shape[2]
        self.model = LSTMModel(input_dim, self.hidden_dim, output_dim=1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class AlgorithmGRU:
    def __init__(self, target_col='Close', hidden_dim=32, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        input_dim = X.shape[2]
        self.model = GRUModel(input_dim, self.hidden_dim)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class CNN1DModel(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=3, output_dim=1):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class AlgorithmCNN:
    def __init__(self, target_col='Close', out_channels=32, kernel_size=3, epochs=5, lr=1e-3):
        """
        :param out_channels: число фильтров в Conv1d
        :param kernel_size: размер окна свёртки
        :param epochs: эпохи
        :param lr: learning rate
        """
        self.target_col = target_col
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)

        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        in_channels = 1
        seq_length = X.shape[2]

        self.model = CNN1DModel(in_channels, self.out_channels, self.kernel_size, output_dim=1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class CNNRNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rnn_hidden_dim, output_dim=1):
        super(CNNRNNModel, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.rnn = nn.RNN(out_channels, rnn_hidden_dim, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class AlgorithmCNNRNN:
    def __init__(self, target_col='Close', out_channels=16, kernel_size=3, rnn_hidden_dim=16, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rnn_hidden_dim = rnn_hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)

        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        in_channels = 1
        self.model = CNNRNNModel(in_channels, self.out_channels, self.kernel_size, self.rnn_hidden_dim)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class CNNLSTMModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, lstm_hidden_dim, output_dim=1):
        super(CNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.lstm = nn.LSTM(out_channels, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class AlgorithmCNNLSTM:
    def __init__(self, target_col='Close', out_channels=16, kernel_size=3, lstm_hidden_dim=16, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        self.model = CNNLSTMModel(in_channels=1,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  lstm_hidden_dim=self.lstm_hidden_dim)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class AlgorithmTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=16, nhead=2, num_layers=1):
        super(AlgorithmTransformer, self).__init__()
        """
        Упрощённый TransformerEncoder для регрессии.
        :param input_dim: Число входных признаков
        :param d_model: Размерность скрытого слоя трансформера
        :param nhead: Кол-во "голов" в multi-head attention
        :param num_layers: Число слоёв encoder
        """
        self.input_linear = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = self.fc(out)
        return out


class RunTransformer:
    def __init__(self, target_col='Close', d_model=16, nhead=2, num_layers=1, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)

        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        input_dim = X.shape[2]
        self.model = AlgorithmTransformer(input_dim=input_dim, d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1

        return df


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class AlgorithmAutoEncoder:
    def __init__(self, latent_dim=4, epochs=5, lr=1e-3):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        X = df.values.astype('float32')

        idx_close = X.shape[1] - 1
        close_actual = X[:, idx_close]

        X_torch = torch.from_numpy(X)

        input_dim = X.shape[1]
        self.model = SimpleAutoEncoder(input_dim, self.latent_dim)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, X_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            reconstructed = self.model(X_torch).numpy()

        predicted_close = reconstructed[:, idx_close]

        df['PredictedValue'] = predicted_close
        df['Signal'] = 0
        df.loc[predicted_close > close_actual, 'Signal'] = 1
        df.loc[predicted_close < close_actual, 'Signal'] = -1

        return df


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class TCNModel(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 16], kernel_size=2):
        """
        num_channels: список out_channels для каждого блока
        Упрощённая реализация TCN (без маскировки casual, padding подгоняем).
        """
        super(TCNModel, self).__init__()
        layers = []
        in_channels = input_size
        dilation_size = 1
        for out_channels in num_channels:
            layer = TemporalBlock(in_channels, out_channels,
                                  kernel_size, stride=1,
                                  dilation=dilation_size,
                                  padding=(kernel_size-1)*dilation_size)
            layers.append(layer)
            in_channels = out_channels
            dilation_size *= 2

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        """
        x: (batch, in_channels, seq_len)
        """
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out


class AlgorithmTCN:
    def __init__(self, target_col='Close', channels=[16,16], kernel_size=2, epochs=5, lr=1e-3):
        self.target_col = target_col
        self.channels = channels
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        X = df.drop(columns=[self.target_col]).values.astype('float32')
        y = df[self.target_col].values.astype('float32')

        X = np.expand_dims(X, axis=1)

        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y).view(-1, 1)

        input_size = 1
        self.model = TCNModel(input_size=input_size,
                              num_channels=self.channels,
                              kernel_size=self.kernel_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = criterion(outputs, y_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = self.model(X_torch).squeeze().numpy()

        df['PredictedValue'] = preds
        df['Signal'] = 0
        df.loc[preds > y, 'Signal'] = 1
        df.loc[preds < y, 'Signal'] = -1
        return df
