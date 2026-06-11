import torch.nn as nn

class EnergyLSTM(nn.Module):
    """Initial LSTM model for energy prediction"""
    def __init__(self, input_size):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TunedLSTM(nn.Module):
    """Configurable LSTM for hyperparameter tuning"""
    def __init__(self, input_size, h_size, drop):
        super(TunedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, h_size, num_layers=2, batch_first=True, dropout=drop)
        self.fc = nn.Linear(h_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])