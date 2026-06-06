import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Initialize Baseline Models
def get_baseline_models():
  """Returns a dictionary of simple baseline models"""

  return {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
  }

# Initialize the Neural Network
class EnergyLSTM(nn.Module):
  """LSTM model for energy prediction"""

  def __init__(self, input_size, hidden_layer_size=64, num_layers=2, dropout=0.2):
    super().__init__()

    self.hidden_layer_size = hidden_layer_size

    self.lstm = nn.LSTM(input_size,
                        hidden_layer_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True)

    self.dropout = nn.Dropout(dropout)
    self.dense = nn.Linear(hidden_layer_size, 16)

    self.output = nn.Linear(16, 1)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    lstm_out = self.dropout(lstm_out)
    dense_out = self.dense(lstm_out)
    output = self.output(dense_out)

    return output