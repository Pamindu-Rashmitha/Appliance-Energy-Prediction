import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import copy
import random
import os

# Import neural networks from model.py
from model import EnergyLSTM, TunedLSTM

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints MAE and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} Performance")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
    return mae, rmse

# Load Data
data_path = 'data\processed\train_ready_data.csv'
df = pd.read_csv(data_path)

# Separate Features and Target
X = df.drop(columns=['date', 'Appliances']).values
y = df['Appliances'].values

# Temporal Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print('Baseline Models')

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae, lr_rmse = evaluate_model(y_test, lr_preds, 'Linear Regression')

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae, rf_rmse = evaluate_model(y_test, rf_preds, 'Random Forest')

print('LSTM Model Training')

# Prepare data for LSTM
X_train_lstm = torch.tensor(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])), dtype=torch.float32).to(device)
y_train_lstm = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_lstm = torch.tensor(X_test.reshape((X_test.shape[0], 1, X_test.shape[1])), dtype=torch.float32).to(device)
y_test_lstm = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train_lstm, y_train_lstm), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_test_lstm, y_test_lstm), batch_size=64, shuffle=False)

# Initialize Model from model.py
model = EnergyLSTM(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simple Training Loop
epochs = 50
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)

    # Validation loss calculation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for v_X, v_y in val_loader:
            v_out = model(v_X)
            v_loss = criterion(v_out, v_y)
            epoch_val_loss += v_loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_losses[-1]:.6f}')

model.eval()
with torch.no_grad():
    lstm_preds = model(X_test_lstm).cpu().numpy().flatten()

lstm_mae, lstm_rmse = evaluate_model(y_test, lstm_preds, 'LSTM')

# Plotting Initial Results 
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_title('LSTM Training & Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].legend()

axes[0, 1].plot(y_test[:200], label='Actual', alpha=0.7)
axes[0, 1].plot(lstm_preds[:200], label='Predicted', alpha=0.7)
axes[0, 1].set_title('Predicted vs Actual (LSTM)')
axes[0, 1].legend()

residuals = y_test - lstm_preds
axes[1, 0].scatter(lstm_preds, residuals, alpha=0.3)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_title('Residual Plot (LSTM)')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Residual')

models_list = ['Linear Reg', 'Random Forest', 'LSTM']
maes_list = [lr_mae, rf_mae, lstm_mae]
axes[1, 1].bar(models_list, maes_list, color=['blue', 'green', 'orange'])
axes[1, 1].set_title('Model MAE Comparison')
axes[1, 1].set_ylabel('MAE')

plt.tight_layout()
plt.show()

# Hyperparameter Tuning 
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5):
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_X, v_y in val_loader:
                val_loss += criterion(model(v_X), v_y).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_wts)
    return model, best_loss

param_grid = {
    'lr': [0.001, 0.0005],
    'hidden_size': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3]
}

best_val_rmse = float('inf')
best_params = {}
best_optimized_model = None

print('\n--- Starting Random Search ---')
for i in range(5):
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print(f'Trial {i+1}: {params}')

    # Use TunedLSTM 
    t_model = TunedLSTM(X_train.shape[1], params['hidden_size'], params['dropout']).to(device)
    t_optimizer = optim.Adam(t_model.parameters(), lr=params['lr'])

    t_model, val_rmse = train_with_early_stopping(t_model, train_loader, val_loader, criterion, t_optimizer)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params
        best_optimized_model = copy.deepcopy(t_model)

print(f'Best Params Found: {best_params}')

# Get predictions for the optimized model to evaluate it
best_optimized_model.eval()
with torch.no_grad():
    opt_preds = best_optimized_model(X_test_lstm).cpu().numpy().flatten()
    
print("\n--- Optimized LSTM Performance ---")
opt_mae, opt_rmse = evaluate_model(y_test, opt_preds, 'Optimized LSTM')

# Final Comparison Plot
labels = ['Linear Reg', 'Random Forest', 'Initial LSTM', 'Optimized LSTM']
mae_values = [lr_mae, rf_mae, lstm_mae, opt_mae]
rmse_values = [lr_rmse, rf_rmse, lstm_rmse, opt_rmse]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='steelblue')

ax.set_ylabel('Error Value')
ax.set_title('Performance Metrics Comparison: MAE vs RMSE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()