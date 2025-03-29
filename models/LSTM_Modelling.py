import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import time
from tqdm import tqdm

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. Load and preprocess data
# -----------------------------
price_df = pd.read_csv(r"C:\StudySources\Master\ADS\data\merged.csv", parse_dates=["DATE"])
holiday_df = pd.read_csv(r"C:\StudySources\Master\ADS\data\Combined_Holidays.csv", parse_dates=["DATE"])

holiday_df["is_holiday"] = 1
merged = price_df.merge(holiday_df[["DATE", "is_holiday"]], on="DATE", how="left")
merged["is_holiday"] = merged["is_holiday"].fillna(0)

# One-hot encode ZONE and SALES_ZONE (force numeric)
zone_dummies = pd.get_dummies(merged["ZONE"], prefix="ZONE", dtype=int)
sales_zone_dummies = pd.get_dummies(merged["SALES_ZONE"], prefix="SALES", dtype=int)


# Concatenate with main DataFrame
merged = pd.concat([merged, zone_dummies, sales_zone_dummies], axis=1)

# Sort and scale MIDAGRI_PRICE
merged = merged.sort_values("DATE")
scaler = MinMaxScaler()
merged["price_scaled"] = scaler.fit_transform(merged[["MIDAGRI_PRICE"]])

# Final input feature set
feature_cols = ["price_scaled", "is_holiday"] + list(zone_dummies.columns) + list(sales_zone_dummies.columns)
features = merged[feature_cols].values

merged[feature_cols] = merged[feature_cols].fillna(0)  # ensure no NaNs
features = merged[feature_cols].values.astype(np.float32)  # ensure dtype is valid

# -----------------------------
# 3. Create sequences
# -----------------------------
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][0])  # price_scaled
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(features, window_size)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

input_size = X.shape[2]

# -----------------------------
# 4. Define LSTM model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. Train loop with logging
# -----------------------------
os.makedirs("model", exist_ok=True)
log_path = "model/training_log.txt"

start_time = time.time()
best_loss = float("inf")
patience = 10
counter = 0

with open(log_path, "w") as log_file:
    log_file.write("Training Log for LSTM MIDAGRI Model\n")
    log_file.write("="*40 + "\n")

    for epoch in range(100):
        model.train()
        running_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test.to(device)).squeeze()
            val_loss = criterion(val_pred, y_test.to(device))

        log_line = f"Epoch {epoch+1}: Train Loss = {running_loss/len(train_loader):.6f}, Val Loss = {val_loss.item():.6f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), "model/midagri_lstm.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                log_file.write("Early stopping triggered.\n")
                break

    # -----------------------------
    # 6. Final evaluation
    # -----------------------------
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).squeeze().cpu().numpy()
        y_true_scaled = y_test.cpu().numpy()

        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        log_file.write("\nFinal Evaluation Metrics:\n")
        log_file.write(f"MAE  : {mae:.4f}\n")
        log_file.write(f"MSE  : {mse:.4f}\n")
        log_file.write(f"RMSE : {rmse:.4f}\n")
        log_file.write(f"MAPE : {mape:.2f}%\n")

    # -----------------------------
    # 7. Save scaler and training time
    # -----------------------------
    joblib.dump(scaler, "model/midagri_scaler.pkl")
    total_time = time.time() - start_time
    log_file.write(f"\nTotal Training Time: {total_time:.2f} seconds\n")

print("Training complete. Model, scaler, and logs saved to /model")
print(f"Training log saved to {log_path}")
