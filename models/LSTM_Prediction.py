import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load model, scaler, and config
# -----------------------------
model_path = r"C:\Github\learning_blog\Data_Science\ADS\models\model\midagri_lstm.pt"
scaler_path = r"C:\Github\learning_blog\Data_Science\ADS\models\model\midagri_lstm_scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Load scaler and model
scaler = joblib.load(scaler_path)
model = LSTMModel(input_size=0)  # placeholder, will reset below

# -----------------------------
# 2. Load training data and holidays
# -----------------------------
df = pd.read_csv(r"C:\StudySources\Master\ADS\data\merged.csv", parse_dates=["DATE"])
holiday_df = pd.read_csv(r"C:\StudySources\Master\ADS\data\Combined_Holidays.csv", parse_dates=["DATE"])

df = df.sort_values("DATE")

# Merge holidays into training data
holiday_df["is_holiday"] = 1
df = df.merge(holiday_df[["DATE", "is_holiday"]], on="DATE", how="left")
df["is_holiday"] = df["is_holiday"].fillna(0)

# One-hot encode zones
zone_dummies = pd.get_dummies(df["ZONE"], prefix="ZONE", dtype=int)
sales_dummies = pd.get_dummies(df["SALES_ZONE"], prefix="SALES", dtype=int)
df = pd.concat([df, zone_dummies, sales_dummies], axis=1)

# Scale price
df["price_scaled"] = scaler.transform(df[["MIDAGRI_PRICE"]])

# Final feature columns
feature_cols = ["price_scaled", "is_holiday"] + list(zone_dummies.columns) + list(sales_dummies.columns)
df[feature_cols] = df[feature_cols].fillna(0)

# Prepare last known window
last_known = df[feature_cols].iloc[-30:].copy().reset_index(drop=True)

# Update model input size and load weights
input_size = last_known.shape[1]
model = LSTMModel(input_size=input_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 3. Prepare future data
# -----------------------------
future_dates = pd.date_range(start="2025-04-01", end="2025-9-30", freq="D")

# Copy holidays from 2024 to 2025-2026
holiday_2024 = holiday_df.copy()
def shift_holiday_date(d):
    try:
        if d.month >= 4:
            return d.replace(year=2025)
        else:
            return d.replace(year=2026)
    except ValueError:
        return pd.NaT  # Drop leap day (Feb 29) if it doesn't exist

holiday_2024["DATE"] = holiday_2024["DATE"].apply(shift_holiday_date)
holiday_2024 = holiday_2024.dropna(subset=["DATE"])

future_df = pd.DataFrame({"DATE": future_dates})
future_df = future_df.merge(
    holiday_2024[["DATE"]].assign(is_holiday=1), on="DATE", how="left"
)
future_df["is_holiday"] = future_df["is_holiday"].fillna(0)

# Get zone/sales_zone one-hot vectors from last known
zone_cols = [col for col in last_known.columns if col.startswith("ZONE_")]
sales_cols = [col for col in last_known.columns if col.startswith("SALES_")]
zone_values = last_known[zone_cols].iloc[-1].values
sales_values = last_known[sales_cols].iloc[-1].values

# -----------------------------
# 4. Predict day by day
# -----------------------------
predictions = []
window = last_known.copy()

for date in future_df["DATE"]:
    is_holiday = future_df.loc[future_df["DATE"] == date, "is_holiday"].values[0]
    input_row = [1] + [is_holiday] + list(zone_values) + list(sales_values)
    
    input_seq = window.copy()
    input_seq.iloc[-1, 0] = 0.0  # remove price leak
    input_tensor = torch.tensor([input_seq.values], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().item()
        pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    predictions.append({"DATE": date, "PREDICTED_PRICE": pred_price})

    new_row = [pred_scaled] + input_row[1:]
    window = pd.concat([window.iloc[1:], pd.DataFrame([new_row], columns=window.columns)], ignore_index=True)

pred_df = pd.DataFrame(predictions)
pred_df.set_index("DATE", inplace=True)

# -----------------------------
# 5. Resample to week & month
# -----------------------------
# --- Apply smoothing ---
pred_df["SMOOTHED_DAILY"] = pred_df["PREDICTED_PRICE"].ewm(span=7).mean()

# --- Drop first month visually ---
pred_df = pred_df[pred_df.index >= "2025-05-01"]
pred_df.to_csv(r"C:\Github\learning_blog\Data_Science\ADS\models\model\midagri_lstm_predictions.csv")

weekly = pred_df.resample("W").mean()
monthly = pred_df.resample("M").mean()

# --- Plot ---
plt.figure(figsize=(16, 6))
plt.plot(pred_df.index, pred_df["SMOOTHED_DAILY"], label="Daily (Smoothed)", linewidth=1.2, alpha=0.8)
plt.plot(weekly.index, weekly["PREDICTED_PRICE"], label="Weekly Avg", linewidth=2)
plt.plot(monthly.index, monthly["PREDICTED_PRICE"], label="Monthly Avg", linewidth=2.5)

plt.title("Forecasted MIDAGRI_PRICE (2025.05 to 2026.12)", fontsize=14, weight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Price", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
plt.show()
