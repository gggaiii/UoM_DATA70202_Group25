import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from tqdm import tqdm
import torch

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_forecast_fixed_future(file_path, model_dir, log_path, output_dir,
                                target_col='MIDAGRI_PRICE', holiday_col='IS_HOLIDAY'):
    forecast_start = pd.to_datetime("2025-02-01")
    forecast_end = pd.to_datetime("2025-04-30")

    forecast_steps = {
        'D': 89,
        'W': 12,
        'ME': 3  # Month-End
    }

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path, parse_dates=['DATE'])
    df = df.set_index('DATE')
    df = df[[target_col, holiday_col]]
    df[holiday_col] = df[holiday_col].astype(int)

    model_name = os.path.splitext(os.path.basename(file_path))[0]
    total_steps = len(forecast_steps)
    progress_bar = tqdm(total=total_steps, desc=f"{model_name} Forecasting", ncols=100)

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n=== Forecast session started: {datetime.now()} | File: {model_name} ===\n")

        for freq, steps in forecast_steps.items():
            start_time = time.time()

            try:
                # Resample and assign explicit frequency to avoid index warnings
                ts = df[target_col].resample(freq).mean().asfreq(freq)
                exog = df[[holiday_col]].resample(freq).max().fillna(0).asfreq(freq)

                # Prepare training data
                train = ts[ts.index < forecast_start]
                train_exog = exog[exog.index < forecast_start]

                # Drop NaNs in target and align exog
                mask = train.notna()
                train = train[mask]
                train_exog = train_exog.loc[mask]

                # Prepare future exog input
                future_dates = pd.date_range(forecast_start, forecast_end, freq=freq)
                future_exog = exog.reindex(future_dates).fillna(0).iloc[:steps]

                # Auto ARIMA selection
                auto_model = pm.auto_arima(train,
                                           exogenous=train_exog,
                                           seasonal=True,
                                           m={'D': 7, 'W': 52, 'ME': 12}[freq],
                                           stepwise=True,
                                           suppress_warnings=True,
                                           error_action="ignore",
                                           max_p=3, max_q=3, max_P=2, max_Q=2)

                model = SARIMAX(train,
                                exog=train_exog,
                                order=auto_model.order,
                                seasonal_order=auto_model.seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

                results = model.fit(disp=False)

                # Forecast
                forecast = results.get_forecast(steps=steps, exog=future_exog)
                pred_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()

                # Log
                elapsed = time.time() - start_time
                log_file.write(
                    f"[{datetime.now()}] FREQ: {freq}, Order: {auto_model.order}, "
                    f"Seasonal: {auto_model.seasonal_order}, Steps: {steps}, Time: {elapsed:.2f}s\n"
                )

                # Save forecast to CSV
                forecast_df = pd.DataFrame({
                    'DATE': pred_mean.index,
                    'Forecast': pred_mean,
                    'Lower_CI': pred_ci.iloc[:, 0],
                    'Upper_CI': pred_ci.iloc[:, 1]
                })
                forecast_df.to_csv(os.path.join(output_dir, f"forecast_{model_name}_{freq}.csv"), index=False)

                # Save plot
                plt.figure(figsize=(12, 5))
                train.plot(label='Observed')
                pred_mean.plot(label='Forecast')
                plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], alpha=0.3)
                plt.axvline(forecast_start, color='gray', linestyle='--')
                plt.title(f'{model_name} - SARIMA Forecast ({freq}) Feb–Apr 2025')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"plot_{model_name}_{freq}.png"))
                plt.close()

                # Save model
                model_file = os.path.join(model_dir, f"sarima_{model_name}_{freq}.pkl")
                joblib.dump(results, model_file)

            except Exception as e:
                log_file.write(f"[{datetime.now()}] FREQ: {freq}, ERROR: {str(e)}\n")
                print(f"Error ({model_name} - {freq}): {e}")

            progress_bar.update(1)

        log_file.write(f"=== Forecast session ended: {datetime.now()} ===\n")
        log_file.flush()
        progress_bar.close()


# === Batch Run on All Zone Files ===
base_data_path = r"C:\StudySources\Master\ADS\data"
model_dir = r"C:\Github\learning_blog\Data_Science\ADS\models\model\sarima_models"
log_path = r"C:\Github\learning_blog\Data_Science\ADS\models\model\SARIMA_training_log.txt"
output_dir = r"C:\Github\learning_blog\Data_Science\ADS\models\model\outputs"

datasets = ['df_nor_extended.csv', 'df_sur_extended.csv', 'df_cen_extended.csv']

for file in datasets:
    train_forecast_fixed_future(
        file_path=os.path.join(base_data_path, file),
        model_dir=model_dir,
        log_path=log_path,
        output_dir=output_dir
    )
