import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Make sure our data folder exists
os.makedirs('data', exist_ok=True)

# Generate a timeline of timestamps for a full year (hourly intervals)
start_date = datetime(2026, 1, 1, 0, 0, 0)
timestamps = [start_date + timedelta(hours=i) for i in range(24 * 365)]

df = pd.DataFrame({'timestamp': timestamps})
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time features so our Machine Learning model can learn the patterns
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Create realistic carbon grid waves (Low at night/midday, heavy spikes during evening demand)
np.random.seed(42)
base_intensity = 250  # Baseline average (gCO2/kWh)
daily_cycle = 80 * np.sin(2 * np.pi * df['hour'] / 24)
evening_peak = 50 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
noise = np.random.normal(0, 15, len(df))

# Combine everything together
df['carbon_intensity'] = base_intensity + daily_cycle + evening_peak + noise
df['carbon_intensity'] = df['carbon_intensity'].clip(50, 450).astype(int)

# Save it safely into our data folder
df.to_csv('data/carbon_grid_data.csv', index=False)
print("✅ Success! data/carbon_grid_data.csv created beautifully.")