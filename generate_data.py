"""
EcoStream - Synthetic Grid Carbon Intensity Dataset Generator
NOTE: This script generates SYNTHETIC, simulated grid carbon intensity data for
benchmarking and demonstration purposes, modeling realistic daily and seasonal grid dynamics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import DATA_DIR, GRID_DATA_PATH


def generate_carbon_data(output_path=GRID_DATA_PATH):
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[INFO] Generating synthetic grid carbon data...")

    # Generate hourly timeline for a full year (365 days)
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(24 * 365)]

    df = pd.DataFrame({"timestamp": timestamps})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Temporal feature extraction
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week  # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month

    np.random.seed(42)

    # 1. Baseline intensity (gCO2/kWh)
    base_intensity = 250.0

    # 2. Midday solar generation dip (centered around 13:00)
    solar_dip = -45.0 * np.exp(-0.5 * ((df["hour"] - 13.0) / 2.5) ** 2)

    # 3. Evening peak demand when solar drops and peaker plants start (centered around 19:00)
    evening_peak = 55.0 * np.exp(-0.5 * ((df["hour"] - 19.0) / 2.5) ** 2)

    # 4. Night baseline dip (low industrial demand between 01:00 and 05:00)
    night_effect = -15.0 * np.cos(2.0 * np.pi * df["hour"] / 24.0)

    # 5. Weekend reduction (lower commercial/industrial load on Sat=5 and Sun=6)
    is_weekend = df["day_of_week"].isin([5, 6]).astype(float)
    weekend_effect = -25.0 * is_weekend

    # 6. Seasonal month effect (higher winter heating demand in Jan/Feb, summer cooling in Jul/Aug)
    seasonal_effect = 18.0 * np.cos(2.0 * np.pi * (df["month"] - 1) / 12.0)

    # 7. Realistic stochastic noise
    noise = np.random.normal(0.0, 10.0, len(df))

    # Combine all components into final intensity
    df["carbon_intensity"] = (
        base_intensity
        + solar_dip
        + evening_peak
        + night_effect
        + weekend_effect
        + seasonal_effect
        + noise
    )

    # Bound within realistic grid operational limits
    df["carbon_intensity"] = df["carbon_intensity"].clip(50, 450).round().astype(int)

    # Save to configured path
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic dataset saved to {output_path} ({len(df)} rows)")
    print(f"[SUMMARY] Min: {df['carbon_intensity'].min()}, Max: {df['carbon_intensity'].max()}, Mean: {df['carbon_intensity'].mean():.1f} gCO2/kWh")
    return df


if __name__ == "__main__":
    generate_carbon_data()