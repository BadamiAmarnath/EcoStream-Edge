import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from config import FEATURES, GRID_DATA_PATH, PREDICTOR_PATH, MODELS_DIR


def train_grid_predictor():
    if not os.path.exists(GRID_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {GRID_DATA_PATH}. Please run generate_data.py first."
        )

    print(f"[INFO] Loading training dataset from {GRID_DATA_PATH}...")
    df = pd.read_csv(GRID_DATA_PATH)

    X = df[FEATURES]
    y = df["carbon_intensity"]

    # Time-based 80/20 train/test split without shuffling (preserving time order)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[INFO] Evaluating model on time-series holdout ({len(X_train)} train, {len(X_test)} test)...")
    eval_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    test_preds = eval_model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    print(f"[EVAL] Holdout Test MAE: {mae:.2f} gCO2/kWh")

    # Fit final model on full historical dataset
    print(f"[INFO] Training production model on full dataset with features: {FEATURES}...")
    final_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    # Verify feature names
    print(f"[INFO] Model feature names locked: {list(final_model.feature_names_in_)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(PREDICTOR_PATH, "wb") as f:
        pickle.dump(final_model, f)

    print(f"[SUCCESS] Weights saved cleanly to {PREDICTOR_PATH}")
    return final_model, mae


if __name__ == "__main__":
    train_grid_predictor()
