"""
RadNet dose-rate regression with uncertainty quantification.

Stage 1: baseline regression (gamma channel counts -> dose-equivalent rate)
Stage 2: deep ensemble for uncertainty (prediction spread across NN seeds)
Stage 3: split conformal prediction (calibrated coverage guarantee on top
         of the ensemble mean)

This is written to scale to the full multi-year, multi-station RadNet
archive -- just point load_radnet_csv() at more files and concat.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# raw gamma channel columns from the RadNet export, and the column we're
# trying to predict


BASE_FEATURES = ["R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"]
TARGET = "DOSE_RATE"

""""
def load_radnet_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def load_radnet_csv(path, sample_size=None, random_state=42):
    df = pd.read_csv(path)
    station_dummies = pd.get_dummies(df["LOCATION_NAME"], prefix="station")
    df = pd.concat([df, station_dummies], axis=1)
    FEATURES = FEATURES + list(station_dummies.columns)  # updating global FEATURES accordingly
    df = df.dropna(subset=FEATURES + [TARGET])
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    return df


def load_radnet_csv(path, sample_size=None, random_state=42):
    df = pd.read_csv(path)
    station_dummies = pd.get_dummies(df["LOCATION_NAME"], prefix="station")
    print(df["LOCATION_NAME"].value_counts())
    df = pd.concat([df, station_dummies], axis=1)
    features = BASE_FEATURES + list(station_dummies.columns)
    df = df.dropna(subset=features + [TARGET])
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    return df, features
"""

def load_radnet_csv(path, sample_size=None, random_state=42):
    df = pd.read_csv(path)

    # LOCATION_NAME covers more than one station, so one-hot encode it --
    # otherwise the model has no way to tell "this station reads high" apart
    # from "the dose rate is actually high"
         
    station_dummies = pd.get_dummies(df["LOCATION_NAME"], prefix="station")
    df = pd.concat([df, station_dummies], axis=1)

    # log-transformed counts (compresses the spikes)
    log_features = []
    for col in BASE_FEATURES:
        log_col = f"log_{col}"
        df[log_col] = np.log1p(df[col])
        log_features.append(log_col)

    # channel ratios (cancel out overall intensity, keep spectral shape --
    # this is closer to what real isotope-ID/dose algorithms use)
    ratio_features = []
    pairs = [("R02", "R08"), ("R02", "R09"), ("R03", "R07"), ("R05", "R09")]
    for num, denom in pairs:
        ratio_col = f"ratio_{num}_{denom}"
        df[ratio_col] = df[num] / df[denom].replace(0, np.nan)
        ratio_features.append(ratio_col)

    features = (
        BASE_FEATURES
        + list(station_dummies.columns)
        + log_features
        + ratio_features
    )
    # quick look at the most extreme ratio_R02_R09 rows before dropping
    # anything -- worth eyeballing since a ratio blowing up usually means
    # either a real spectral anomaly or a denominator sitting near zero
         
    print(df["ratio_R02_R09"].sort_values(ascending=False).head(10))
    df = df.dropna(subset=features + [TARGET])

    # sanity check on a handful of the rows that showed up in the sort above
    print(df.loc[[121617, 121616, 121672, 121632, 121606], ["LOCATION_NAME", "SAMPLE_TIME", "R02", "R09", "DOSE_RATE"]])
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    return df, features




#df = df[(df["DOSE_RATE"] > 0) & (df[FEATURES] > 0).all(axis=1)]

def baseline_regression(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    #rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print(f"[Baseline RF] MAE = {mean_absolute_error(y_test, pred):.2f} nSv/h, "
          f"R^2 = {r2_score(y_test, pred):.3f}")
    return rf, X_train, X_test, y_train, y_test


#def deep_ensemble(X_train, y_train, X_test, y_test, n_models=8):
def deep_ensemble(X_train, y_train, X_test, y_test, n_models=15):
    """Train n_models MLPs with different seeds/bootstraps; use spread as UQ."""
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    preds = []
    rng = np.random.default_rng(0)
    for i in range(n_models):
        idx = rng.integers(0, len(Xtr), len(Xtr))  # bootstrap resample
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=3000,
            random_state=i,
            alpha=1e-3,
        )
        model.fit(Xtr[idx], y_train[idx])
        preds.append(model.predict(Xte))
    preds = np.array(preds)  # (n_models, n_test)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    mae = mean_absolute_error(y_test, mean_pred)
    print(f"[Deep Ensemble, n={n_models}] MAE = {mae:.2f} nSv/h, "
          f"mean predictive std = {std_pred.mean():.2f} nSv/h")
    return mean_pred, std_pred, scaler


def split_conformal(mean_pred_cal, y_cal, mean_pred_test, alpha=0.1):
    """
    Split conformal prediction: use held-out calibration residuals to build
    a distribution-free prediction interval with guaranteed marginal
    coverage of 1 - alpha, regardless of whether the underlying model is
    well-calibrated.
    """
    residuals = np.abs(y_cal - mean_pred_cal)
    q = np.quantile(residuals, 1 - alpha)
    lower = mean_pred_test - q
    upper = mean_pred_test + q
    print(f"[Conformal] target coverage = {100*(1-alpha):.0f}%, "
          f"half-width = {q:.2f} nSv/h")
    return lower, upper, q


if __name__ == "__main__":
    #df = load_radnet_csv("/Users/binishbatool/PycharmProjects/pythonProject/radnet_full.csv")
    df, FEATURES = load_radnet_csv("radnet_full.csv", sample_size=None)
    corr = df[["DOSE_RATE"] + BASE_FEATURES].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson correlation"})
    plt.title(f"DOSE_RATE vs gamma channels (n={len(df)})")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    #plt.show()

    df = df[(df["DOSE_RATE"] > 0) & (df[BASE_FEATURES] > 0).all(axis=1)]

    print(f"Loaded {len(df)} hourly readings from Pittsburgh RadNet station")
    print(df[[TARGET] + FEATURES].describe().round(1))
    print()

    rf, X_train, X_test, y_train, y_test = baseline_regression(df)
    print()

    # split test into calibration + final test for conformal
    X_cal, X_final, y_cal, y_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=1
    )

    mean_pred_cal, std_cal, scaler = deep_ensemble(X_train, y_train, X_cal, y_cal)
    Xte_final_scaled = scaler.transform(X_final)
    # re-predict on final split using same ensemble logic (quick re-fit for demo)
    mean_pred_final, std_final, _ = deep_ensemble(X_train, y_train, X_final, y_final)
    print()

    lower, upper, half_width = split_conformal(mean_pred_cal, y_cal, mean_pred_final)
    covered = np.mean((y_final >= lower) & (y_final <= upper))
    print(f"[Conformal] empirical coverage on held-out test = {100*covered:.0f}%")
