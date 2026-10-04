"""
NSW electricity demand forecasting.

Data expected in one folder (--data-dir):
  - PRICE_AND_DEMAND_YYYYMM_NSW1.csv   AEMO, 5-min intervals
        cols: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
  - Sydney.YYYYMM.csv / Wollongong.YYYYMM.csv / IDCJDW2124.202605.csv
        BoM daily reports, preamble before a '"Date"' header line

Target: hourly TOTALDEMAND (MW), one step ahead.

Usage:
    pip install pandas numpy scikit-learn lightgbm matplotlib
    python nsw_demand_forecast.py --data-dir ./data
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_demand(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "PRICE_AND_DEMAND_*_NSW1.csv")))
    if not files:
        raise FileNotFoundError("No PRICE_AND_DEMAND_*_NSW1.csv files found.")
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["SETTLEMENTDATE", "TOTALDEMAND", "RRP"])
        frames.append(df)
    demand = pd.concat(frames, ignore_index=True)
    demand["SETTLEMENTDATE"] = pd.to_datetime(
        demand["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    )
    demand = demand.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE")

    # 5-min -> hourly. Timestamps mark the END of each 5-min interval, so the
    # readings 00:05..01:00 belong to hour 00. Label each row by the hour it
    # falls into using a start-of-hour convention: shift back 5 min, then floor.
    idx = (demand["SETTLEMENTDATE"] - pd.Timedelta(minutes=5)).dt.floor("h")
    demand = demand.assign(hour=idx)
    hourly = (
        demand.groupby("hour")
        .agg(demand_mw=("TOTALDEMAND", "mean"), rrp=("RRP", "mean"))
        .reset_index()
        .rename(columns={"hour": "ts"})
    )
    return hourly


def _find_header_line(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if '"Date"' in line or line.strip().startswith('"Date"') or ',"Date",' in line:
                return i
            if re.match(r'^,?"?Date"?,', line):
                return i
    raise ValueError(f"No Date header found in {path}")


def load_weather(data_dir, prefix, label):
    # Sydney's May 2026 report is misnamed IDCJDW2124.202605.csv
    patterns = [os.path.join(data_dir, f"{prefix}.*.csv")]
    if prefix == "Sydney":
        patterns.append(os.path.join(data_dir, "IDCJDW2124.*.csv"))
    files = sorted(set(sum([glob.glob(p) for p in patterns], [])))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        skip = _find_header_line(f)
        df = pd.read_csv(f, skiprows=skip, encoding="utf-8", encoding_errors="ignore")
        df.columns = [str(c).strip() for c in df.columns]
        # first blank column -> drop
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        def col(sub):
            hits = [c for c in df.columns if sub.lower() in c.lower()]
            return hits[0] if hits else None

        c_date = col("Date")
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[c_date], errors="coerce")
        rename = {
            "Minimum temperature": "tmin",
            "Maximum temperature": "tmax",
            "Rainfall": "rain",
            "Speed of maximum wind gust": "gust",
            "9am Temperature": "t9",
            "9am relative humidity": "rh9",
            "3pm Temperature": "t3",
            "3pm relative humidity": "rh3",
        }
        for sub, name in rename.items():
            c = col(sub)
            out[name] = pd.to_numeric(df[c], errors="coerce") if c else np.nan

        out = out.dropna(subset=["date"])
        frames.append(out)

    w = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    w = w.add_suffix(f"_{label}").rename(columns={f"date_{label}": "date"})
    return w


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def build_features(hourly, syd, wol):
    df = hourly.copy()
    df["date"] = df["ts"].dt.normalize()

    # Daily weather is only fully known after the day ends. To avoid leakage,
    # attach YESTERDAY's weather to each hour (a real forecast-time proxy).
    for w in (syd, wol):
        if not w.empty:
            wl = w.copy()
            wl["date"] = wl["date"] + pd.Timedelta(days=1)  # shift forward 1 day
            df = df.merge(wl, on="date", how="left")

    # calendar
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["ts"].dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["ts"].dt.dayofyear / 365)

    # lags / rolling on demand (all use PAST values only)
    for lag in (1, 2, 3, 24, 48, 168):
        df[f"lag_{lag}"] = df["demand_mw"].shift(lag)
    df["roll24_mean"] = df["demand_mw"].shift(1).rolling(24).mean()
    df["roll24_std"] = df["demand_mw"].shift(1).rolling(24).std()
    df["roll168_mean"] = df["demand_mw"].shift(1).rolling(168).mean()

    df = df.dropna(subset=[f"lag_{l}" for l in (1, 24, 168)]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def train_and_eval(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = df["demand_mw"]
    drop = ["ts", "date", "demand_mw"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    # chronological split: last 20% is the test set
    cut = int(len(df) * 0.8)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    ts_te = df["ts"].iloc[cut:]

    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.03, random_state=42
        )

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    mape = np.mean(np.abs((yte - pred) / yte)) * 100
    r2 = r2_score(yte, pred)

    print("\n=== Test set (chronological last 20%) ===")
    print(f"rows       : {len(yte)}")
    print(f"MAE        : {mae:8.1f} MW")
    print(f"RMSE       : {rmse:8.1f} MW")
    print(f"MAPE       : {mape:8.2f} %")
    print(f"R^2        : {r2:8.4f}")

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop features:")
        print(imp.head(12).to_string())

    plot_residuals(ts_te.values, yte.values, pred)

    return model, X.columns.tolist()


def plot_residuals(ts, actual, pred, out_path="residuals.png"):
    """Four diagnostic views of the test-set residuals (actual - predicted)."""
    import matplotlib
    matplotlib.use("Agg")  # save to file without needing a display
    import matplotlib.pyplot as plt

    resid = actual - pred

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # 1) residuals over time — look for periods the model drifts
    ax[0, 0].plot(ts, resid, lw=0.5, color="steelblue")
    ax[0, 0].axhline(0, color="black", lw=0.8)
    ax[0, 0].set_title("Residuals over time")
    ax[0, 0].set_ylabel("actual - predicted (MW)")

    # 2) residuals vs predicted — should be a shapeless band around 0
    ax[0, 1].scatter(pred, resid, s=4, alpha=0.3, color="steelblue")
    ax[0, 1].axhline(0, color="black", lw=0.8)
    ax[0, 1].set_title("Residuals vs predicted")
    ax[0, 1].set_xlabel("predicted demand (MW)")
    ax[0, 1].set_ylabel("residual (MW)")

    # 3) histogram — should be roughly bell-shaped, centred on 0
    ax[1, 0].hist(resid, bins=60, color="steelblue", edgecolor="white")
    ax[1, 0].axvline(0, color="black", lw=0.8)
    ax[1, 0].set_title("Residual distribution")
    ax[1, 0].set_xlabel("residual (MW)")

    # 4) predicted vs actual — points should hug the diagonal
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax[1, 1].scatter(actual, pred, s=4, alpha=0.3, color="steelblue")
    ax[1, 1].plot([lo, hi], [lo, hi], color="black", lw=0.8)
    ax[1, 1].set_title("Predicted vs actual")
    ax[1, 1].set_xlabel("actual demand (MW)")
    ax[1, 1].set_ylabel("predicted demand (MW)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved residual plots to {out_path}")


def _make_model():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.03, random_state=42
        )


def predict_next_hour(hourly, syd, wol, feature_cols):
    """Train on ALL available data, then predict the single hour right after the
    data ends. Run this once the test scores from train_and_eval look good."""
    # Add one empty hour after the last real hour, so build_features computes its
    # lags/rolling stats from real recent demand. demand_mw is unknown (NaN) there.
    next_ts = hourly["ts"].iloc[-1] + pd.Timedelta(hours=1)
    extended = pd.concat(
        [hourly, pd.DataFrame({"ts": [next_ts], "demand_mw": [np.nan], "rrp": [np.nan]})],
        ignore_index=True,
    )

    feats = build_features(extended, syd, wol)  # target-NaN row survives: its lags are real

    train = feats[feats["demand_mw"].notna()]
    target_row = feats[feats["ts"] == next_ts]
    if target_row.empty:
        print("\nCould not build the next-hour feature row (missing recent data).")
        return

    model = _make_model()
    model.fit(train[feature_cols], train["demand_mw"])
    pred = model.predict(target_row[feature_cols])[0]

    print(f"\n=== Next-hour forecast ===")
    print(f"predicted demand for {next_ts}: {pred:8.1f} MW")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    print("Loading demand...")
    hourly = load_demand(args.data_dir)
    print(f"  {len(hourly)} hourly rows, {hourly['ts'].min()} -> {hourly['ts'].max()}")

    print("Loading weather...")
    syd = load_weather(args.data_dir, "Sydney", "syd")
    wol = load_weather(args.data_dir, "Wollongong", "wol")
    print(f"  Sydney days: {len(syd)}, Wollongong days: {len(wol)}")

    print("Building features...")
    df = build_features(hourly, syd, wol)
    print(f"  {len(df)} training rows, {df.shape[1]} columns")

    # 1) Check how good the model is (train on past 80%, test on last 20%).
    _, feature_cols = train_and_eval(df)

    # 2) Now use ALL the data to predict the next hour after your data ends.
    predict_next_hour(hourly, syd, wol, feature_cols)


if __name__ == "__main__":
    main()