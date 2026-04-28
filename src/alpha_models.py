import numpy as np
import pandas as pd
import duckdb
import torch
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

from config import DB_PATH
from dataset import load_feature_matrix, normalize, WINDOW_SIZE
from encoder import TemporalEncoder

SYMBOL    = "BTCUSDT"
SAVE_DIR  = "../models"
N_REGIMES = 4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.65   # only train on high-confidence regime windows

FEATURE_COLS = [
    "ret_1h", "ret_5h", "ret_15h", "ret_60h",
    "vol_5h", "vol_20h", "vol_of_vol",
    "amihud", "volume_zscore", "ret_autocorr",
    "spread_proxy", "ofi_proxy", "rsi_14", "gk_vol",
    "skewness", "kurtosis", "macd_signal",
    "bband_pct_b", "atr_14", "close_vs_vwap",
    "log_vol_trend", "ret_dispersion"
]

os.makedirs(SAVE_DIR, exist_ok=True)


# ── 1. Load features + prices from DB ────────────────────────────────────────
print("Loading data...")
con = duckdb.connect(DB_PATH, read_only=True)

feat_df = con.execute(f"""
    SELECT open_time, {', '.join(FEATURE_COLS)}
    FROM features
    WHERE symbol = ?
    ORDER BY open_time
""", [SYMBOL]).df()

price_df = con.execute("""
    SELECT open_time, close
    FROM ohlcv
    WHERE symbol = ?
    ORDER BY open_time
""", [SYMBOL]).df()

con.close()

feat_df["open_time"]  = pd.to_datetime(feat_df["open_time"])
price_df["open_time"] = pd.to_datetime(price_df["open_time"])

# Merge features + prices
df = feat_df.merge(price_df, on="open_time", how="inner")

# Target: next-bar return (forward 1h return)
df["target_return"] = df["close"].pct_change(1).shift(-1)
df["target_sign"]   = (df["target_return"] > 0).astype(int)  # 1=up, 0=down
df = df.dropna().reset_index(drop=True)
print(f"Dataset: {len(df):,} rows")


# ── 2. Load regime posteriors ─────────────────────────────────────────────────
print("Loading regime posteriors...")
posteriors = np.load(f"{SAVE_DIR}/posteriors.npy")   # shape (N_windows, 4)
labels     = np.load(f"{SAVE_DIR}/labels.npy")

# posteriors were extracted with stride=4 — align to feature df
# Each posterior at index i corresponds to feature row i*4 + WINDOW_SIZE
stride = 4
regime_rows = []
for i in range(len(posteriors)):
    feat_idx = i * stride + WINDOW_SIZE - 1
    if feat_idx < len(df):
        row = {"feat_idx": feat_idx}
        for k in range(N_REGIMES):
            row[f"post_{k}"] = posteriors[i, k]
        row["regime"] = labels[i]
        regime_rows.append(row)

regime_df = pd.DataFrame(regime_rows).set_index("feat_idx")
print(f"Regime posteriors aligned: {len(regime_df):,} rows")


# ── 3. Walk-forward validation ────────────────────────────────────────────────
print("\nRunning walk-forward validation...")

# 6-month expanding window: train on months 1-6, predict month 7, roll forward
HOURS_PER_MONTH = 24 * 30
INITIAL_TRAIN   = HOURS_PER_MONTH * 6    # 6 months initial training
STEP_SIZE       = HOURS_PER_MONTH        # roll 1 month at a time

all_predictions = []   # collect out-of-sample predictions

fold = 0
train_end = INITIAL_TRAIN

while train_end + STEP_SIZE < len(df):
    test_start = train_end
    test_end   = min(train_end + STEP_SIZE, len(df) - 1)

    fold += 1
    print(f"  Fold {fold}: train [0:{train_end}] | test [{test_start}:{test_end}]")

    # Get indices that have regime labels
    train_idx = [i for i in regime_df.index if i < train_end]
    test_idx  = [i for i in regime_df.index if test_start <= i < test_end]

    if len(train_idx) < 100 or len(test_idx) < 10:
        train_end += STEP_SIZE
        continue

    # Train one LightGBM model per regime
    fold_preds = []

    for k in range(N_REGIMES):
        # High-confidence training samples for this regime
        regime_train_idx = [
            i for i in train_idx
            if regime_df.loc[i, f"post_{k}"] >= CONFIDENCE_THRESHOLD
        ]

        if len(regime_train_idx) < 30:
            continue

        X_train = df.loc[regime_train_idx, FEATURE_COLS].values
        y_train = df.loc[regime_train_idx, "target_sign"].values

        # Test: ALL test windows, weighted by regime posterior
        X_test = df.loc[test_idx, FEATURE_COLS].values

        model_k = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model_k.fit(X_train, y_train)

        # Probability of up-move from regime k's model
        prob_up = model_k.predict_proba(X_test)[:, 1]

        for j, idx in enumerate(test_idx):
            fold_preds.append({
                "feat_idx": idx,
                "regime":   k,
                "prob_up":  prob_up[j],
                "weight":   regime_df.loc[idx, f"post_{k}"],
            })

    if fold_preds:
        fold_df = pd.DataFrame(fold_preds)
        # Ensemble: weighted average of all regime models
        ensemble = (
            fold_df.groupby("feat_idx")
            .apply(lambda x: (x["prob_up"] * x["weight"]).sum() / x["weight"].sum())
            .reset_index()
        )
        ensemble.columns = ["feat_idx", "ensemble_prob"]
        ensemble["pred_sign"]     = (ensemble["ensemble_prob"] > 0.5).astype(int)
        ensemble["actual_sign"]   = df.loc[ensemble["feat_idx"].values, "target_sign"].values
        ensemble["actual_return"] = df.loc[ensemble["feat_idx"].values, "target_return"].values
        all_predictions.append(ensemble)

    train_end += STEP_SIZE

# ── 4. Aggregate all OOS predictions ─────────────────────────────────────────
print("\nAggregating out-of-sample predictions...")
oos_df = pd.concat(all_predictions, ignore_index=True)
oos_df = oos_df.merge(
    df[["open_time"]].reset_index().rename(columns={"index": "feat_idx"}),
    on="feat_idx", how="left"
)

# ── 5. Performance metrics ────────────────────────────────────────────────────
print("\n── Out-of-Sample Performance ─────────────────────────")

# Direction accuracy
acc = accuracy_score(oos_df["actual_sign"], oos_df["pred_sign"])
print(f"Direction accuracy:  {acc*100:.2f}%  (baseline = 50%)")

# Information coefficient
ic = oos_df["ensemble_prob"].corr(oos_df["actual_return"])
print(f"Information coefficient (IC): {ic:.4f}  (>0.02 is good for hourly)")

# Signal: go long when prob_up > 0.5, short otherwise
oos_df["signal"] = np.where(oos_df["ensemble_prob"] > 0.5, 1, -1)
oos_df["strat_return"] = oos_df["signal"] * oos_df["actual_return"]

# Transaction cost: 10bps per trade (5bps spread each side)
oos_df["trade"]       = oos_df["signal"].diff().abs() / 2
oos_df["tc"]          = oos_df["trade"] * 0.001
oos_df["net_return"]  = oos_df["strat_return"] - oos_df["tc"]

sharpe = oos_df["net_return"].mean() / (oos_df["net_return"].std() + 1e-8) * np.sqrt(8760)
print(f"Annualized Sharpe (TC-adjusted): {sharpe:.3f}")

# Max drawdown
cum = (1 + oos_df["net_return"]).cumprod()
roll_max = cum.cummax()
drawdown = (cum - roll_max) / roll_max
max_dd = drawdown.min()
print(f"Max drawdown: {max_dd*100:.2f}%")

# ── 6. Equity curve ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

# Strategy vs buy-and-hold
ax1 = axes[0]
strat_cum  = (1 + oos_df["net_return"]).cumprod()
bh_returns = oos_df["actual_return"]
bh_cum     = (1 + bh_returns).cumprod()

ax1.plot(strat_cum.values,  label="Adaptive Alpha Engine", color="#3B82F6", linewidth=1.5)
ax1.plot(bh_cum.values,     label="Buy & Hold BTC",        color="#94A3B8", linewidth=1, alpha=0.7)
ax1.set_title("Equity Curve — Out-of-Sample (TC-adjusted)", fontsize=12)
ax1.set_ylabel("Portfolio Value (starting=1)")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Drawdown
ax2 = axes[1]
ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="#EF4444", alpha=0.4)
ax2.set_title("Drawdown", fontsize=12)
ax2.set_ylabel("Drawdown %")
ax2.grid(True, alpha=0.3)

# Rolling IC (30-bar)
ax3 = axes[2]
rolling_ic = (
    oos_df["ensemble_prob"]
    .rolling(200)
    .corr(oos_df["actual_return"])
)
ax3.plot(rolling_ic.values, color="#10B981", linewidth=1)
ax3.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("Rolling IC (200-bar window) — Signal Decay Check", fontsize=12)
ax3.set_ylabel("IC")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/equity_curve.png", dpi=150)
plt.show()

# ── 7. Per-regime performance ─────────────────────────────────────────────────
print("\n── Per-Regime Performance ────────────────────────────")
oos_df["regime"] = df.loc[
    oos_df["feat_idx"].values, "ret_1h"   # placeholder index align
].values

# Load regime labels aligned to oos predictions
regime_labels_aligned = []
for idx in oos_df["feat_idx"].values:
    if idx in regime_df.index:
        regime_labels_aligned.append(regime_df.loc[idx, "regime"])
    else:
        regime_labels_aligned.append(-1)

oos_df["regime_label"] = regime_labels_aligned

for k in range(N_REGIMES):
    r = oos_df[oos_df["regime_label"] == k]
    if len(r) < 10:
        continue
    r_sharpe = r["net_return"].mean() / (r["net_return"].std() + 1e-8) * np.sqrt(8760)
    r_acc    = accuracy_score(r["actual_sign"], r["pred_sign"])
    print(f"  Regime {k}: {len(r):4d} samples | "
          f"accuracy {r_acc*100:.1f}% | Sharpe {r_sharpe:.3f}")

np.save(f"{SAVE_DIR}/oos_predictions.npy", oos_df.values)
oos_df.to_csv(f"{SAVE_DIR}/oos_predictions.csv", index=False)
print(f"\n✓ Phase 3 complete. OOS predictions saved.")
print("Ready for Phase 4 — backtesting + risk engine.")