import duckdb
import pandas as pd
import numpy as np
from config import DB_PATH

# ── Load raw OHLCV from DuckDB ─────────────────────────────────────────────────

def load_ohlcv(symbol: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT open_time, open, high, low, close, volume,
               taker_buy_base_vol
        FROM ohlcv
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time")
    return df


# ── The 18 features ───────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # 1. Multi-horizon log returns
    for h in [1, 5, 15, 60]:
        f[f"ret_{h}h"] = np.log(df["close"] / df["close"].shift(h))

    # 2. Realized volatility (rolling std of 1h returns)
    f["vol_5h"]  = f["ret_1h"].rolling(5).std()
    f["vol_20h"] = f["ret_1h"].rolling(20).std()

    # 3. Volatility of volatility
    f["vol_of_vol"] = f["vol_20h"].rolling(20).std()

    # 4. Amihud illiquidity — |return| / volume
    f["amihud"] = f["ret_1h"].abs() / (df["volume"] + 1e-8)

    # 5. Volume Z-score
    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std()
    f["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)

    # 6. Return autocorrelation (lag-1)
    f["ret_autocorr"] = (
        f["ret_1h"]
        .rolling(20)
        .apply(lambda x: x.autocorr(lag=1), raw=False)
    )

    # 7. Bid-ask spread proxy
    f["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # 8. Order flow imbalance proxy
    hl = df["high"] - df["low"] + 1e-8
    f["ofi_proxy"] = (df["close"] - df["open"]) / hl

    # 9. RSI (14-bar)
    f["rsi_14"] = _rsi(df["close"], 14)

    # 10. Garman-Klass volatility
    f["gk_vol"] = _garman_klass(df)

    # 11. Rolling skewness
    f["skewness"] = f["ret_1h"].rolling(20).skew()

    # 12. Rolling kurtosis
    f["kurtosis"] = f["ret_1h"].rolling(20).kurt()

    # 13. MACD signal line
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd  = ema12 - ema26
    f["macd_signal"] = macd - macd.ewm(span=9).mean()

    # 14. Bollinger %B
    sma20  = df["close"].rolling(20).mean()
    std20  = df["close"].rolling(20).std()
    f["bband_pct_b"] = (df["close"] - (sma20 - 2*std20)) / (4*std20 + 1e-8)

    # 15. ATR (14-bar)
    f["atr_14"] = _atr(df, 14)

    # 16. Close vs VWAP proxy
    vwap = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-8)
    f["close_vs_vwap"] = (df["close"] - vwap) / (vwap + 1e-8)

    # 17. Log volume trend (5-bar slope)
    log_vol = np.log(df["volume"] + 1)
    f["log_vol_trend"] = log_vol.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # 18. Return dispersion
    f["ret_dispersion"] = pd.concat([
        f["ret_1h"], f["ret_5h"], f["ret_15h"], f["ret_60h"]
    ], axis=1).std(axis=1)

    return f


# ── Helper functions ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl   = df["high"] - df["low"]
    hpc  = (df["high"] - df["close"].shift(1)).abs()
    lpc  = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Save features back to DuckDB ──────────────────────────────────────────────
def save_features(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    symbol: str
):
    df = df.copy()
    df["symbol"] = symbol
    df = df.reset_index()  # brings open_time back as column
    df = df.dropna()       # drop warmup rows with NaN

    con.register("feat_data", df)

    # Drop and recreate cleanly each run — simpler than upsert for now
    con.execute("DROP TABLE IF EXISTS features")
    con.execute("""
        CREATE TABLE features AS
        SELECT * FROM feat_data
    """)

    con.unregister("feat_data")
    print(f"  Saved {len(df):,} rows for {symbol}")

# ── Run it ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)

    all_features = []

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\nComputing features for {symbol}...")
        raw = load_ohlcv(symbol, con)
        features = compute_features(raw)
        features["symbol"] = symbol
        features = features.reset_index().dropna()
        all_features.append(features)
        print(f"  Computed {len(features):,} rows for {symbol}")

    # Combine both symbols and save once
    combined = pd.concat(all_features, ignore_index=True)
    con.register("feat_data", combined)
    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM feat_data")
    con.unregister("feat_data")
    print(f"\n  Total saved: {len(combined):,} rows")

    # Sanity check
    result = con.execute("""
        SELECT symbol, COUNT(*) as rows,
               ROUND(AVG(vol_20h), 6) as avg_vol,
               ROUND(AVG(rsi_14), 2)  as avg_rsi
        FROM features
        GROUP BY symbol
    """).df()
    print("\nFeature summary:")
    print(result.to_string(index=False))

    con.close()
    print("\n✓ Features saved to DuckDB. Ready for Phase 2 — the encoder.")