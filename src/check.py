import duckdb
import pandas as pd
from config import DB_PATH, FEATURE_COLS

con = duckdb.connect(DB_PATH, read_only=True)

# ── OHLCV summary ─────────────────────────────────────────────────────────────
print("=" * 55)
print("OHLCV SUMMARY")
print("=" * 55)
ohlcv = con.execute("""
    SELECT
        symbol,
        COUNT(*)       AS total_bars,
        MIN(open_time) AS earliest,
        MAX(open_time) AS latest
    FROM ohlcv
    GROUP BY symbol
    ORDER BY symbol
""").df()
print(ohlcv.to_string(index=False))

# ── Feature summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FEATURE SUMMARY")
print("=" * 55)
feat = con.execute("""
    SELECT
        symbol,
        COUNT(*)            AS total_rows,
        MIN(open_time)      AS earliest,
        MAX(open_time)      AS latest,
        ROUND(AVG(rsi_14), 2) AS avg_rsi,
        ROUND(AVG(vol_20h), 6) AS avg_vol
    FROM features
    GROUP BY symbol
    ORDER BY symbol
""").df()
print(feat.to_string(index=False))
print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── Gap check ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("GAP CHECK")
print("=" * 55)
for symbol in ["BTCUSDT", "ETHUSDT"]:
    df = con.execute("""
        SELECT open_time FROM ohlcv
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["diff"]      = df["open_time"].diff()
    gaps = df[df["diff"] > pd.Timedelta("1h")].dropna()
    if gaps.empty:
        print(f"  {symbol}: no gaps found")
    else:
        print(f"  {symbol}: {len(gaps)} gaps found")
        print(gaps[["open_time", "diff"]].head(5).to_string(index=False))

con.close()
print("\nOK: Data check complete.")
