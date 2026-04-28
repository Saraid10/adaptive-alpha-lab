import duckdb

con = duckdb.connect("data/market.duckdb", read_only=True)

# Get BTC hourly bars as a DataFrame
df = con.execute("""
    SELECT * FROM ohlcv
    WHERE symbol = 'BTCUSDT'
    ORDER BY open_time
""").df()

print(df.head())
print(f"\n{len(df):,} bars loaded")
con.close()