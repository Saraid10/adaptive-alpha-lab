import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
INTERVAL  = "1h"          # 1-hour bars
DB_PATH   = "data/market.duckdb"

# How far back to pull on first run
LOOKBACK_DAYS = 730       # 2 years