import numpy as np
import pandas as pd
import duckdb
import torch
from torch.utils.data import Dataset
from config import DB_PATH

WINDOW_SIZE = 60  # 60 hourly bars per sample = 2.5 days of context


def load_feature_matrix(symbol: str = "BTCUSDT") -> np.ndarray:
    """Load all 18 features for a symbol as a numpy array, time-ordered."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            ret_1h, ret_5h, ret_15h, ret_60h,
            vol_5h, vol_20h, vol_of_vol,
            amihud, volume_zscore, ret_autocorr,
            spread_proxy, ofi_proxy, rsi_14, gk_vol,
            skewness, kurtosis, macd_signal,
            bband_pct_b, atr_14, close_vs_vwap,
            log_vol_trend, ret_dispersion
        FROM features
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    con.close()
    return df.values.astype(np.float32)


def normalize(X: np.ndarray) -> np.ndarray:
    """Standardize each feature to zero mean, unit variance."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


class WindowDataset(Dataset):
    """
    Sliding window dataset for contrastive learning.
    Each item is a (anchor, positive) pair:
      - anchor:   window starting at index i
      - positive: window starting at index i+1 (adjacent = same regime)
    Negatives are handled inside the loss function (other items in the batch).
    """
    def __init__(self, X: np.ndarray, window: int = WINDOW_SIZE):
        self.X = X
        self.window = window
        # Valid start indices — need room for anchor + positive
        self.indices = list(range(len(X) - window - 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        anchor   = self.X[i     : i + self.window]       # shape (60, 22)
        positive = self.X[i + 1 : i + self.window + 1]   # shape (60, 22)
        return (
            torch.tensor(anchor,   dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
        )