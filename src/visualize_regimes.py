import numpy as np
import torch
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from dataset import load_feature_matrix, normalize, WINDOW_SIZE
from encoder import TemporalEncoder
from config import DB_PATH

SYMBOL   = "BTCUSDT"
SAVE_DIR = "../models"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_REGIMES = 4

# ── 1. Load model + features ──────────────────────────────────────────────────
print("Loading encoder...")
model = TemporalEncoder(n_features=22, latent_dim=16)
model.load_state_dict(torch.load(f"{SAVE_DIR}/encoder.pt", map_location=DEVICE))
model.eval()

X_raw = load_feature_matrix(SYMBOL)
X, _, _ = normalize(X_raw)

# ── 2. Extract embeddings with stride=1 (dense) ───────────────────────────────
print("Extracting dense embeddings (stride=1)...")
embeddings = []
timestamps = []

# Load timestamps from DB
con = duckdb.connect(DB_PATH, read_only=True)
times = con.execute("""
    SELECT open_time FROM features
    WHERE symbol = ? ORDER BY open_time
""", [SYMBOL]).df()
con.close()
times["open_time"] = pd.to_datetime(times["open_time"])

with torch.no_grad():
    for i in range(0, len(X) - WINDOW_SIZE, 4):   # stride=4, much denser
        window = torch.tensor(
            X[i : i + WINDOW_SIZE], dtype=torch.float32
        ).unsqueeze(0)
        z = model(window)
        embeddings.append(z.cpu().numpy())
        timestamps.append(times["open_time"].iloc[i + WINDOW_SIZE - 1])

embeddings = np.vstack(embeddings)
timestamps = pd.DatetimeIndex(timestamps)
print(f"Embeddings: {embeddings.shape}")

# ── 3. Re-fit GMM ─────────────────────────────────────────────────────────────
print("Fitting GMM...")
gmm = GaussianMixture(n_components=N_REGIMES, covariance_type="full",
                      n_init=10, random_state=42)
gmm.fit(embeddings)
labels     = gmm.predict(embeddings)
posteriors = gmm.predict_proba(embeddings)

# Silhouette score — measures cluster separation (higher = better, max=1)
sil = silhouette_score(embeddings, labels, sample_size=2000)
print(f"Silhouette score: {sil:.4f}  (baseline HMM typically ~0.15)")

# ── 4. UMAP with better settings ──────────────────────────────────────────────
print("Running UMAP...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=50,      # larger = more global structure preserved
    min_dist=0.1,        # tighter clusters
    random_state=42,
)
emb2d = reducer.fit_transform(embeddings)

# ── 5. Plot 1: UMAP colored by regime ────────────────────────────────────────
colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]
labels_named = {0: "Regime 0", 1: "Regime 1", 2: "Regime 2", 3: "Regime 3"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: colored by regime
ax = axes[0]
for k in range(N_REGIMES):
    mask = labels == k
    ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
               c=colors[k], label=f"Regime {k} ({mask.mean()*100:.0f}%)",
               alpha=0.5, s=6)
ax.set_title(f"UMAP — Regime Labels\nSilhouette: {sil:.3f}", fontsize=12)
ax.legend(markerscale=3, fontsize=9)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

# Right: colored by regime certainty (max posterior)
ax2 = axes[1]
certainty = posteriors.max(axis=1)
sc = ax2.scatter(emb2d[:, 0], emb2d[:, 1],
                 c=certainty, cmap="RdYlGn", alpha=0.5, s=6,
                 vmin=0.4, vmax=1.0)
plt.colorbar(sc, ax=ax2, label="Regime certainty")
ax2.set_title("UMAP — Regime Certainty\n(green=confident, red=uncertain)", fontsize=12)
ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")

plt.suptitle(f"Adaptive Alpha Engine — Learned Regime Structure ({SYMBOL})", fontsize=13)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/umap_improved.png", dpi=150)
plt.show()

# ── 6. Plot 2: Regime timeline overlaid on price ──────────────────────────────
print("Plotting regime timeline...")
con = duckdb.connect(DB_PATH, read_only=True)
price_df = con.execute("""
    SELECT open_time, close FROM ohlcv
    WHERE symbol = ? ORDER BY open_time
""", [SYMBOL]).df()
con.close()
price_df["open_time"] = pd.to_datetime(price_df["open_time"])

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

# Top: BTC price
ax3.plot(price_df["open_time"], price_df["close"],
         color="#1e293b", linewidth=0.6, alpha=0.9)
ax3.set_ylabel("BTC Price (USDT)", fontsize=10)
ax3.set_title("BTC Price", fontsize=11)
ax3.grid(True, alpha=0.2)

# Bottom: regime over time (colored bands)
for k in range(N_REGIMES):
    mask = labels == k
    ax4.scatter(timestamps[mask], np.ones(mask.sum()) * k,
                c=colors[k], s=3, alpha=0.6, label=f"Regime {k}")

ax4.set_ylabel("Detected Regime", fontsize=10)
ax4.set_title("Regime Timeline", fontsize=11)
ax4.set_yticks(range(N_REGIMES))
ax4.legend(markerscale=4, fontsize=9, loc="upper right")
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/regime_timeline.png", dpi=150)
plt.show()

# ── 7. Regime statistics ──────────────────────────────────────────────────────
print("\n── Regime Statistics ─────────────────────────────────")
for k in range(N_REGIMES):
    mask = labels == k
    avg_certainty = posteriors[mask, k].mean()
    print(f"Regime {k}: {mask.sum():4d} windows | "
          f"{mask.mean()*100:.1f}% | "
          f"avg certainty: {avg_certainty:.3f}")

print(f"\nSilhouette score: {sil:.4f}")
print("(>0.20 = good separation for financial data)")
print("\n✓ Visualizations saved. Ready for Phase 3.")