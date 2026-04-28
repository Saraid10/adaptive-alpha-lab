import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import umap
import os

from dataset import WindowDataset, load_feature_matrix, normalize, WINDOW_SIZE
from encoder import TemporalEncoder, NTXentLoss

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL      = "BTCUSDT"
N_EPOCHS    = 30
BATCH_SIZE  = 128
LR          = 3e-4
LATENT_DIM  = 16
N_REGIMES   = 4          # number of market regimes to discover
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "../models"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Training on: {DEVICE}")


# ── 1. Load + normalize features ─────────────────────────────────────────────
print("Loading features...")
X_raw = load_feature_matrix(SYMBOL)
X, mean, std = normalize(X_raw)
print(f"Feature matrix: {X.shape}")  # (17460, 22)


# ── 2. Dataset + DataLoader ───────────────────────────────────────────────────
dataset    = WindowDataset(X, window=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Dataset: {len(dataset):,} windows | {len(dataloader)} batches/epoch")


# ── 3. Model + optimizer ──────────────────────────────────────────────────────
model     = TemporalEncoder(n_features=X.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
criterion = NTXentLoss(temperature=0.07)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ── 4. Training loop ──────────────────────────────────────────────────────────
print("\nTraining encoder...")
loss_history = []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for anchor, positive in dataloader:
        anchor   = anchor.to(DEVICE)
        positive = positive.to(DEVICE)

        z_anchor   = model(anchor)
        z_positive = model(positive)

        loss = criterion(z_anchor, z_positive)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

# Save model
torch.save(model.state_dict(), f"{SAVE_DIR}/encoder.pt")
print(f"\nModel saved to {SAVE_DIR}/encoder.pt")


# ── 5. Extract all embeddings ─────────────────────────────────────────────────
print("\nExtracting embeddings for all windows...")
model.eval()
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(X) - WINDOW_SIZE, 8):   # stride 8 for speed
        window = torch.tensor(
            X[i : i + WINDOW_SIZE], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        z = model(window)
        all_embeddings.append(z.cpu().numpy())

embeddings = np.vstack(all_embeddings)
np.save(f"{SAVE_DIR}/embeddings.npy", embeddings)
print(f"Embeddings shape: {embeddings.shape}")


# ── 6. GMM clustering → regime labels ────────────────────────────────────────
print(f"\nFitting GMM with K={N_REGIMES} regimes...")
gmm = GaussianMixture(
    n_components=N_REGIMES,
    covariance_type="full",
    n_init=5,
    random_state=42,
)
gmm.fit(embeddings)
labels      = gmm.predict(embeddings)
posteriors  = gmm.predict_proba(embeddings)  # shape (N, 4) — regime probabilities

np.save(f"{SAVE_DIR}/labels.npy", labels)
np.save(f"{SAVE_DIR}/posteriors.npy", posteriors)

print(f"Regime distribution:")
for k in range(N_REGIMES):
    pct = (labels == k).mean() * 100
    print(f"  Regime {k}: {pct:.1f}% of windows")


# ── 7. UMAP visualization ─────────────────────────────────────────────────────
print("\nRunning UMAP projection...")
reducer   = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
embedding_2d = reducer.fit_transform(embeddings)

colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]
fig, ax = plt.subplots(figsize=(10, 8))

for k in range(N_REGIMES):
    mask = labels == k
    ax.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        c=colors[k], label=f"Regime {k}",
        alpha=0.5, s=8,
    )

ax.set_title(f"UMAP Projection of Learned Regime Embeddings — {SYMBOL}", fontsize=13)
ax.legend(markerscale=3)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/umap_regimes.png", dpi=150)
plt.show()
print(f"UMAP plot saved to {SAVE_DIR}/umap_regimes.png")


# ── 8. Loss curve ─────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(loss_history, color="#3B82F6", linewidth=2)
ax2.set_title("Contrastive Training Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("NT-Xent Loss")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/loss_curve.png", dpi=150)
plt.show()
print("Loss curve saved.")

print("\n✓ Phase 2 complete. Ready for Phase 3 — regime-conditioned alpha models.")