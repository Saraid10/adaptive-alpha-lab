import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import DB_PATH, LATENT_DIM, N_FEATURES, N_REGIMES, SAVE_DIR, SYMBOLS, WINDOW_SIZE
from dataset import load_feature_matrix
from encoder import TemporalEncoder
from train_encoder import BATCH_SIZE, LR


RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POST_COLS = [f"post_{idx}" for idx in range(N_REGIMES)]
ASSIGNMENT_COLS = ["method", "symbol", "open_time", "feat_idx", "regime"] + POST_COLS
GUIDED_METHODS = ["hmm_guided_gmm", "hmm_guided_hmm"]


@dataclass
class GuidedSamples:
    matrices: dict[str, np.ndarray]
    samples: pd.DataFrame


class GuidedWindowDataset(Dataset):
    def __init__(self, matrices: dict[str, np.ndarray], samples: pd.DataFrame):
        self.matrices = matrices
        self.samples = samples.reset_index(drop=True)
        symbol_order = {symbol: idx for idx, symbol in enumerate(sorted(matrices))}
        self.samples["symbol_id"] = self.samples["symbol"].map(symbol_order).astype(int)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        start_idx = int(row.start_idx)
        window = self.matrices[row.symbol][start_idx : start_idx + WINDOW_SIZE]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(int(row.hmm_regime), dtype=torch.long),
            torch.tensor(int(row.symbol_id), dtype=torch.long),
            torch.tensor(int(row.feat_idx), dtype=torch.long),
        )


class HMMGuidedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        min_positive_gap: int = 24,
        hard_negative_gap: int = 24,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.min_positive_gap = min_positive_gap
        self.hard_negative_gap = hard_negative_gap
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        symbol_ids: torch.Tensor,
        feat_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = z.shape[0]
        sim = torch.mm(z, z.T) / self.temperature
        eye = torch.eye(batch_size, device=z.device, dtype=torch.bool)

        same_state = labels[:, None].eq(labels[None, :]) & ~eye
        same_symbol = symbol_ids[:, None].eq(symbol_ids[None, :])
        temporal_gap = torch.abs(feat_idx[:, None] - feat_idx[None, :])
        distant_or_cross_symbol = (~same_symbol) | (temporal_gap >= self.min_positive_gap)
        positive_mask = same_state & distant_or_cross_symbol

        hard_negative_mask = (~labels[:, None].eq(labels[None, :])) & same_symbol
        hard_negative_mask = hard_negative_mask & (temporal_gap <= self.hard_negative_gap) & ~eye

        if positive_mask.sum() == 0:
            positive_mask = same_state

        logits = sim - sim.masked_fill(eye, -1e9).max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(logits).masked_fill(eye, 0.0)
        denominator_weights = torch.ones_like(exp_sim)
        denominator_weights = torch.where(
            hard_negative_mask,
            torch.full_like(denominator_weights, self.hard_negative_weight),
            denominator_weights,
        )

        numerator = (exp_sim * positive_mask.float()).sum(dim=1)
        denominator = (exp_sim * denominator_weights).sum(dim=1)
        valid = (numerator > 0) & (denominator > 0)

        if valid.sum() == 0:
            loss = z.sum() * 0.0
        else:
            loss = -torch.log(numerator[valid] / denominator[valid]).mean()

        stats = {
            "valid_anchor_pct": float(valid.float().mean().detach().cpu()),
            "positive_pair_count": float(positive_mask.float().sum().detach().cpu()),
            "hard_negative_pair_count": float(hard_negative_mask.float().sum().detach().cpu()),
        }
        return loss, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Phase 18 HMM-guided contrastive encoder.")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min-positive-gap", type=int, default=24)
    parser.add_argument("--hard-negative-gap", type=int, default=24)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


def load_hmm_labels(symbols: list[str]) -> pd.DataFrame:
    path = Path(SAVE_DIR) / "regime_assignments.csv"
    if not path.exists():
        raise RuntimeError("regime_assignments.csv missing. Run baselines.py before guided_encoder.py.")
    assignments = pd.read_csv(path)
    required = {"method", "symbol", "feat_idx", "regime"}
    missing = required - set(assignments.columns)
    if missing:
        raise RuntimeError(f"regime_assignments.csv missing columns: {sorted(missing)}")
    labels = assignments[
        (assignments["method"] == "hmm") & (assignments["symbol"].isin(symbols))
    ][["symbol", "feat_idx", "regime"]].copy()
    labels["feat_idx"] = labels["feat_idx"].astype(int)
    labels["hmm_regime"] = labels["regime"].astype(int)
    return labels.drop(columns="regime")


def fit_guided_scaler(matrices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack(list(matrices.values()))
    mean = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-8
    np.save(Path(SAVE_DIR) / "guided_norm_mean.npy", mean)
    np.save(Path(SAVE_DIR) / "guided_norm_std.npy", std)
    return mean, std


def load_guided_samples(symbols: list[str]) -> GuidedSamples:
    raw = {}
    for symbol in symbols:
        matrix = load_feature_matrix(symbol)
        if len(matrix) <= WINDOW_SIZE + 1:
            print(f"NOTE: {symbol} has too few feature rows for guided windows; skipping.")
            continue
        raw[symbol] = matrix
    if not raw:
        raise RuntimeError("No feature matrices loaded. Run features.py first.")

    mean, std = fit_guided_scaler(raw)
    matrices = {symbol: (matrix - mean) / std for symbol, matrix in raw.items()}
    hmm_labels = load_hmm_labels(list(matrices))

    sample_rows = []
    for symbol, matrix in matrices.items():
        label_map = hmm_labels[hmm_labels["symbol"] == symbol].set_index("feat_idx")["hmm_regime"].to_dict()
        for start_idx in range(0, len(matrix) - WINDOW_SIZE - 1):
            feat_idx = start_idx + WINDOW_SIZE - 1
            if feat_idx not in label_map:
                continue
            sample_rows.append(
                {
                    "symbol": symbol,
                    "start_idx": start_idx,
                    "feat_idx": feat_idx,
                    "hmm_regime": int(label_map[feat_idx]),
                }
            )
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        raise RuntimeError("No HMM-labeled encoder windows found.")
    return GuidedSamples(matrices, samples)


def load_feature_times(symbol: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        times = con.execute(
            """
            SELECT open_time
            FROM features
            WHERE symbol = ?
            ORDER BY open_time
            """,
            [symbol],
        ).df()
    finally:
        con.close()
    times["open_time"] = pd.to_datetime(times["open_time"])
    return times


def train_guided_encoder(samples: GuidedSamples, args: argparse.Namespace) -> tuple[TemporalEncoder, pd.DataFrame]:
    dataset = GuidedWindowDataset(samples.matrices, samples.samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if len(dataloader) == 0:
        raise RuntimeError("Guided dataloader is empty; reduce batch size or check feature rows.")

    model = TemporalEncoder(n_features=N_FEATURES, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = HMMGuidedContrastiveLoss(
        temperature=args.temperature,
        min_positive_gap=args.min_positive_gap,
        hard_negative_gap=args.hard_negative_gap,
        hard_negative_weight=args.hard_negative_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training HMM-guided encoder on {DEVICE}")
    print(f"Guided windows: {len(dataset):,} | batches/epoch: {len(dataloader):,}")

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        valid_sum = 0.0
        positive_sum = 0.0
        hard_negative_sum = 0.0
        batches = 0

        for windows, labels, symbol_ids, feat_idx in dataloader:
            windows = windows.to(DEVICE)
            labels = labels.to(DEVICE)
            symbol_ids = symbol_ids.to(DEVICE)
            feat_idx = feat_idx.to(DEVICE)

            z = model(windows)
            loss, stats = criterion(z, labels, symbol_ids, feat_idx)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            valid_sum += stats["valid_anchor_pct"]
            positive_sum += stats["positive_pair_count"]
            hard_negative_sum += stats["hard_negative_pair_count"]
            batches += 1

        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(batches, 1),
            "valid_anchor_pct": valid_sum / max(batches, 1),
            "positive_pairs_per_batch": positive_sum / max(batches, 1),
            "hard_negative_pairs_per_batch": hard_negative_sum / max(batches, 1),
            "lr": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | loss={row['loss']:.4f} | "
            f"valid={row['valid_anchor_pct']:.3f} | hard_neg={row['hard_negative_pairs_per_batch']:.1f}"
        )

    torch.save(model.state_dict(), Path(SAVE_DIR) / "guided_encoder.pt")
    loss_history = pd.DataFrame(rows)
    loss_history.to_csv(Path(SAVE_DIR) / "guided_encoder_loss.csv", index=False)
    return model, loss_history


def extract_guided_embeddings(model: TemporalEncoder, matrices: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    model.eval()
    embeddings = []
    rows = []
    with torch.no_grad():
        for symbol, matrix in matrices.items():
            times = load_feature_times(symbol)
            print(f"Extracting guided dense embeddings for {symbol}...")
            for start_idx in range(0, len(matrix) - WINDOW_SIZE):
                feat_idx = start_idx + WINDOW_SIZE - 1
                window = torch.tensor(
                    matrix[start_idx : start_idx + WINDOW_SIZE], dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                z = model(window)
                embeddings.append(z.cpu().numpy())
                rows.append(
                    {
                        "symbol": symbol,
                        "start_idx": start_idx,
                        "feat_idx": feat_idx,
                        "open_time": times["open_time"].iloc[feat_idx],
                    }
                )
    embedding_matrix = np.vstack(embeddings)
    np.save(Path(SAVE_DIR) / "guided_embeddings.npy", embedding_matrix)
    return embedding_matrix, pd.DataFrame(rows)


def probability_posts(probs: np.ndarray) -> pd.DataFrame:
    posts = np.zeros((len(probs), N_REGIMES), dtype=float)
    cols = min(probs.shape[1], N_REGIMES)
    posts[:, :cols] = probs[:, :cols]
    row_sum = posts.sum(axis=1, keepdims=True)
    posts = np.divide(posts, row_sum, out=np.full_like(posts, 1.0 / N_REGIMES), where=row_sum != 0)
    return pd.DataFrame({f"post_{idx}": posts[:, idx] for idx in range(N_REGIMES)})


def fit_guided_assignments(embeddings: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    outputs = []

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    gmm.fit(embeddings)
    gmm_labels = gmm.predict(embeddings).astype(int)
    gmm_posts = probability_posts(gmm.predict_proba(embeddings))
    gmm_frame = meta[["symbol", "open_time", "feat_idx"]].copy()
    gmm_frame["method"] = "hmm_guided_gmm"
    gmm_frame["regime"] = gmm_labels
    outputs.append(pd.concat([gmm_frame, gmm_posts], axis=1)[ASSIGNMENT_COLS])

    implementation = "hmmlearn_guided_embedding_hmm"
    try:
        from hmmlearn.hmm import GaussianHMM

        x = StandardScaler().fit_transform(embeddings)
        lengths = meta.groupby("symbol", sort=False).size().astype(int).tolist()
        hmm = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type="diag",
            n_iter=250,
            random_state=RANDOM_STATE,
            min_covar=1e-4,
        )
        hmm.fit(x, lengths)
        hmm_labels = hmm.predict(x, lengths).astype(int)
        hmm_posts = probability_posts(hmm.predict_proba(x, lengths))
    except Exception as exc:
        implementation = "guided_embedding_gmm_fallback"
        print(f"NOTE: guided embedding HMM failed ({exc}). Falling back to GMM posteriors.")
        hmm_labels = gmm_labels
        hmm_posts = gmm_posts

    hmm_frame = meta[["symbol", "open_time", "feat_idx"]].copy()
    hmm_frame["method"] = "hmm_guided_hmm"
    hmm_frame["regime"] = hmm_labels
    hmm_out = pd.concat([hmm_frame, hmm_posts], axis=1)[ASSIGNMENT_COLS]
    hmm_out["implementation"] = implementation
    outputs.append(hmm_out[ASSIGNMENT_COLS])

    assignments = pd.concat(outputs, ignore_index=True)
    assignments.to_csv(Path(SAVE_DIR) / "guided_encoder_assignments.csv", index=False)
    np.save(Path(SAVE_DIR) / "guided_labels.npy", assignments["regime"].to_numpy(dtype=int))
    return assignments


def transition_counts(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((N_REGIMES, N_REGIMES), dtype=float)
    for _, group in frame.sort_values(["symbol", "feat_idx"]).groupby("symbol", sort=False):
        labels = group["regime"].to_numpy(dtype=int)
        for source, target in zip(labels[:-1], labels[1:]):
            matrix[source, target] += 1.0
    return matrix


def avg_duration(frame: pd.DataFrame) -> float:
    runs = []
    for _, group in frame.sort_values(["symbol", "feat_idx"]).groupby("symbol", sort=False):
        switches = group["regime"].ne(group["regime"].shift()).fillna(True)
        runs.extend(group.groupby(switches.cumsum()).size().tolist())
    return float(np.mean(runs)) if runs else np.nan


def reference_metrics(method_frame: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    merged = method_frame[["symbol", "feat_idx", "regime"]].merge(
        reference[["symbol", "feat_idx", "regime"]],
        on=["symbol", "feat_idx"],
        suffixes=("", "_reference"),
        how="inner",
    )
    if merged.empty:
        return {"hmm_reference_nmi": np.nan, "hmm_reference_ari": np.nan, "hmm_reference_purity": np.nan}
    labels = merged["regime"].astype(int)
    ref = merged["regime_reference"].astype(int)
    contingency = pd.crosstab(labels, ref)
    purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())
    return {
        "hmm_reference_nmi": float(normalized_mutual_info_score(ref, labels)),
        "hmm_reference_ari": float(adjusted_rand_score(ref, labels)),
        "hmm_reference_purity": purity,
    }


def summarize_guided(
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    loss_history: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    reference = pd.read_csv(Path(SAVE_DIR) / "regime_assignments.csv")
    reference = reference[reference["method"] == "hmm"].copy()
    rows = []
    for method in GUIDED_METHODS:
        frame = assignments[assignments["method"] == method].reset_index(drop=True)
        common = frame[["symbol", "feat_idx"]].merge(
            reference[["symbol", "feat_idx"]], on=["symbol", "feat_idx"], how="inner"
        )
        common_positions = frame.reset_index(names="_pos").merge(common, on=["symbol", "feat_idx"], how="inner")["_pos"]
        labels = frame.loc[common_positions, "regime"].to_numpy(dtype=int)
        x = embeddings[common_positions.to_numpy(dtype=int)]
        sil = (
            silhouette_score(x, labels, sample_size=min(2000, len(labels)), random_state=RANDOM_STATE)
            if len(np.unique(labels)) > 1
            else np.nan
        )
        transitions = transition_counts(frame)
        transition_total = transitions.sum()
        post = frame[POST_COLS].to_numpy(dtype=float)
        row = {
            "method": method,
            "loss": "hmm_guided_contrastive",
            "augmentation": "time_only",
            "assignment_method": method.replace("hmm_guided_", ""),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "min_positive_gap": int(args.min_positive_gap),
            "hard_negative_gap": int(args.hard_negative_gap),
            "hard_negative_weight": float(args.hard_negative_weight),
            "n_rows": int(len(common)),
            "n_symbols": int(frame["symbol"].nunique()),
            "n_regimes": int(frame["regime"].nunique()),
            "silhouette": float(sil) if np.isfinite(sil) else np.nan,
            "avg_regime_duration": avg_duration(frame),
            "transition_diagonal_probability": float(np.trace(transitions) / transition_total)
            if transition_total
            else np.nan,
            "mean_confidence": float(post.max(axis=1).mean()),
            "final_loss": float(loss_history["loss"].iloc[-1]),
            "final_valid_anchor_pct": float(loss_history["valid_anchor_pct"].iloc[-1]),
            **reference_metrics(frame, reference),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(SAVE_DIR) / "guided_encoder_summary.csv", index=False)
    return summary


def save_loss_plot(loss_history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history["epoch"], loss_history["loss"], color="#2563EB", marker="o")
    ax.set_title("Phase 18 - HMM-Guided Encoder Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(Path(SAVE_DIR) / "guided_encoder_loss_curve.png", dpi=150)
    plt.close(fig)


def save_transition_plot(method: str, frame: pd.DataFrame) -> None:
    counts = transition_counts(frame)
    row_sum = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Phase 18 Transition Matrix - {method}")
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")
    ax.set_xticks(range(N_REGIMES))
    ax.set_yticks(range(N_REGIMES))
    for row in range(N_REGIMES):
        for col in range(N_REGIMES):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(Path(SAVE_DIR) / f"guided_encoder_transition_{method}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    guided = load_guided_samples(args.symbols)
    model, loss_history = train_guided_encoder(guided, args)
    embeddings, meta = extract_guided_embeddings(model, guided.matrices)
    assignments = fit_guided_assignments(embeddings, meta)
    summary = summarize_guided(embeddings, assignments, loss_history, args)

    save_loss_plot(loss_history)
    for method in GUIDED_METHODS:
        save_transition_plot(method, assignments[assignments["method"] == method])

    print("\nPhase 18 guided encoder summary:")
    print(summary.to_string(index=False))
    print("\nOK: HMM-guided encoder artifacts saved.")


if __name__ == "__main__":
    main()
