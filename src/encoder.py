import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    Lightweight Temporal Fusion Transformer encoder.
    Input:  (batch, window, n_features)
    Output: (batch, latent_dim) — the regime embedding
    
    Architecture:
      1. Linear projection → d_model
      2. Positional encoding
      3. 2-layer Transformer encoder
      4. Mean pooling over time
      5. MLP projection head → latent_dim
    """
    def __init__(
        self,
        n_features: int = 22,
        d_model:    int = 64,
        n_heads:    int = 4,
        n_layers:   int = 2,
        latent_dim: int = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # Project raw features to model dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projection head: maps pooled representation → latent space
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        B, T, _ = x.shape

        # Project to model dim
        x = self.input_proj(x)                      # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_enc[:, :T, :]              # (B, T, d_model)

        # Transformer
        x = self.transformer(x)                     # (B, T, d_model)

        # Mean pool over time dimension
        x = x.mean(dim=1)                           # (B, d_model)

        # Project to latent space
        z = self.proj_head(x)                       # (B, latent_dim)

        # L2 normalize — important for contrastive loss
        z = F.normalize(z, dim=-1)

        return z


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    This is the SimCLR contrastive loss.
    
    For each anchor, its positive is the adjacent window.
    All other items in the batch are treated as negatives.
    Higher temperature = softer distribution.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        B = z_anchor.shape[0]

        # Concatenate anchor and positive embeddings
        z = torch.cat([z_anchor, z_positive], dim=0)   # (2B, latent_dim)

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.temperature       # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Labels: for anchor i, positive is at index i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B),
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss