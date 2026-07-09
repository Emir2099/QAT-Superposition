"""
Metrics from Elhage et al. (2022) "Toy Models of Superposition":

Per-feature dimensionality:
    D_i = ||W_i||^2 / sum_j ( (W_i . W_j) / ||W_i|| )^2
        = ||W_i||^4 / sum_j (W_i . W_j)^2

Interpretation: D_i = 1 means feature i has a dedicated orthogonal dimension.
D_i -> 0 means feature i is packed into heavy superposition with other features.

Total interference: sum of squared off-diagonal entries of W W^T (normalized),
a scalar summary of how much features interfere with each other.
"""
import torch


def feature_dimensionality(W):
    # W: (n_features, m_hidden)
    norms_sq = (W ** 2).sum(dim=1)  # ||W_i||^2, shape (n,)
    G = W @ W.T                     # Gram matrix, (n, n), G_ij = W_i . W_j
    denom = (G ** 2).sum(dim=1)     # sum_j (W_i . W_j)^2
    D = (norms_sq ** 2) / denom.clamp(min=1e-12)
    return D  # shape (n,)


def total_interference(W):
    n = W.shape[0]
    G = W @ W.T
    norms = torch.sqrt((W ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-8)
    G_norm = G / (norms @ norms.T)  # cosine similarity matrix
    off_diag_sq = (G_norm ** 2).sum() - (G_norm.diag() ** 2).sum()
    return (off_diag_sq / (n * (n - 1))).item()


def active_feature_fraction(W, threshold=0.05):
    # fraction of features whose representation norm is non-negligible
    norms = torch.sqrt((W ** 2).sum(dim=1))
    return (norms > threshold * norms.max()).float().mean().item()
