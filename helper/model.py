"""
Toy model of superposition (Elhage et al. 2022 setup) with weight quantization.

Model: x_hat = ReLU(W^T W x + b)
- W: (n_features, m_hidden), m_hidden << n_features
- Features are synthetic, independently sparse, with exponentially decaying importance
- Loss: importance-weighted MSE

Quantization: fake-quantization with straight-through estimator (STE) on W.
  - 'fp32'   : no quantization
  - 'binary' : W_q = alpha * sign(W), alpha = mean(|W|) per-row
  - 'ternary': W_q in {-alpha, 0, alpha} via threshold-based ternarization (TWN-style),
               threshold = 0.7 * mean(|W|) per-row
"""
import torch
import torch.nn as nn
import numpy as np


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, mode):
        if mode == "fp32":
            return w
        elif mode == "binary":
            alpha = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            return alpha * torch.sign(w)
        elif mode == "ternary":
            thresh = 0.7 * w.abs().mean(dim=1, keepdim=True)
            alpha = (w.abs() * (w.abs() > thresh)).sum(dim=1, keepdim=True) / (
                (w.abs() > thresh).sum(dim=1, keepdim=True).clamp(min=1)
            )
            q = torch.zeros_like(w)
            q[w > thresh] = 1.0
            q[w < -thresh] = -1.0
            return alpha * q
        else:
            raise ValueError(mode)

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through: gradient passes unchanged to the fp32 latent weight
        return grad_output, None


def quantize(w, mode):
    return STEQuantize.apply(w, mode)


class ToyModel(nn.Module):
    def __init__(self, n_features, m_hidden, precision="fp32"):
        super().__init__()
        self.n_features = n_features
        self.m_hidden = m_hidden
        self.precision = precision
        self.W = nn.Parameter(torch.randn(n_features, m_hidden) * (1.0 / np.sqrt(m_hidden)))
        self.b = nn.Parameter(torch.zeros(n_features))

    def effective_W(self):
        return quantize(self.W, self.precision)

    def forward(self, x):
        Wq = self.effective_W()
        hidden = x @ Wq                       # (batch, m_hidden)
        out = hidden @ Wq.T + self.b          # (batch, n_features)
        return torch.relu(out)


def make_importance(n_features, decay=0.9):
    # feature i has importance decay^i (feature 0 most important)
    return decay ** torch.arange(n_features).float()


def sample_batch(batch_size, n_features, sparsity, device):
    # each feature independently present with prob (1 - sparsity), magnitude ~ U(0,1)
    presence = (torch.rand(batch_size, n_features, device=device) > sparsity).float()
    magnitude = torch.rand(batch_size, n_features, device=device)
    return presence * magnitude


def train_model(n_features, m_hidden, sparsity, precision, importance, steps=4000,
                 batch_size=1024, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = ToyModel(n_features, m_hidden, precision=precision).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    importance = importance.to(device)

    for step in range(steps):
        x = sample_batch(batch_size, n_features, sparsity, device)
        x_hat = model(x)
        loss = (importance * (x - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x = sample_batch(4096, n_features, sparsity, device)
        x_hat = model(x)
        final_loss = (importance * (x - x_hat) ** 2).mean().item()

    return model, final_loss
