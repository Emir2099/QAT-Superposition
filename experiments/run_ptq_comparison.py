"""
QAT vs PTQ comparison:
For each sparsity level, train an fp32 model, then compare:
  (a) QAT: model trained from scratch with ternary/binary STE quantization (already in results.json)
  (b) PTQ: take the fp32-trained weights and quantize them post-hoc, no retraining
Measure reconstruction loss and superposition geometry for both.
"""
import torch
import json
import os
import sys

# Add helper directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'helper'))

from model import train_model, make_importance, quantize, sample_batch
from metrics import feature_dimensionality, total_interference

# Save results to results folder
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

device = "cpu"
N_FEATURES = 40
M_HIDDEN = 5
IMPORTANCE_DECAY = 0.9
SPARSITY_LEVELS = [0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]
STEPS = 1200
SEED = 0

importance = make_importance(N_FEATURES, IMPORTANCE_DECAY)
results = []

for sparsity in SPARSITY_LEVELS:
    model, fp32_loss = train_model(
        N_FEATURES, M_HIDDEN, sparsity, "fp32", importance,
        steps=STEPS, device=device, seed=SEED,
    )
    W_fp32 = model.W.detach().clone()

    with torch.no_grad():
        x = sample_batch(4096, N_FEATURES, sparsity, device)

    for mode in ["ternary", "binary"]:
        W_ptq = quantize(W_fp32, mode)
        with torch.no_grad():
            hidden = x @ W_ptq
            out = torch.relu(hidden @ W_ptq.T + model.b)
            ptq_loss = (importance * (x - out) ** 2).mean().item()
        D = feature_dimensionality(W_ptq)
        interference = total_interference(W_ptq)
        results.append({
            "sparsity": sparsity,
            "mode": mode,
            "ptq_loss": ptq_loss,
            "fp32_loss_same_weights": fp32_loss,
            "loss_degradation_ratio": ptq_loss / max(fp32_loss, 1e-8),
            "mean_dimensionality": D.mean().item(),
            "total_interference": interference,
        })
        print(f"sparsity={sparsity:.2f} mode={mode:8s} ptq_loss={ptq_loss:.4f} "
              f"fp32_loss={fp32_loss:.4f} degradation={ptq_loss/max(fp32_loss,1e-8):.2f}x "
              f"meanD={D.mean().item():.3f} interference={interference:.4f}")

with open(os.path.join(RESULTS_PATH, "ptq_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/ptq_results.json")
