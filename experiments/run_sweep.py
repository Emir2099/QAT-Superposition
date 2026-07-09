import torch
import json
import os
import sys
import numpy as np

# Add helper directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'helper'))

from model import train_model, make_importance
from metrics import feature_dimensionality, total_interference, active_feature_fraction

# Save results to results folder
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

device = "cuda" if torch.cuda.is_available() else "cpu"

N_FEATURES = 40
M_HIDDEN = 5
IMPORTANCE_DECAY = 0.9
SPARSITY_LEVELS = [0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]
PRECISIONS = ["fp32", "ternary", "binary"]
SEEDS = [0, 1]
STEPS = 1200

importance = make_importance(N_FEATURES, IMPORTANCE_DECAY)

results = []

for precision in PRECISIONS:
    for sparsity in SPARSITY_LEVELS:
        for seed in SEEDS:
            model, final_loss = train_model(
                N_FEATURES, M_HIDDEN, sparsity, precision, importance,
                steps=STEPS, device=device, seed=seed,
            )
            with torch.no_grad():
                W = model.effective_W().detach().cpu()
            D = feature_dimensionality(W)
            interference = total_interference(W)
            active_frac = active_feature_fraction(W)

            results.append({
                "precision": precision,
                "sparsity": sparsity,
                "seed": seed,
                "final_loss": final_loss,
                "mean_dimensionality": D.mean().item(),
                "median_dimensionality": D.median().item(),
                "total_interference": interference,
                "active_feature_fraction": active_frac,
            })
            print(f"[{precision:8s}] sparsity={sparsity:.2f} seed={seed} "
                  f"loss={final_loss:.4f} meanD={D.mean().item():.3f} "
                  f"interference={interference:.4f}")

with open(os.path.join(RESULTS_PATH, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Saved to results/results.json")
