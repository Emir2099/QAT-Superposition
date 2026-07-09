import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Load results from results folder
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results')

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(os.path.join(RESULTS_PATH, "results.json")) as f:
    results = json.load(f)
with open(os.path.join(RESULTS_PATH, "ptq_results.json")) as f:
    ptq_results = json.load(f)

precisions = ["fp32", "ternary", "binary"]
colors = {"fp32": "#2A6F97", "ternary": "#D97706", "binary": "#B91C1C"}
sparsity_levels = sorted(set(r["sparsity"] for r in results))

def agg(metric, precision):
    means, stds = [], []
    for s in sparsity_levels:
        vals = [r[metric] for r in results if r["precision"] == precision and r["sparsity"] == s]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

# --- Figure 1: mean dimensionality vs sparsity (QAT) ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for p in precisions:
    m, s = agg("mean_dimensionality", p)
    ax.plot(sparsity_levels, m, marker="o", color=colors[p], label=p)
    ax.fill_between(sparsity_levels, m - s, m + s, color=colors[p], alpha=0.15)
ax.set_xlabel("Sparsity")
ax.set_ylabel("Mean feature dimensionality  $\\bar{D}_i$")
ax.set_title("Superposition geometry is preserved under\nquantization-aware training (QAT)")
ax.legend(title="Weight precision")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_PATH, "fig1_dimensionality_qat.png"))
plt.close(fig)

# --- Figure 2: total interference vs sparsity (QAT) ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for p in precisions:
    m, s = agg("total_interference", p)
    ax.plot(sparsity_levels, m, marker="o", color=colors[p], label=p)
    ax.fill_between(sparsity_levels, m - s, m + s, color=colors[p], alpha=0.15)
ax.set_xlabel("Sparsity")
ax.set_ylabel("Total feature interference")
ax.set_title("Feature interference under QAT: near-identical\nacross precision levels")
ax.legend(title="Weight precision")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_PATH, "fig2_interference_qat.png"))
plt.close(fig)

# --- Figure 3: reconstruction loss vs sparsity (QAT), log scale ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for p in precisions:
    m, s = agg("final_loss", p)
    ax.plot(sparsity_levels, m, marker="o", color=colors[p], label=p)
ax.set_yscale("log")
ax.set_xlabel("Sparsity")
ax.set_ylabel("Importance-weighted reconstruction loss (log)")
ax.set_title("QAT models converge to nearly identical loss\nregardless of weight precision")
ax.legend(title="Weight precision")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_PATH, "fig3_loss_qat.png"))
plt.close(fig)

# --- Figure 4: PTQ degradation ratio vs sparsity ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mode, color in [("ternary", "#D97706"), ("binary", "#B91C1C")]:
    xs = [r["sparsity"] for r in ptq_results if r["mode"] == mode]
    ys = [r["loss_degradation_ratio"] for r in ptq_results if r["mode"] == mode]
    ax.plot(xs, ys, marker="o", color=color, label=mode)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="no degradation")
ax.set_xlabel("Sparsity")
ax.set_ylabel("Loss ratio: post-hoc quantized / fp32-trained")
ax.set_title("Post-training quantization (PTQ) degrades more\nas sparsity increases — unlike QAT")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_PATH, "fig4_ptq_degradation.png"))
plt.close(fig)

print("Saved 4 figures to results/")
