"""Figures for the chapter (print-safe: two hues + marker shape, direct labels). Run after run_experiments.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a5d9f", "#c4651d", "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
d1 = pd.read_csv(R / "e1_pairs.csv"); d2 = pd.read_csv(R / "e2_pairs.csv"); d3 = pd.read_csv(R / "e3_pairs.csv")
lr, rf = d1[d1.model == "LR"].reset_index(drop=True), d1[d1.model == "RF"].reset_index(drop=True)

# Fig A: prevalence shift vs calibration error
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.scatter(lr["shift"].abs(), lr.ece, s=28, color=BLUE, zorder=3)
for i, (_, r) in enumerate(lr.sort_values("ece").tail(3).iloc[::-1].iterrows()):
    ax.annotate(f"{r.train} → {r.test.split('-')[-1]}", (abs(r["shift"]), r.ece), xytext=(-6, -3 - 11 * i), textcoords="offset points", ha="right", fontsize=7.5, color=INK)
ax.set_xlabel("|prevalence(r+1) − prevalence(r)| of defective classes"); ax.set_ylabel("ECE of logistic model on release r+1")
fig.tight_layout(); fig.savefig(F / "fig_e1_shift_ece.png", dpi=300); plt.close(fig)

# Fig B: AUC difference vs net-benefit difference (b = 5)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
dA, dN = (lr.auc - rf.auc).to_numpy(), (lr["nb_b5"] - rf["nb_b5"]).to_numpy()
disc = np.sign(dA) != np.sign(dN)
ax.axhline(0, color=MUTED, lw=.8); ax.axvline(0, color=MUTED, lw=.8)
ax.scatter(dA[~disc], dN[~disc], s=30, marker="o", color=BLUE, zorder=3, label="same order")
ax.scatter(dA[disc], dN[disc], s=38, marker="D", color=ORANGE, zorder=3, label="order reversed")
ax.set_xlabel("AUC(LR) − AUC(RF)"); ax.set_ylabel("Net benefit(LR) − net benefit(RF),  b = 5")
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig_e1_auc_vs_nb.png", dpi=300); plt.close(fig)

# Fig C: aggregation experiment
S = json.loads((R / "summary.json").read_text())["E2"]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
names = [("max_minus_mean", "max − mean"), ("top3_minus_mean", "top-3 mean − mean"), ("share_ge_.5_minus_mean", "share(p≥0.5) − mean")]
for i, (k, lab) in enumerate(names):
    m, (lo, hi) = S[k]["mean"], S[k]["ci95"]
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round"); ax.scatter([m], [i], s=34, color=BLUE, zorder=3)
    ax.text(hi + .006, i, f"{m:+.3f}", va="center", fontsize=8, color=INK)
ax.axvline(0, color=MUTED, lw=.8); ax.set_yticks(range(3)); ax.set_yticklabels([n[1] for n in names]); ax.invert_yaxis()
ax.set_xlabel("Δρ vs. mean aggregation (mean of 28 pairs, 95% bootstrap CI)")
fig.tight_layout(); fig.savefig(F / "fig_e2_aggregation.png", dpi=300); plt.close(fig)

# Fig D: rubric rank stability
fig, ax = plt.subplots(figsize=(5.6, 3.2))
cols = [("p_winner_differs", "raw attributes"), ("p_winner_differs_minmax", "min–max normalised"), ("p_winner_differs_constrained", "raw + recall ≥ 0.5 gate")]
rng = np.random.default_rng(1)
for i, (k, lab) in enumerate(cols):
    y = d3[k].to_numpy(); ax.scatter(i + rng.uniform(-.12, .12, len(y)), y, s=16, color=BLUE, alpha=.8, zorder=3)
    ax.plot([i - .25, i + .25], [np.median(y)] * 2, color=ORANGE, lw=2.4, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in cols]); ax.set_ylim(0, 1)
ax.set_ylabel("P(winner ≠ equal-weight winner)")
fig.tight_layout(); fig.savefig(F / "fig_e3_rank_stability.png", dpi=300); plt.close(fig)
print("figures written:", sorted(p.name for p in F.glob("*.png")))
print("discordant share b=5:", disc.mean())
