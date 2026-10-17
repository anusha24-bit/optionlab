"""Figures from results/results.json -> results/*.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.loads(Path("results/results.json").read_text())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = ["#2f5cff", "#e07a00", "#1b8a4b"]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
h = res["hedging"]
ax[0].loglog(h["rebalances"], h["error_std"], "o-", color=C[0], label="measured")
ref = np.array(h["rebalances"], float)
ax[0].loglog(ref, h["error_std"][0] / np.sqrt(ref), "--", color="gray", label="slope -1/2")
ax[0].set_xlabel("rebalances"); ax[0].set_ylabel("std of hedging error ($)"); ax[0].set_title(f"Delta hedging (fitted slope {h['loglog_slope']:.2f})"); ax[0].legend()

for (name, v), c in zip([(k, v) for k, v in res["monte_carlo"].items() if isinstance(v, dict)], C):
    n = [int(k) for k in v["se_by_N"]]
    ax[1].loglog(n, list(v["se_by_N"].values()), "o-", color=c, label=f"{name} ({v['loglog_slope']:.2f})")
ax[1].set_xlabel("paths N"); ax[1].set_ylabel("standard error ($)"); ax[1].set_title("Monte Carlo error"); ax[1].legend(fontsize=7)

cv = res["convergence"]
ax[2].loglog(cv["crr_steps"], cv["crr_abs_error"], "o-", color=C[0], label=f"binomial tree ({cv['crr_loglog_slope']:.2f})")
ax[2].loglog(cv["cn_grid"], cv["cn_abs_error"], "o-", color=C[1], label=f"Crank-Nicolson ({cv['cn_loglog_slope']:.2f})")
ax[2].set_xticks([25, 100, 400, 800]); ax[2].set_xticklabels(["25", "100", "400", "800"]); ax[2].minorticks_off(); ax[2].set_xlabel("steps / grid points"); ax[2].set_ylabel("abs error vs Black-Scholes ($)"); ax[2].set_title("Convergence"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/summary.png", dpi=150)
print("saved")
