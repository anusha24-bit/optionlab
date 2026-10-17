"""Run all experiments and write results/results.json. Usage: python run_experiments.py"""
import json
import time
from pathlib import Path

import numpy as np

from optionlab import bs, hedge, lattice, mc

S, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.2
out = {"params": {"S": S, "K": K, "T": T, "r": r, "sigma": SIG}}


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# 1. closed-form sanity checks
rng = np.random.default_rng(1)
Ss, Ks = rng.uniform(60, 140, 2000), rng.uniform(70, 130, 2000)
Ts, rs, sg = rng.uniform(0.1, 2, 2000), rng.uniform(0.0, 0.08, 2000), rng.uniform(0.1, 0.6, 2000)
parity = np.abs(bs.price(Ss, Ks, Ts, rs, sg, "call") - bs.price(Ss, Ks, Ts, rs, sg, "put") - (Ss - Ks * np.exp(-rs * Ts)))
h = 1e-4
g = bs.greeks(Ss, Ks, Ts, rs, sg, "call")
fd_delta = (bs.price(Ss + h, Ks, Ts, rs, sg) - bs.price(Ss - h, Ks, Ts, rs, sg)) / (2 * h)
fd_gamma = (bs.price(Ss + h, Ks, Ts, rs, sg) - 2 * bs.price(Ss, Ks, Ts, rs, sg) + bs.price(Ss - h, Ks, Ts, rs, sg)) / h**2
fd_vega = (bs.price(Ss, Ks, Ts, rs, sg + h) - bs.price(Ss, Ks, Ts, rs, sg - h)) / (2 * h)
out["closed_form"] = {"put_call_parity_max_abs_error": float(parity.max()),
                      "max_rel_err_delta_vs_finite_diff": float(np.max(np.abs(g["delta"] - fd_delta) / np.abs(g["delta"]).clip(1e-6))),
                      "max_abs_err_vega_vs_finite_diff": float(np.max(np.abs(g["vega"] - fd_vega))),
                      "max_rel_err_vega_vs_finite_diff_where_vega_gt_0.01": float(np.max((np.abs(g["vega"] - fd_vega) / g["vega"])[g["vega"] > 0.01])),
                      "max_abs_err_gamma_vs_finite_diff": float(np.max(np.abs(g["gamma"] - fd_gamma)))}
iv_err = max(abs(bs.implied_vol(bs.price(100, 100 + k, 0.5, 0.03, v), 100, 100 + k, 0.5, 0.03) - v) for k in (-20, -5, 0, 5, 20) for v in (0.1, 0.25, 0.5))
out["closed_form"]["implied_vol_roundtrip_max_abs_error"] = float(iv_err)
exact = float(bs.price(S, K, T, r, SIG))
out["bs_call_price"] = exact

# 2. Monte Carlo
Ns = [1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000]
variants = {"plain": dict(antithetic=False, control_variate=False), "antithetic": dict(antithetic=True, control_variate=False),
            "antithetic+control": dict(antithetic=True, control_variate=True)}
mcres = {}
for name, kw in variants.items():
    ses = [mc.mc_price(S, K, T, r, SIG, n=n, seed=3, **kw)[1] for n in Ns]
    mcres[name] = {"se_by_N": dict(zip(map(str, Ns), ses)), "loglog_slope": slope(Ns, ses)}
cover = []
for sd in range(300):
    p, se = mc.mc_price(S, K, T, r, SIG, n=20_000, seed=sd + 100)
    cover.append(abs(p - exact) <= 1.96 * se)
mcres["ci95_coverage_300_seeds_N20000"] = float(np.mean(cover))
mcres["variance_reduction_at_N256000"] = float((mcres["plain"]["se_by_N"]["256000"] / mcres["antithetic+control"]["se_by_N"]["256000"]) ** 2)
out["monte_carlo"] = mcres

# 3. Lattice and PDE convergence
steps = [25, 50, 100, 200, 400, 800]
crr_err = [abs(lattice.crr_price(S, K, T, r, SIG, "call", n) - exact) for n in steps]
grid = [25, 50, 100, 200, 400]
cn_err = [abs(lattice.cn_price(S, K, T, r, SIG, "call", n_space=n, n_time=n) - exact) for n in grid]
out["convergence"] = {"crr_steps": steps, "crr_abs_error": crr_err, "crr_loglog_slope": slope(steps, crr_err),
                      "cn_grid": grid, "cn_abs_error": cn_err, "cn_loglog_slope": slope(grid, cn_err),
                      "crr_local_order_finest_two": float(np.log(crr_err[-2] / crr_err[-1]) / np.log(steps[-1] / steps[-2])),
                      "cn_local_order_finest_two": float(np.log(cn_err[-2] / cn_err[-1]) / np.log(grid[-1] / grid[-2]))}
am = {}
for Sx in (80, 100, 120):
    eu = lattice.crr_price(Sx, K, T, r, SIG, "put", 1000, False)
    us = lattice.crr_price(Sx, K, T, r, SIG, "put", 1000, True)
    am[str(Sx)] = {"european_put": eu, "american_put": us, "early_exercise_premium": us - eu}
out["american_put_crr_1000_steps"] = am

# 4. Delta hedging
ns = [1, 2, 5, 10, 25, 50, 100, 250, 500]
base = {n: hedge.hedging_errors(n_rebalances=n, n_paths=20000, seed=7) for n in ns}
sd = [float(base[n].std()) for n in ns]
out["hedging"] = {"rebalances": ns, "error_std": sd, "error_mean": [float(base[n].mean()) for n in ns], "loglog_slope": slope(ns, sd),
                  "premium": float(bs.price(100.0, 100.0, 0.25, 0.03, 0.2, "call")), "setup": "S0=K=100, T=0.25, r=3%, sigma=20%, 20000 paths"}
wrong = {n: hedge.hedging_errors(n_rebalances=n, n_paths=20000, seed=7, sigma=0.30, hedge_sigma=0.20) for n in (50, 500)}
out["hedging_model_risk_true_vol_30_hedge_vol_20"] = {str(n): {"mean": float(e.mean()), "std": float(e.std())} for n, e in wrong.items()}
costly = {n: hedge.hedging_errors(n_rebalances=n, n_paths=20000, seed=7, cost_per_share=0.0005) for n in (10, 50, 500)}
out["hedging_with_5bp_transaction_cost"] = {str(n): {"mean": float(e.mean()), "std": float(e.std())} for n, e in costly.items()}

# 5. throughput of vectorised closed form
N = 1_000_000
Sv = np.random.default_rng(0).uniform(80, 120, N)
t0 = time.perf_counter(); bs.price(Sv, 100.0, 1.0, 0.05, 0.2); dt = time.perf_counter() - t0
out["throughput"] = {"bs_prices_per_second_vectorised": N / dt,
                     "mc_100k_paths_ms": None}
t0 = time.perf_counter(); mc.mc_price(S, K, T, r, SIG, n=100_000); out["throughput"]["mc_100k_paths_ms"] = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter(); lattice.crr_price(S, K, T, r, SIG, "call", 500); out["throughput"]["crr_500_steps_ms"] = (time.perf_counter() - t0) * 1000

Path("results").mkdir(exist_ok=True)
Path("results/results.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("closed_form", "convergence")}, indent=1)[:1500])
print("mc slopes", {k: v["loglog_slope"] for k, v in mcres.items() if isinstance(v, dict)}, "coverage", mcres["ci95_coverage_300_seeds_N20000"], "VR", mcres["variance_reduction_at_N256000"])
print("hedge", out["hedging"]["error_std"], out["hedging"]["loglog_slope"])
print("model risk", out["hedging_model_risk_true_vol_30_hedge_vol_20"])
print("costs", out["hedging_with_5bp_transaction_cost"])
print("am", am); print("thr", out["throughput"])
