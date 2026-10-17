"""Discrete delta hedging of a short European call under GBM with a flat volatility.

The seller receives the Black-Scholes premium at t=0 and holds delta shares, rebalanced n_rebalances times at equal intervals,
financing everything at the risk-free rate. Hedging error = final portfolio value minus the option payoff owed.
"""
from __future__ import annotations

import numpy as np

from . import bs


def hedging_errors(S0=100.0, K=100.0, T=0.25, r=0.03, sigma=0.2, n_rebalances=50, n_paths=20000, seed=0, mu=None,
                   hedge_sigma=None, cost_per_share=0.0):
    """Returns the array of per-path hedging errors (one per simulated path).

    mu: real-world drift of the stock (defaults to r). hedge_sigma: vol used to compute deltas (defaults to sigma, i.e. no model error).
    cost_per_share: proportional transaction cost as a fraction of traded notional.
    """
    mu = r if mu is None else mu
    hv = sigma if hedge_sigma is None else hedge_sigma
    rng = np.random.default_rng(seed)
    dt = T / n_rebalances
    S = np.full(n_paths, S0)
    premium = float(bs.price(S0, K, T, r, hv, "call"))
    delta = bs.greeks(S, K, T, r, hv, "call")["delta"]
    cash = premium - delta * S - cost_per_share * np.abs(delta) * S
    for i in range(1, n_rebalances + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt)
        if i < n_rebalances:
            new_delta = bs.greeks(S, K, T - i * dt, r, hv, "call")["delta"]
            trade = new_delta - delta
            cash = cash - trade * S - cost_per_share * np.abs(trade) * S
            delta = new_delta
    portfolio = cash + delta * S
    return portfolio - np.maximum(S - K, 0.0)
