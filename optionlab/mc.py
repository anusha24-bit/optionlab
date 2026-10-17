"""Monte Carlo pricing of a European option under geometric Brownian motion, with variance reduction and a standard error."""
from __future__ import annotations

import numpy as np


def mc_price(S, K, T, r, sigma, kind="call", n=100_000, seed=0, antithetic=True, control_variate=True):
    """Returns (price, standard_error). Terminal value is simulated exactly (no time stepping).

    antithetic: pair each normal draw z with -z.
    control_variate: use the discounted terminal asset price, whose expectation is exactly S, as a control.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n // 2 if antithetic else n)
    if antithetic:
        z = np.concatenate([z, -z])
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    x = disc * payoff
    if control_variate:
        c = disc * ST  # E[c] = S
        beta = np.cov(x, c)[0, 1] / np.var(c, ddof=1)
        x = x - beta * (c - S)
    if antithetic:
        # average each antithetic pair first; pairs are the independent units
        half = len(x) // 2
        x = 0.5 * (x[:half] + x[half:])
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
