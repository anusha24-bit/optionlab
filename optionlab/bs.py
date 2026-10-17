"""Black-Scholes-Merton closed form for European options on a non-dividend-paying asset, with Greeks and implied volatility."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF


def _d1d2(S, K, T, r, sigma):
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sq
    return d1, d1 - sq


def price(S, K, T, r, sigma, kind="call"):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.asarray(T))
    if kind == "call":
        return S * ndtr(d1) - K * disc * ndtr(d2)
    if kind == "put":
        return K * disc * ndtr(-d2) - S * ndtr(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def _pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def greeks(S, K, T, r, sigma, kind="call"):
    """Returns dict with delta, gamma, vega (per 1.00 vol), theta (per year), rho (per 1.00 rate)."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    disc = np.exp(-r * np.asarray(T))
    sq = np.sqrt(T)
    gamma = _pdf(d1) / (S * sigma * sq)
    vega = S * _pdf(d1) * sq
    if kind == "call":
        delta = ndtr(d1)
        theta = -S * _pdf(d1) * sigma / (2 * sq) - r * K * disc * ndtr(d2)
        rho = K * T * disc * ndtr(d2)
    else:
        delta = ndtr(d1) - 1.0
        theta = -S * _pdf(d1) * sigma / (2 * sq) + r * K * disc * ndtr(-d2)
        rho = -K * T * disc * ndtr(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(market_price, S, K, T, r, kind="call", tol=1e-10, max_iter=100):
    """Newton's method on vega with a bisection safeguard. Raises ValueError if the price is outside no-arbitrage bounds."""
    disc = np.exp(-r * T)
    lo_bound = max(S - K * disc, 0.0) if kind == "call" else max(K * disc - S, 0.0)
    hi_bound = S if kind == "call" else K * disc
    if not (lo_bound - 1e-12 <= market_price <= hi_bound + 1e-12):
        raise ValueError("price outside no-arbitrage bounds; no implied volatility exists")
    lo, hi, sigma = 1e-6, 5.0, 0.3
    for _ in range(max_iter):
        diff = price(S, K, T, r, sigma, kind) - market_price
        if abs(diff) < tol:
            return float(sigma)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        v = greeks(S, K, T, r, sigma, kind)["vega"]
        step = sigma - diff / v if v > 1e-12 else None
        sigma = step if step is not None and lo < step < hi else 0.5 * (lo + hi)
    return float(sigma)
