"""Binomial tree (Cox-Ross-Rubinstein) and Crank-Nicolson finite differences for the Black-Scholes PDE."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def crr_price(S, K, T, r, sigma, kind="call", steps=500, american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (2 * j - steps)
    v = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for i in range(steps - 1, -1, -1):
        v = disc * (p * v[1 : i + 2] + (1 - p) * v[0 : i + 1])
        if american:
            Si = S * u ** (2 * np.arange(i + 1) - i)
            ex = np.maximum(Si - K, 0.0) if kind == "call" else np.maximum(K - Si, 0.0)
            v = np.maximum(v, ex)
    return float(v[0])


def cn_price(S, K, T, r, sigma, kind="call", n_space=400, n_time=400, s_max_mult=4.0):
    """European option by Crank-Nicolson on a uniform grid in S. Boundary conditions: call V(0)=0, V(Smax)=Smax-K e^{-r tau};
    put V(0)=K e^{-r tau}, V(Smax)=0. Linear interpolation reads the value at S."""
    s_max = s_max_mult * max(S, K)
    ds = s_max / n_space
    dt = T / n_time
    s = np.linspace(0, s_max, n_space + 1)
    v = np.maximum(s - K, 0.0) if kind == "call" else np.maximum(K - s, 0.0)
    i = np.arange(1, n_space)
    a = 0.25 * dt * (sigma**2 * i**2 - r * i)      # lower
    b = -0.5 * dt * (sigma**2 * i**2 + r)          # diagonal
    c = 0.25 * dt * (sigma**2 * i**2 + r * i)      # upper
    # (I - L) v_new = (I + L) v_old, where L = tridiag(a, b, c)
    ab = np.zeros((3, n_space - 1))
    ab[0, 1:] = -c[:-1]
    ab[1, :] = 1 - b
    ab[2, :-1] = -a[1:]
    for k in range(1, n_time + 1):
        tau_new, tau_old = k * dt, (k - 1) * dt
        if kind == "call":
            lo_old, lo_new = 0.0, 0.0
            hi_old, hi_new = s_max - K * np.exp(-r * tau_old), s_max - K * np.exp(-r * tau_new)
        else:
            lo_old, lo_new = K * np.exp(-r * tau_old), K * np.exp(-r * tau_new)
            hi_old, hi_new = 0.0, 0.0
        rhs = a * v[:-2] + (1 + b) * v[1:-1] + c * v[2:]
        rhs[0] += a[0] * lo_new
        rhs[-1] += c[-1] * hi_new
        v[1:-1] = solve_banded((1, 1), ab, rhs)
        v[0], v[-1] = lo_new, hi_new
    return float(np.interp(S, s, v))
