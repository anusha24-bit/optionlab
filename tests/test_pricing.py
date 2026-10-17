import numpy as np
import pytest

from optionlab import bs, lattice, mc

S, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


def test_known_reference_value():
    # Widely quoted value for S=K=100, T=1, r=5%, sigma=20% (call ~ 10.4506); also cross-checked by the independent Monte Carlo and PDE prices
    assert bs.price(S, K, T, r, SIG, "call") == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity_holds_across_random_inputs():
    rng = np.random.default_rng(0)
    Ss, Ks = rng.uniform(50, 150, 500), rng.uniform(50, 150, 500)
    Ts, rs, sg = rng.uniform(0.05, 3, 500), rng.uniform(0, 0.1, 500), rng.uniform(0.05, 0.8, 500)
    lhs = bs.price(Ss, Ks, Ts, rs, sg, "call") - bs.price(Ss, Ks, Ts, rs, sg, "put")
    assert np.allclose(lhs, Ss - Ks * np.exp(-rs * Ts), atol=1e-10)


def test_price_respects_no_arbitrage_bounds():
    p = bs.price(S, 90.0, T, r, SIG, "call")
    assert max(S - 90 * np.exp(-r * T), 0) <= p <= S


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    h = 1e-4
    g = bs.greeks(S, K, T, r, SIG, kind)
    fd_delta = (bs.price(S + h, K, T, r, SIG, kind) - bs.price(S - h, K, T, r, SIG, kind)) / (2 * h)
    fd_vega = (bs.price(S, K, T, r, SIG + h, kind) - bs.price(S, K, T, r, SIG - h, kind)) / (2 * h)
    fd_gamma = (bs.price(S + h, K, T, r, SIG, kind) - 2 * bs.price(S, K, T, r, SIG, kind) + bs.price(S - h, K, T, r, SIG, kind)) / h**2
    fd_theta = -(bs.price(S, K, T + h, r, SIG, kind) - bs.price(S, K, T - h, r, SIG, kind)) / (2 * h)
    assert g["delta"] == pytest.approx(fd_delta, abs=1e-6)
    assert g["vega"] == pytest.approx(fd_vega, abs=1e-5)
    assert g["gamma"] == pytest.approx(fd_gamma, abs=1e-4)
    assert g["theta"] == pytest.approx(fd_theta, abs=1e-4)


def test_implied_vol_roundtrip_and_bounds():
    for v in (0.05, 0.2, 0.6, 1.2):
        assert bs.implied_vol(bs.price(S, 110, 0.5, r, v), S, 110, 0.5, r) == pytest.approx(v, abs=1e-8)
    with pytest.raises(ValueError):
        bs.implied_vol(S + 1.0, S, K, T, r)  # call worth more than the stock: impossible


def test_monte_carlo_is_within_four_standard_errors():
    exact = bs.price(S, K, T, r, SIG)
    p, se = mc.mc_price(S, K, T, r, SIG, n=200_000, seed=1)
    assert abs(p - exact) < 4 * se and se < 0.03


def test_variance_reduction_actually_reduces_standard_error():
    _, se_plain = mc.mc_price(S, K, T, r, SIG, n=100_000, seed=2, antithetic=False, control_variate=False)
    _, se_both = mc.mc_price(S, K, T, r, SIG, n=100_000, seed=2)
    assert se_both < 0.8 * se_plain


def test_monte_carlo_error_shrinks_like_inverse_sqrt_n():
    ns = [4_000, 64_000]
    ses = [mc.mc_price(S, K, T, r, SIG, n=n, seed=3)[1] for n in ns]
    assert ses[0] / ses[1] == pytest.approx(4.0, rel=0.25)


def test_binomial_and_pde_converge_to_black_scholes():
    exact = bs.price(S, K, T, r, SIG)
    assert abs(lattice.crr_price(S, K, T, r, SIG, "call", 800) - exact) < 5e-3
    assert abs(lattice.cn_price(S, K, T, r, SIG, "call", n_space=300, n_time=300) - exact) < 1e-2
    # doubling the tree depth should roughly halve the error (first order)
    e1 = abs(lattice.crr_price(S, K, T, r, SIG, "call", 200) - exact)
    e2 = abs(lattice.crr_price(S, K, T, r, SIG, "call", 400) - exact)
    assert 1.5 < e1 / e2 < 2.6


def test_american_put_is_worth_at_least_european_and_at_least_intrinsic():
    for s0 in (70, 100, 130):
        eu = lattice.crr_price(s0, K, T, r, SIG, "put", 500, False)
        am = lattice.crr_price(s0, K, T, r, SIG, "put", 500, True)
        assert am >= eu - 1e-12 and am >= max(K - s0, 0) - 1e-12


def test_american_call_without_dividends_equals_european_call():
    eu = lattice.crr_price(S, K, T, r, SIG, "call", 500, False)
    am = lattice.crr_price(S, K, T, r, SIG, "call", 500, True)
    assert am == pytest.approx(eu, abs=1e-9)
