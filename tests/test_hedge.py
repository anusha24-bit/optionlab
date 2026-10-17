import numpy as np
import pytest

from optionlab import hedge


def test_hedging_error_has_near_zero_mean_without_frictions():
    e = hedge.hedging_errors(n_rebalances=100, n_paths=20000, seed=1)
    assert abs(e.mean()) < 4 * e.std() / np.sqrt(len(e))


def test_hedging_error_std_falls_like_inverse_sqrt_of_rebalances():
    a = hedge.hedging_errors(n_rebalances=25, n_paths=20000, seed=2).std()
    b = hedge.hedging_errors(n_rebalances=100, n_paths=20000, seed=2).std()
    assert a / b == pytest.approx(2.0, rel=0.2)


def test_transaction_costs_make_the_seller_lose_more_as_rebalancing_rises():
    few = hedge.hedging_errors(n_rebalances=10, n_paths=10000, seed=3, cost_per_share=0.0005).mean()
    many = hedge.hedging_errors(n_rebalances=200, n_paths=10000, seed=3, cost_per_share=0.0005).mean()
    assert many < few < 0


def test_hedging_with_too_low_a_volatility_loses_money_on_average():
    e = hedge.hedging_errors(n_rebalances=100, n_paths=10000, seed=4, sigma=0.30, hedge_sigma=0.20)
    assert e.mean() < -1.0
