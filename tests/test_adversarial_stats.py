import math

from eval.adversarial import stats


def test_paired_deltas():
    assert stats.paired_deltas([3, 5], [1, 2]) == [2, 3]


def test_paired_effect_mean_and_ci():
    effect = stats.paired_effect([10, 10, 10], [2, 2, 2])
    assert effect["n_pairs"] == 3
    assert effect["mean_delta"] == 8
    lo, hi = effect["ci95"]
    assert lo <= 8 <= hi


def test_null_band_zero_when_deterministic():
    band = stats.null_band([0.0, 0.0, 0.0])
    assert band["std"] == 0.0 and band["high"] == 0.0


def test_null_band_widens_with_noise():
    band = stats.null_band([-5.0, 0.0, 5.0])
    assert band["high"] > 0.0


def test_wilson_interval_bounds():
    wi = stats.wilson_interval(1, 10)
    assert 0.0 <= wi["low"] <= wi["rate"] <= wi["high"] <= 1.0
    assert math.isnan(stats.wilson_interval(0, 0)["rate"])


def test_rate_all_true():
    r = stats.rate([True, True, True])
    assert r["rate"] == 1.0 and r["n"] == 3
