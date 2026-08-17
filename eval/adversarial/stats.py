"""Statistical analysis for adversarial.

Paired A/B analysis with a bootstrap CI, Wilson intervals for binary rates, and an
A/A' null (noise-floor) band. Kept deliberately modular so multiple-comparison
correction can be layered on later. Causal (paired-Delta) and absolute-harm metrics
are computed by separate functions and never merged into a single score.
"""

import math
import random
from statistics import mean, median


def paired_deltas(a_welfares, b_welfares):
    if len(a_welfares) != len(b_welfares):
        raise ValueError("A and B must have equal length (matched pairs)")
    return [a - b for a, b in zip(a_welfares, b_welfares)]


def _bootstrap_ci(values, statistic, iterations=2000, alpha=0.05, rng=None):
    if not values:
        return (float("nan"), float("nan"))
    rng = rng or random.Random(0)
    n = len(values)
    samples = []
    for _ in range(iterations):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(resample))
    samples.sort()
    lo = samples[int((alpha / 2) * iterations)]
    hi = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def paired_effect(a_welfares, b_welfares, iterations=2000, seed=0):
    """Mean/median A->B welfare difference with a bootstrap 95% CI. A positive mean
    means B (dishonest) reduced buyer welfare relative to A (honest)."""
    deltas = paired_deltas(a_welfares, b_welfares)
    rng = random.Random(seed)
    return {
        "n_pairs": len(deltas),
        "mean_delta": mean(deltas) if deltas else float("nan"),
        "median_delta": median(deltas) if deltas else float("nan"),
        "ci95": _bootstrap_ci(deltas, mean, iterations=iterations, rng=rng),
        "deltas": deltas,
    }


def null_band(aa_prime_deltas, k=2.0):
    """A/A' noise floor: honest-vs-honest paired differences estimate the baseline
    stochastic variation. `high` is the upper edge of the band; a B effect counts as
    attack-induced only when its per-pair Delta exceeds this. Falls back to 0 when no
    null pairs are supplied."""
    if not aa_prime_deltas:
        return {"mean": 0.0, "std": 0.0, "high": 0.0, "low": 0.0, "n": 0}
    m = mean(aa_prime_deltas)
    if len(aa_prime_deltas) > 1:
        var = sum((d - m) ** 2 for d in aa_prime_deltas) / (len(aa_prime_deltas) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": m, "std": std, "high": m + k * std, "low": m - k * std, "n": len(aa_prime_deltas)}


def wilson_interval(successes, n, z=1.96):
    """Wilson score CI for a binomial proportion (robust at small n / extreme rates)."""
    if n == 0:
        return {"rate": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {"rate": p, "low": max(0.0, center - half), "high": min(1.0, center + half), "n": n}


def rate(flags):
    flags = list(flags)
    return wilson_interval(sum(1 for f in flags if f), len(flags))
