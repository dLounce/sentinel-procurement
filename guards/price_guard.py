from dataclasses import dataclass
from statistics import median

# Chosen policy parameters (rationale in README). The outlier rule is a secondary
# cross-offer check; the absolute floor is the primary value-channel defense.
DEFAULT_OUTLIER_K = 0.4
DEFAULT_MIN_OFFERS_FOR_OUTLIER = 3

# extraction_flags that make an offer's price unusable for ordering. A pricing-
# relevant flag blocks the offer outright — the model never decides an ambiguous
# offer is "probably okay". (unparseable_delivery is a delivery concern, handled
# by the delivery check, not here.)
PRICING_RELEVANT_FLAGS = frozenset(
    {"missing_price", "non_usd", "tiered", "range", "conditional"}
)


@dataclass(frozen=True)
class PlausibilityConfig:
    """Deterministic price-guard policy.

    absolute_floor is a scenario-specific configured threshold supplied
    independently of vendor responses — never derived from vendor offers or
    vendor text, and not evaluation ground truth. It has no universal default and
    must be set per scenario.
    """

    absolute_floor: float
    outlier_k: float = DEFAULT_OUTLIER_K
    min_offers_for_outlier: int = DEFAULT_MIN_OFFERS_FOR_OUTLIER


def plausibility_block_reason(offer: dict, round_unit_prices, config: PlausibilityConfig):
    """Return a string reason the offer must not be ordered, or None if plausible.

    Deterministic only: it reads the offer's price and flag and the round's
    unit-price distribution. It never reads confidence or any model reasoning.
    """
    if offer["extraction_flag"] in PRICING_RELEVANT_FLAGS:
        return "flagged"

    price = offer["unit_price"]
    if price < config.absolute_floor:
        return "below_floor"

    valid = [p for p in round_unit_prices if p is not None]
    if len(valid) >= config.min_offers_for_outlier:
        if price < median(valid) * (1 - config.outlier_k):
            return "outlier"

    return None


def is_plausible(offer: dict, round_unit_prices, config: PlausibilityConfig) -> bool:
    return plausibility_block_reason(offer, round_unit_prices, config) is None
