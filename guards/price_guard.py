from dataclasses import dataclass

# Extraction flags that make an offer's price unusable for ordering. A pricing-
# relevant flag blocks the offer outright — the model never decides an ambiguous
# offer is "probably okay". (unparseable_delivery is a delivery concern, handled at
# the order gate, not here.)
PRICING_RELEVANT_FLAGS = frozenset(
    {"missing_price", "non_usd", "tiered", "range", "conditional"}
)


@dataclass(frozen=True)
class PlausibilityConfig:
    """Deterministic order-policy thresholds.

    ``absolute_floor`` is a scenario-specific configured minimum unit price supplied
    independently of vendor responses — never derived from vendor offers or vendor
    text, and never evaluation ground truth. It has no universal default and must be
    set per scenario.
    """

    absolute_floor: float


def plausibility_block_reason(offer: dict, round_unit_prices, config: PlausibilityConfig):
    """Return a deterministic reason the offer must not be ordered, or None.

    Enforces configured order policy only: a pricing-relevant extraction flag, or a
    unit price below the configured absolute floor. It deliberately does NOT compare a
    vendor's claim against other vendors' claims or against hidden truth — vendor-claim
    accuracy is not a security property. ``round_unit_prices`` is accepted for
    interface stability with existing callers but is not consulted.
    """
    if offer["extraction_flag"] in PRICING_RELEVANT_FLAGS:
        return "flagged"

    if offer["unit_price"] < config.absolute_floor:
        return "below_floor"

    return None


def is_plausible(offer: dict, round_unit_prices, config: PlausibilityConfig) -> bool:
    return plausibility_block_reason(offer, round_unit_prices, config) is None
