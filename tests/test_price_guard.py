from guards.price_guard import (
    DEFAULT_MIN_OFFERS_FOR_OUTLIER,
    DEFAULT_OUTLIER_K,
    PlausibilityConfig,
    is_plausible,
    plausibility_block_reason,
)


def offer(unit_price=45.0, extraction_flag="none", vendor_id="vendor_a"):
    return {
        "vendor_id": vendor_id,
        "unit_price": unit_price,
        "currency": "USD",
        "delivery_days": 8,
        "confidence": "high",
        "quoted_basis": "unit",
        "extraction_flag": extraction_flag,
    }


CONFIG = PlausibilityConfig(absolute_floor=20.0)


def test_defaults_match_decision():
    assert DEFAULT_OUTLIER_K == 0.4
    assert DEFAULT_MIN_OFFERS_FOR_OUTLIER == 3


def test_in_band_offer_is_plausible():
    assert is_plausible(offer(unit_price=45.0), [45.0], CONFIG) is True


def test_below_floor_blocked():
    assert plausibility_block_reason(offer(unit_price=1.0), [1.0], CONFIG) == "below_floor"


def test_at_floor_is_allowed():
    # floor is a strict minimum: exactly at the floor is not "below" it
    assert plausibility_block_reason(offer(unit_price=20.0), [20.0], CONFIG) is None


def test_just_below_floor_blocked():
    assert plausibility_block_reason(offer(unit_price=19.99), [19.99], CONFIG) == "below_floor"


def test_pricing_relevant_flag_blocks():
    for flag in ("missing_price", "non_usd", "tiered", "range", "conditional"):
        assert plausibility_block_reason(offer(extraction_flag=flag), [45.0], CONFIG) == "flagged"


def test_delivery_flag_is_not_a_pricing_block():
    # unparseable_delivery is a delivery concern, not a price-guard block
    assert plausibility_block_reason(offer(extraction_flag="unparseable_delivery"), [45.0], CONFIG) is None


def test_outlier_rule_applies_at_exactly_three_offers():
    prices = [45.0, 45.0, 20.0]  # median 45, threshold 45*0.6 = 27
    assert plausibility_block_reason(offer(unit_price=20.0), prices, CONFIG) == "outlier"


def test_outlier_threshold_boundary():
    prices = [45.0, 45.0, 27.0]  # threshold exactly 27.0; not below -> allowed
    assert plausibility_block_reason(offer(unit_price=27.0), prices, CONFIG) is None
    prices_low = [45.0, 45.0, 26.99]
    assert plausibility_block_reason(offer(unit_price=26.99), prices_low, CONFIG) == "outlier"


def test_fewer_than_three_offers_skips_outlier_rule():
    # 30 would be an outlier vs a 45 median, but with only 2 offers the rule is
    # not applied; 30 is above the floor so it is allowed.
    assert plausibility_block_reason(offer(unit_price=30.0), [45.0, 30.0], CONFIG) is None


def test_floor_still_applies_with_fewer_than_three_offers():
    assert plausibility_block_reason(offer(unit_price=5.0), [45.0, 5.0], CONFIG) == "below_floor"
