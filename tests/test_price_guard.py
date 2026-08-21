from guards.price_guard import (
    PlausibilityConfig,
    is_plausible,
    plausibility_block_reason,
)


def offer(unit_price=45.0, extraction_flag="none", vendor_id="vendor_a", delivery_days=8):
    return {
        "vendor_id": vendor_id,
        "unit_price": unit_price,
        "currency": "USD",
        "delivery_days": delivery_days,
        "confidence": "high",
        "quoted_basis": "unit",
        "extraction_flag": extraction_flag,
    }


CONFIG = PlausibilityConfig(absolute_floor=20.0)


def test_in_band_offer_is_plausible():
    assert is_plausible(offer(unit_price=45.0), [45.0], CONFIG) is True


def test_below_floor_blocked():
    assert plausibility_block_reason(offer(unit_price=1.0), [1.0], CONFIG) == "below_floor"


def test_at_floor_is_allowed():
    assert plausibility_block_reason(offer(unit_price=20.0), [20.0], CONFIG) is None


def test_just_below_floor_blocked():
    assert plausibility_block_reason(offer(unit_price=19.99), [19.99], CONFIG) == "below_floor"


def test_pricing_relevant_flag_blocks():
    for flag in ("missing_price", "non_usd", "tiered", "range", "conditional"):
        assert plausibility_block_reason(offer(extraction_flag=flag), [45.0], CONFIG) == "flagged"


def test_delivery_flag_is_not_a_pricing_block():
    assert plausibility_block_reason(offer(extraction_flag="unparseable_delivery"), [45.0], CONFIG) is None


def test_peer_prices_are_not_consulted_only_floor_and_flags_gate():
    # an above-floor price that is low relative to peers is NOT blocked: peer
    # comparison is not a security control (vendor-claim accuracy is out of scope)
    assert plausibility_block_reason(offer(unit_price=25.0), [45.0, 45.0, 25.0], CONFIG) is None
    # the configured floor still applies regardless of the peer distribution
    assert plausibility_block_reason(offer(unit_price=5.0), [45.0, 45.0, 5.0], CONFIG) == "below_floor"
