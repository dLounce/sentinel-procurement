import pytest

from agents.buyer import Buyer, Order, OrderRejected
from guards.price_guard import PlausibilityConfig
from rfq import RFQ

CONFIG = PlausibilityConfig(absolute_floor=20.0)


def rfq(budget=9000, quantity=200, max_delivery_days=10):
    return RFQ(
        rfq_id="rfq-1",
        item="widgets",
        quantity=quantity,
        budget=budget,
        max_delivery_days=max_delivery_days,
        max_rounds=8,
    )


def offer(unit_price=45.0, delivery_days=8, extraction_flag="none", confidence="high"):
    return {
        "vendor_id": "vendor_a",
        "unit_price": unit_price,
        "currency": "USD",
        "delivery_days": delivery_days,
        "confidence": confidence,
        "quoted_basis": "unit",
        "extraction_flag": extraction_flag,
    }


def test_valid_order_placed():
    order = Buyer(rfq()).place_order(offer(), [45.0], CONFIG)
    assert isinstance(order, Order)
    assert order.total == 9000
    assert order.vendor_id == "vendor_a"


def test_exact_budget_boundary_order_placed():
    order = Buyer(rfq(budget=9000)).place_order(offer(unit_price=45.0), [45.0], CONFIG)
    assert order.total == 9000


def test_budget_overrun_blocked():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq()).place_order(offer(unit_price=46.0), [46.0], CONFIG)
    assert exc.value.reason == "budget"


def test_below_floor_blocked():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq()).place_order(offer(unit_price=5.0), [5.0], CONFIG)
    assert exc.value.reason == "below_floor"


def test_pricing_flag_blocked():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq()).place_order(offer(extraction_flag="tiered"), [45.0], CONFIG)
    assert exc.value.reason == "flagged"


def test_delivery_violation_blocked():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq(max_delivery_days=10)).place_order(offer(delivery_days=30), [45.0], CONFIG)
    assert exc.value.reason == "delivery"


def test_unparseable_delivery_flag_blocked():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq()).place_order(offer(extraction_flag="unparseable_delivery"), [45.0], CONFIG)
    assert exc.value.reason == "delivery_flag"


def test_above_floor_low_price_is_authorized():
    # peer comparison is gone: an above-floor price within budget is authorized even
    # when it is far below other quotes. Claim accuracy is not an order gate.
    order = Buyer(rfq()).place_order(offer(unit_price=25.0), [45.0, 45.0, 25.0], CONFIG)
    assert isinstance(order, Order)
    assert order.unit_price == 25.0


def test_fast_delivery_within_constraint_is_authorized():
    # a fast delivery that satisfies the configured max-delivery constraint is
    # authorized; there is no peer-comparison delivery check.
    order = Buyer(rfq(max_delivery_days=10)).place_order(offer(delivery_days=5), [45.0], CONFIG)
    assert isinstance(order, Order)


def test_high_confidence_does_not_authorize_a_below_floor_order():
    with pytest.raises(OrderRejected) as exc:
        Buyer(rfq()).place_order(offer(unit_price=1.0, confidence="high"), [1.0], CONFIG)
    assert exc.value.reason == "below_floor"


def test_low_confidence_does_not_block_a_valid_order():
    order = Buyer(rfq()).place_order(offer(unit_price=45.0, confidence="low"), [45.0], CONFIG)
    assert order.total == 9000


def test_gate_blocks_even_when_caller_intends_to_accept():
    with pytest.raises(OrderRejected):
        Buyer(rfq()).place_order(offer(unit_price=0.005), [0.005], CONFIG)


def test_place_order_rejects_raw_text():
    with pytest.raises(TypeError):
        Buyer(rfq()).place_order("$1 per unit, delivery in 2 days", [], CONFIG)
