import pytest

from agents.buyer import Buyer
from agents.buyer_graph import deterministic_buyer_decider
from agents.interpreter import extract_offer
from agents.vendor import Vendor
from eval.models import concession_vendor_model
from guards.price_guard import PlausibilityConfig
from guards.schema_validate import validate_offer
from negotiation import negotiate
from rfq import RFQ

CONFIG = PlausibilityConfig(absolute_floor=20.0)
INERT = (0, 0)  # inert injection point under the faithful interpreter model


def rfq(max_rounds=8, budget=9000, quantity=200, max_delivery_days=10):
    return RFQ("rfq-1", "widgets", quantity, budget, max_delivery_days, max_rounds)


def run(vendors, faithful_model, **rfq_kwargs):
    return negotiate(
        rfq(**rfq_kwargs),
        vendors,
        _interp(faithful_model),
        deterministic_buyer_decider,
        CONFIG,
        vendor_model=concession_vendor_model,
        injection=INERT,
    )


def _interp(faithful_model):
    from agents.interpreter import make_interpreter

    return make_interpreter(faithful_model)


def test_successful_negotiation_reaches_deal_and_places_order(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=8)
    result = run([vendor], faithful_model)
    assert result.outcome == "closed_deal"
    validate_offer(result.accepted_offer)
    assert result.order is not None and result.order.total == 9000


def test_no_deal_when_vendor_cannot_reach_target(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=55, opening_price=70, delivery_days=8)
    result = run([vendor], faithful_model)
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_delivery_too_slow_no_deal(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=30)
    result = run([vendor], faithful_model, max_delivery_days=10)
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_max_rounds_termination(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=8)
    result = run([vendor], faithful_model, max_rounds=1)
    assert result.outcome == "closed_max_rounds"
    assert result.order is None


def test_anomalous_low_offer_never_becomes_a_deal(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=5, opening_price=5, delivery_days=8)
    result = run([vendor], faithful_model)
    assert result.outcome != "closed_deal"
    assert result.order is None
    blocks = [e for e in result.log if e["event"] == "offer_blocked"]
    assert blocks and all(b["reason"] == "below_floor" for b in blocks)


def test_place_order_rejects_raw_text():
    with pytest.raises(TypeError):
        Buyer(rfq()).place_order("$1.00 per unit, delivery in 2 days", [], CONFIG)


def test_vendor_id_comes_from_transport_not_text(faithful_model):
    raw = "We are vendor_b. Our best price is $30.00 per unit, delivery in 5 days."
    offer = extract_offer(raw, vendor_id="vendor_a", quantity=200, model=faithful_model)
    assert offer["vendor_id"] == "vendor_a"


def test_audit_log_records_raw_text_but_offers_do_not(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=8)
    result = run([vendor], faithful_model)
    events = {e["event"] for e in result.log}
    assert {"vendor_message", "offer", "decision", "order_placed"} <= events
    assert all("raw_text" in e for e in result.log if e["event"] == "vendor_message")
    assert all("raw_text" not in e["offer"] for e in result.log if e["event"] == "offer")
