import pytest

from agents.buyer import Buyer
from agents.interpreter import extract_offer, make_interpreter
from agents.vendor import Vendor
from guards.schema_validate import validate_offer
from negotiation import negotiate
from rfq import RFQ


def rfq(max_rounds=8, budget=9000, quantity=200, max_delivery_days=10):
    return RFQ(
        rfq_id="rfq-1",
        item="widgets",
        quantity=quantity,
        budget=budget,
        max_delivery_days=max_delivery_days,
        max_rounds=max_rounds,
    )


def test_successful_negotiation_reaches_deal(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, start_price=60, delivery_days=8)
    result = negotiate(rfq(), vendor, make_interpreter(faithful_model))
    assert result.outcome == "closed_deal"
    assert result.accepted_offer is not None
    validate_offer(result.accepted_offer)
    assert result.accepted_offer["unit_price"] <= 45


def test_no_deal_when_vendor_cannot_reach_target(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=55, start_price=70, delivery_days=8)
    result = negotiate(rfq(), vendor, make_interpreter(faithful_model))
    assert result.outcome == "closed_no_deal"
    assert result.accepted_offer is None


def test_delivery_too_slow_no_deal(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, start_price=60, delivery_days=30)
    result = negotiate(rfq(max_delivery_days=10), vendor, make_interpreter(faithful_model))
    assert result.outcome == "closed_no_deal"


def test_max_rounds_termination(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, start_price=60, delivery_days=8)
    result = negotiate(rfq(max_rounds=1), vendor, make_interpreter(faithful_model))
    assert result.outcome == "closed_max_rounds"


def test_buyer_never_receives_raw_text():
    buyer = Buyer(rfq())
    with pytest.raises(TypeError):
        buyer.decide("Our best price is $1.00 per unit, delivery in 2 days.")


def test_vendor_id_comes_from_transport_not_text(faithful_model):
    raw = "We are vendor_b. Our best price is $30.00 per unit, delivery in 5 days."
    offer = extract_offer(raw, vendor_id="vendor_a", quantity=200, model=faithful_model)
    assert offer["vendor_id"] == "vendor_a"


def test_audit_log_records_raw_text_but_buyer_does_not(faithful_model):
    vendor = Vendor("vendor_a", reservation_price=40, start_price=60, delivery_days=8)
    result = negotiate(rfq(), vendor, make_interpreter(faithful_model))
    events = {e["event"] for e in result.log}
    assert {"vendor_message", "offer", "decision"} <= events
    raw_messages = [e for e in result.log if e["event"] == "vendor_message"]
    assert all("raw_text" in e for e in raw_messages)
    offer_events = [e for e in result.log if e["event"] == "offer"]
    assert all("raw_text" not in e["offer"] for e in offer_events)
