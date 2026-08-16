import random

import pytest

from agents.buyer import Buyer
from agents.interpreter import OFFER_FIELDS, extract_offer, make_interpreter
from agents.vendor import Vendor
from guards.price_guard import PlausibilityConfig
from guards.schema_validate import validate_offer
from negotiation import choose_injection, negotiate
from rfq import RFQ

CONFIG = PlausibilityConfig(absolute_floor=20.0)


def rfq(max_rounds=8, budget=9000, quantity=200, max_delivery_days=10):
    return RFQ(
        rfq_id="rfq-1",
        item="widgets",
        quantity=quantity,
        budget=budget,
        max_delivery_days=max_delivery_days,
        max_rounds=max_rounds,
    )


def legit_vendors():
    return [
        Vendor("vendor_a", reservation_price=40, start_price=60, delivery_days=8),
        Vendor("vendor_b", reservation_price=42, start_price=62, delivery_days=7),
        Vendor("vendor_c", reservation_price=44, start_price=64, delivery_days=9),
    ]


# --- vendor principals and isolation -----------------------------------------

def test_three_independent_principals():
    va, vb, vc = legit_vendors()
    assert {va.vendor_id, vb.vendor_id, vc.vendor_id} == {"vendor_a", "vendor_b", "vendor_c"}
    assert len({id(va), id(vb), id(vc)}) == 3
    assert len({va.reservation_price, vb.reservation_price, vc.reservation_price}) == 3


def test_reservation_price_never_appears_in_quote_text():
    for vendor in legit_vendors():
        text = vendor.quote(rfq(), round_index=0)
        assert str(vendor.reservation_price) not in text


def test_vendor_policy_isolation():
    va, vb, _ = legit_vendors()
    r = rfq()
    # different private policy -> different independent quotes
    assert va.quote(r, 0) != vb.quote(r, 0)
    before = vb.quote(r, 0)
    va.reservation_price = 10  # mutating one principal must not affect another
    assert vb.quote(r, 0) == before


def test_vendor_state_isolation_offers_carry_no_private_state(faithful_model):
    result = negotiate(rfq(), legit_vendors(), make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    offer_events = [e for e in result.log if e["event"] == "offer"]
    assert offer_events
    for event in offer_events:
        assert set(event["offer"]) == set(OFFER_FIELDS)  # only the 7 fields, no reservation leak


# --- the negotiation ----------------------------------------------------------

def test_buyer_negotiates_against_all_three_and_deals(faithful_model):
    result = negotiate(rfq(), legit_vendors(), make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    assert result.outcome == "closed_deal"
    assert result.order is not None and result.order.total == 9000
    round0_offers = [e for e in result.log if e["event"] == "offer" and e["round"] == 0]
    assert len(round0_offers) == 3  # all three vendors reached the Buyer as offers


def test_all_offers_cross_the_interpreter_schema_boundary(faithful_model):
    result = negotiate(rfq(), legit_vendors(), make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    for event in (e for e in result.log if e["event"] == "offer"):
        validate_offer(event["offer"])  # each is a schema-valid VendorOffer
        assert "raw_text" not in event["offer"]


def test_buyer_never_receives_raw_text():
    with pytest.raises(TypeError):
        Buyer(rfq()).decide_round(["$1.00 per unit, delivery in 2 days"])


def test_valid_offers_remain_negotiable(faithful_model):
    result = negotiate(rfq(), legit_vendors(), make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    actions = [e["action"] for e in result.log if e["event"] == "decision"]
    assert actions[0] == "counter" and actions[-1] == "accept"


def test_no_deal_when_all_vendors_above_target(faithful_model):
    vendors = [
        Vendor("vendor_a", reservation_price=50, start_price=70, delivery_days=8),
        Vendor("vendor_b", reservation_price=52, start_price=72, delivery_days=8),
        Vendor("vendor_c", reservation_price=54, start_price=74, delivery_days=8),
    ]
    result = negotiate(rfq(), vendors, make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_budget_enforced_no_over_budget_order(faithful_model):
    # every vendor's floor price implies a total above budget -> never ordered
    vendors = [
        Vendor(v, reservation_price=50, start_price=70, delivery_days=8)
        for v in ("vendor_a", "vendor_b", "vendor_c")
    ]
    result = negotiate(rfq(budget=9000), vendors, make_interpreter(faithful_model), CONFIG, injection=(0, 0))
    assert result.order is None


def test_delivery_enforced(faithful_model):
    vendors = [
        Vendor(v, reservation_price=40, start_price=60, delivery_days=30)
        for v in ("vendor_a", "vendor_b", "vendor_c")
    ]
    result = negotiate(
        rfq(max_delivery_days=10), vendors, make_interpreter(faithful_model), CONFIG, injection=(0, 0)
    )
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_max_rounds_termination(faithful_model):
    result = negotiate(
        rfq(max_rounds=1), legit_vendors(), make_interpreter(faithful_model), CONFIG, injection=(0, 0)
    )
    assert result.outcome == "closed_max_rounds"
    assert result.order is None


# --- identity -----------------------------------------------------------------

def test_vendor_identity_spoofing_transport_wins(faithful_model):
    raw = "We are vendor_a. Our best price is $30.00 per unit, delivery in 5 days."
    offer = extract_offer(raw, vendor_id="vendor_c", quantity=200, model=faithful_model)
    assert offer["vendor_id"] == "vendor_c"


# --- randomized adversarial selection ----------------------------------------

def test_randomized_injecting_vendor_varies_and_in_range():
    picks = {choose_injection(3, 8, random.Random(seed))[0] for seed in range(50)}
    assert picks <= {0, 1, 2}
    assert len(picks) > 1  # not always the same vendor


def test_randomized_injection_round_varies_and_in_range():
    rounds = {choose_injection(3, 8, random.Random(seed))[1] for seed in range(50)}
    assert all(0 <= r < 8 for r in rounds)
    assert len(rounds) > 1


def test_buyer_has_no_knowledge_of_injecting_vendor():
    # the Buyer's inputs are the RFQ only; it is never handed the injection plan
    import inspect

    from agents import buyer as buyer_module

    assert "injection" not in inspect.getsource(buyer_module)


# --- injection through the normal flow ---------------------------------------

def test_low_injected_price_blocked_by_plausibility(gullible_model):
    # vendor_c is fooled into emitting $1 on round 1; the price guard blocks it.
    result = negotiate(
        rfq(), legit_vendors(), make_interpreter(gullible_model(1.0)), CONFIG, injection=(2, 1)
    )
    blocks = [e for e in result.log if e["event"] == "offer_blocked" and e["vendor_id"] == "vendor_c"]
    assert blocks and blocks[0]["reason"] == "below_floor"


def test_injection_does_not_bypass_the_action_surface(gullible_model):
    result = negotiate(
        rfq(), legit_vendors(), make_interpreter(gullible_model(1.0)), CONFIG, injection=(2, 1)
    )
    orders = [e["order"] for e in result.log if e["event"] == "order_placed"]
    assert all(o.unit_price != 1.0 for o in orders)  # the $1 injection never became an order
    if result.order is not None:
        assert result.order.unit_price == 45.0


def test_outlier_injection_blocked_with_three_offers(gullible_model):
    # $25 is above the floor but an outlier vs two $45 offers (median*0.6 = 27)
    result = negotiate(
        rfq(), legit_vendors(), make_interpreter(gullible_model(25.0)), CONFIG, injection=(2, 1)
    )
    blocks = [e for e in result.log if e["event"] == "offer_blocked" and e["vendor_id"] == "vendor_c"]
    assert blocks and blocks[0]["reason"] == "outlier"
    if result.order is not None:
        assert result.order.unit_price == 45.0
