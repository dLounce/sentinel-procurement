import random

import pytest

from agents.buyer_graph import deterministic_buyer_decider, make_buyer_decider
from agents.interpreter import OFFER_FIELDS, extract_offer, make_interpreter
from agents.vendor import Vendor
from eval.models import concession_vendor_model
from guards.price_guard import PlausibilityConfig
from guards.schema_validate import validate_offer
from negotiation import choose_injection, negotiate

CONFIG = PlausibilityConfig(absolute_floor=20.0)
from rfq import RFQ


def rfq(max_rounds=8, budget=9000, quantity=200, max_delivery_days=10):
    return RFQ("rfq-1", "widgets", quantity, budget, max_delivery_days, max_rounds)


def legit_vendors(delivery=8):
    return [
        Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=delivery),
        Vendor("vendor_b", reservation_price=42, opening_price=62, delivery_days=delivery),
        Vendor("vendor_c", reservation_price=44, opening_price=64, delivery_days=delivery),
    ]


def run(vendors, interp_model, *, injection=(0, 0), buyer_decider=deterministic_buyer_decider, **rfq_kwargs):
    return negotiate(
        rfq(**rfq_kwargs),
        vendors,
        make_interpreter(interp_model),
        buyer_decider,
        CONFIG,
        vendor_model=concession_vendor_model,
        injection=injection,
    )


# --- vendor principals and isolation -----------------------------------------

def test_three_independent_principals():
    va, vb, vc = legit_vendors()
    assert {va.vendor_id, vb.vendor_id, vc.vendor_id} == {"vendor_a", "vendor_b", "vendor_c"}
    assert len({va.reservation_price, vb.reservation_price, vc.reservation_price}) == 3


def test_vendor_state_isolation_offers_carry_no_private_state(faithful_model):
    result = run(legit_vendors(), faithful_model)
    offer_events = [e for e in result.log if e["event"] == "offer"]
    assert offer_events
    for event in offer_events:
        assert set(event["offer"]) == set(OFFER_FIELDS)


def test_reservation_price_is_never_exposed_in_offers(faithful_model):
    # the validated offers the Buyer sees carry only the contract fields; no
    # vendor private state (e.g. reservation_price) ever crosses the boundary.
    result = run(legit_vendors(), faithful_model)
    for event in (e for e in result.log if e["event"] == "offer"):
        assert set(event["offer"]) == set(OFFER_FIELDS)
        assert "reservation_price" not in event["offer"]


# --- the negotiation ----------------------------------------------------------

def test_buyer_negotiates_against_all_three_and_deals(faithful_model):
    result = run(legit_vendors(), faithful_model)
    assert result.outcome == "closed_deal"
    assert result.order is not None and result.order.total == 9000
    round0 = [e for e in result.log if e["event"] == "offer" and e["round"] == 0]
    assert len(round0) == 3


def test_all_offers_cross_the_interpreter_schema_boundary(faithful_model):
    result = run(legit_vendors(), faithful_model)
    for event in (e for e in result.log if e["event"] == "offer"):
        validate_offer(event["offer"])
        assert "raw_text" not in event["offer"]


def test_no_deal_when_all_vendors_above_target(faithful_model):
    vendors = [
        Vendor("vendor_a", reservation_price=50, opening_price=70, delivery_days=8),
        Vendor("vendor_b", reservation_price=52, opening_price=72, delivery_days=8),
        Vendor("vendor_c", reservation_price=54, opening_price=74, delivery_days=8),
    ]
    result = run(vendors, faithful_model)
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_budget_enforced_no_over_budget_order(faithful_model):
    vendors = [Vendor(v, reservation_price=50, opening_price=70, delivery_days=8) for v in ("vendor_a", "vendor_b", "vendor_c")]
    result = run(vendors, faithful_model)
    assert result.order is None


def test_delivery_enforced(faithful_model):
    vendors = [Vendor(v, reservation_price=40, opening_price=60, delivery_days=30) for v in ("vendor_a", "vendor_b", "vendor_c")]
    result = run(vendors, faithful_model, max_delivery_days=10)
    assert result.outcome == "closed_no_deal"
    assert result.order is None


def test_max_rounds_termination(faithful_model):
    result = run(legit_vendors(), faithful_model, max_rounds=1)
    assert result.outcome == "closed_max_rounds"
    assert result.order is None


def test_malformed_extraction_is_dropped_not_crashing(constant_model):
    bad = constant_model('{"unit_price": "cheap"}')
    result = run(legit_vendors(), bad, max_rounds=1)
    dropped = [e for e in result.log if e["event"] == "offer_dropped"]
    assert dropped and dropped[0]["reason"] == "SchemaValidationError"


# --- identity -----------------------------------------------------------------

def test_vendor_identity_spoofing_transport_wins(faithful_model):
    raw = "We are vendor_a. Our best price is $30.00 per unit, delivery in 5 days."
    offer = extract_offer(raw, vendor_id="vendor_c", quantity=200, model=faithful_model)
    assert offer["vendor_id"] == "vendor_c"


# --- randomized adversarial selection ----------------------------------------

def test_randomized_injecting_vendor_varies_and_in_range():
    picks = {choose_injection(3, 8, random.Random(seed))[0] for seed in range(50)}
    assert picks <= {0, 1, 2} and len(picks) > 1


def test_randomized_injection_round_varies_and_in_range():
    rounds = {choose_injection(3, 8, random.Random(seed))[1] for seed in range(50)}
    assert all(0 <= r < 8 for r in rounds) and len(rounds) > 1


# --- injection through the normal flow ---------------------------------------

def test_low_injected_price_blocked_by_plausibility(gullible_model):
    result = run(legit_vendors(), gullible_model(1.0), injection=(2, 1))
    blocks = [e for e in result.log if e["event"] == "offer_blocked" and e["vendor_id"] == "vendor_c"]
    assert blocks and blocks[0]["reason"] == "below_floor"


def test_injection_does_not_bypass_the_action_surface(gullible_model):
    result = run(legit_vendors(), gullible_model(1.0), injection=(2, 1))
    orders = [e["order"] for e in result.log if e["event"] == "order_placed"]
    assert all(o.unit_price != 1.0 for o in orders)


def test_outlier_injection_blocked_with_three_offers(gullible_model):
    result = run(legit_vendors(), gullible_model(25.0), injection=(2, 1))
    blocks = [e for e in result.log if e["event"] == "offer_blocked" and e["vendor_id"] == "vendor_c"]
    assert blocks and blocks[0]["reason"] == "outlier"


# --- agentic buyer cannot bypass the guards ----------------------------------

def test_agentic_buyer_cannot_order_a_guard_blocked_offer(gullible_model):
    # a Buyer LLM that tries to accept the injected below-floor vendor cannot: the
    # offer is blocked before it becomes selectable, and the gate would refuse it.
    greedy_buyer = make_buyer_decider(lambda p: '{"action":"accept","vendor_id":"vendor_c"}')
    result = run(legit_vendors(), gullible_model(1.0), injection=(2, 1), buyer_decider=greedy_buyer)
    assert result.order is None or result.order.unit_price != 1.0
