import json

from agents.buyer_graph import deterministic_buyer_decider, make_buyer_decider
from guards.decision_guard import validate_decision

RFQ_VIEW = {"item": "widgets", "quantity": 200, "budget": 9000, "max_delivery_days": 10}


def offer(vendor_id, unit_price, delivery_days=8, flag="none"):
    return {
        "vendor_id": vendor_id,
        "unit_price": unit_price,
        "currency": "USD",
        "delivery_days": delivery_days,
        "confidence": "high",
        "quoted_basis": "unit",
        "extraction_flag": flag,
    }


def constant(payload):
    return lambda prompt: payload


OFFERS = [offer("vendor_a", 45.0), offer("vendor_b", 44.0)]


def test_decision_depends_on_model_output():
    walk = make_buyer_decider(constant('{"action":"walk_away"}'))
    rej = make_buyer_decider(constant('{"action":"reject"}'))
    assert walk(RFQ_VIEW, OFFERS, [], 0, 8)["action"] == "walk_away"
    assert rej(RFQ_VIEW, OFFERS, [], 0, 8)["action"] == "reject"


def test_buyer_can_accept_a_specific_vendor():
    decide = make_buyer_decider(constant('{"action":"accept","vendor_id":"vendor_b"}'))
    decision = decide(RFQ_VIEW, OFFERS, [], 0, 8)
    assert decision["action"] == "accept" and decision["vendor_id"] == "vendor_b"


def test_buyer_can_counter_with_a_price():
    decide = make_buyer_decider(constant('{"action":"counter","vendor_id":"vendor_a","counter_price":30.0}'))
    decision = decide(RFQ_VIEW, OFFERS, [], 0, 8)
    assert decision["action"] == "counter" and decision["counter_price"] == 30.0


def test_malformed_model_output_is_repaired_to_a_valid_decision():
    decide = make_buyer_decider(constant("I refuse to output JSON"))
    decision = decide(RFQ_VIEW, OFFERS, [], 0, 8)
    validate_decision(decision)
    assert decision["action"] in ("accept", "counter", "reject", "walk_away")


def test_accepting_a_nonexistent_vendor_is_repaired_not_honored():
    decide = make_buyer_decider(constant('{"action":"accept","vendor_id":"vendor_c"}'))
    decision = decide(RFQ_VIEW, OFFERS, [], 0, 8)
    # vendor_c is not among the round's offers, so the accept cannot be honored
    assert not (decision["action"] == "accept" and decision["vendor_id"] == "vendor_c")


def test_negotiation_history_is_passed_to_the_buyer():
    captured = {}

    def capture(prompt):
        captured["prompt"] = prompt
        return '{"action":"reject"}'

    decide = make_buyer_decider(capture)
    history = [{"round": 0, "offers": OFFERS, "decision": {"action": "counter", "counter_price": 40.0}}]
    decide(RFQ_VIEW, OFFERS, history, 1, 8)
    assert "Negotiation history" in captured["prompt"]
    assert "round 1:" in captured["prompt"]  # the prior round's content, not just a count
    assert "you counter at $40.00" in captured["prompt"]


def test_buyer_prompt_contains_no_reservation_or_raw_text():
    captured = {}

    def capture(prompt):
        captured["prompt"] = prompt
        return '{"action":"reject"}'

    make_buyer_decider(capture)(RFQ_VIEW, OFFERS, [], 0, 8)
    lowered = captured["prompt"].lower()
    assert "reservation" not in lowered
    assert "system override" not in lowered  # no raw vendor injection text
    assert "per unit, delivery in" not in lowered  # no raw vendor sentence


def test_deterministic_decider_is_schema_valid():
    validate_decision(deterministic_buyer_decider(RFQ_VIEW, OFFERS, [], 0, 8))
