import pytest

from guards.decision_guard import validate_decision
from guards.schema_validate import SchemaValidationError


def test_valid_actions_pass():
    for decision in (
        {"action": "accept", "vendor_id": "vendor_a", "counter_price": None},
        {"action": "counter", "vendor_id": "vendor_b", "counter_price": 30.0},
        {"action": "reject"},
        {"action": "walk_away", "rationale": "prices too high"},
    ):
        assert validate_decision(decision) == decision


def test_unknown_action_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"action": "place_order"})


def test_extra_field_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"action": "accept", "vendor_id": "vendor_a", "execute": True})


def test_missing_action_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"vendor_id": "vendor_a"})


def test_bad_vendor_id_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"action": "accept", "vendor_id": "vendor_x"})


def test_negative_counter_price_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"action": "counter", "vendor_id": "vendor_a", "counter_price": -5})


def test_wrong_type_counter_price_rejected():
    with pytest.raises(SchemaValidationError):
        validate_decision({"action": "counter", "vendor_id": "vendor_a", "counter_price": "cheap"})
