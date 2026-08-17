import json

import pytest

from agents.vendor import Vendor, VendorProposalError
from rfq import RFQ


def rfq():
    return RFQ("rfq-1", "widgets", 200, 9000, 10, 8)


def proposal(unit_price, delivery_days=8, note="ok"):
    return lambda prompt: json.dumps({"unit_price": unit_price, "delivery_days": delivery_days, "note": note})


def test_vendor_model_output_drives_the_quote():
    vendor = Vendor("vendor_a", reservation_price=30, opening_price=60, delivery_days=8)
    high = vendor.propose_quote(rfq(), 0, None, [], model=proposal(55))
    low = vendor.propose_quote(rfq(), 0, None, [], model=proposal(40))
    assert "$55.00 per unit" in high
    assert "$40.00 per unit" in low


def test_reservation_clamp_is_deterministic():
    # the model proposes below the private floor; the vendor must not issue it
    vendor = Vendor("vendor_a", reservation_price=27, opening_price=60, delivery_days=8)
    message = vendor.propose_quote(rfq(), 3, 20.0, [], model=proposal(25))
    assert "$27.00 per unit" in message
    assert "$25.00" not in message


def test_reservation_price_never_appears_in_message():
    vendor = Vendor("vendor_a", reservation_price=27, opening_price=60, delivery_days=8)
    message = vendor.propose_quote(rfq(), 0, None, [], model=proposal(45))
    assert "27" not in message.replace("$45.00", "").replace("8 days", "")


def test_different_vendors_produce_different_model_driven_quotes():
    def by_vendor(prompt):
        price = 50.0 if "vendor_a" in prompt else 41.0
        return json.dumps({"unit_price": price, "delivery_days": 8, "note": "ok"})

    va = Vendor("vendor_a", reservation_price=30, opening_price=60, delivery_days=8)
    vb = Vendor("vendor_b", reservation_price=30, opening_price=60, delivery_days=8)
    assert "$50.00 per unit" in va.propose_quote(rfq(), 0, None, [], model=by_vendor)
    assert "$41.00 per unit" in vb.propose_quote(rfq(), 0, None, [], model=by_vendor)


def test_malformed_vendor_proposal_raises():
    vendor = Vendor("vendor_a", reservation_price=30, opening_price=60, delivery_days=8)
    with pytest.raises(VendorProposalError):
        vendor.propose_quote(rfq(), 0, None, [], model=lambda prompt: "no json here")


def test_vendor_prompt_carries_reservation_to_its_own_brain_only():
    # the vendor's own model legitimately reasons about its reservation, but it is
    # never emitted to anyone else.
    vendor = Vendor("vendor_a", reservation_price=27, opening_price=60, delivery_days=8)
    prompt = vendor.build_prompt(rfq(), 0, None, [])
    assert "private_reservation_price: 27" in prompt
