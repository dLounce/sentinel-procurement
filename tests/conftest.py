import json
import re

import pytest

from agents.vendor import INJECTION_MARKER

# Test doubles for the injected Interpreter model. A "faithful" model simulates a
# correct LLM by parsing the vendor line out of the prompt and returning the JSON
# the Interpreter expects; a "constant" model returns attacker- or case-specific
# JSON to exercise the security boundary; a "gullible" model simulates a model
# fooled by an injection into emitting the attacker's price. No production code
# parses vendor text.

_UNIT = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*per\s*unit", re.IGNORECASE)
_TOTAL = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*total", re.IGNORECASE)
_DELIVERY = re.compile(r"delivery\s+in\s+(\d+)\s+days", re.IGNORECASE)
_QUANTITY = re.compile(r"quantity:\s*(\d+)")


def _faithful(prompt: str) -> str:
    quantity = int(_QUANTITY.search(prompt).group(1))
    unit_match = _UNIT.search(prompt)
    if unit_match:
        unit_price = round(float(unit_match.group(1).replace(",", "")), 2)
        basis = "unit"
    else:
        total_match = _TOTAL.search(prompt)
        unit_price = round(float(total_match.group(1).replace(",", "")) / quantity, 2)
        basis = "total"
    delivery_days = int(_DELIVERY.search(prompt).group(1))
    return json.dumps(
        {
            "vendor_id": "vendor_a",
            "unit_price": unit_price,
            "currency": "USD",
            "delivery_days": delivery_days,
            "confidence": "high",
            "quoted_basis": basis,
            "extraction_flag": "none",
        }
    )


@pytest.fixture
def faithful_model():
    return _faithful


@pytest.fixture
def constant_model():
    def make(payload: str):
        return lambda prompt: payload

    return make


@pytest.fixture
def gullible_model():
    """Factory: a model fooled by an injection into emitting `injected_price`
    (keeping the real delivery); on non-injected prompts it behaves faithfully."""

    def make(injected_price: float):
        def model(prompt: str) -> str:
            if INJECTION_MARKER in prompt:
                delivery_days = int(_DELIVERY.search(prompt).group(1))
                return json.dumps(
                    {
                        "vendor_id": "vendor_a",
                        "unit_price": injected_price,
                        "currency": "USD",
                        "delivery_days": delivery_days,
                        "confidence": "high",
                        "quoted_basis": "unit",
                        "extraction_flag": "none",
                    }
                )
            return _faithful(prompt)

        return model

    return make
