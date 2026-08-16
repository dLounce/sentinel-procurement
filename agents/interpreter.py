"""Interpreter boundary: raw vendor text -> validated VendorOffer.

The security boundary is this interface, not the extractor body. The Buyer only
ever receives the schema-validated VendorOffer returned here, never raw text, and
vendor_id is supplied by trusted transport rather than read from the message.

M2 ships a deterministic stand-in extractor so the negotiation flow can run and be
tested without live model calls. In M3 the extractor body is replaced by the
quarantined LLM Interpreter (no tools, no credentials, no network path to any
tool); the signature and the schema-validated return contract stay fixed.
"""

import re

from guards.schema_validate import validate_offer


class ExtractionError(ValueError):
    """Raised when the M2 stand-in extractor cannot parse a vendor message."""


_UNIT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*per\s*unit", re.IGNORECASE)
_TOTAL_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*total", re.IGNORECASE)
_DELIVERY_RE = re.compile(r"delivery\s+in\s+(\d+)\s+days", re.IGNORECASE)


def extract_offer(raw_text: str, *, vendor_id: str, quantity: int) -> dict:
    unit_price, quoted_basis = _extract_price(raw_text, quantity)
    delivery_days = _extract_delivery(raw_text)
    offer = {
        "vendor_id": vendor_id,
        "unit_price": unit_price,
        "currency": "USD",
        "delivery_days": delivery_days,
        "confidence": "high",
        "quoted_basis": quoted_basis,
        "extraction_flag": "none",
    }
    return validate_offer(offer)


def _to_number(text: str) -> float:
    return float(text.replace(",", ""))


def _extract_price(raw_text: str, quantity: int) -> tuple[float, str]:
    match = _UNIT_RE.search(raw_text)
    if match:
        return round(_to_number(match.group(1)), 2), "unit"
    match = _TOTAL_RE.search(raw_text)
    if match:
        return round(_to_number(match.group(1)) / quantity, 2), "total"
    raise ExtractionError("no parseable price in vendor message")


def _extract_delivery(raw_text: str) -> int:
    match = _DELIVERY_RE.search(raw_text)
    if not match:
        raise ExtractionError("no parseable delivery in vendor message")
    return int(match.group(1))
