import json
from pathlib import Path

from jsonschema import Draft7Validator

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "vendor_offer.json"
_VALIDATOR = Draft7Validator(json.loads(_SCHEMA_PATH.read_text()))


class SchemaValidationError(ValueError):
    """Raised when an object does not conform to the VendorOffer contract."""


def validate_offer(offer):
    """Validate a candidate VendorOffer against the schema.

    The Buyer only ever receives objects that clear this check, so validation
    is a security boundary: it rejects free-text, extra fields, and
    action-smuggling fields before any offer reaches privileged logic. It does
    not judge whether the values are economically sensible — the business
    guards do that.
    """
    errors = sorted(_VALIDATOR.iter_errors(offer), key=lambda e: list(e.path))
    if errors:
        raise SchemaValidationError("; ".join(e.message for e in errors))
    return offer
