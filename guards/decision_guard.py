import json
from pathlib import Path

from jsonschema import Draft7Validator

from guards.schema_validate import SchemaValidationError

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "buyer_decision.json"
_VALIDATOR = Draft7Validator(json.loads(_SCHEMA_PATH.read_text()))


def validate_decision(decision):
    """Validate a candidate BuyerDecision against the schema.

    The Buyer LLM's output is untrusted: this rejects free text, extra fields,
    and unknown actions before the deterministic authorization layer acts on it.
    Structural only — cross-field semantics (e.g. an accept must reference a real
    offer) and every business rule are enforced deterministically downstream.
    """
    errors = sorted(_VALIDATOR.iter_errors(decision), key=lambda e: list(e.path))
    if errors:
        raise SchemaValidationError("; ".join(e.message for e in errors))
    return decision
