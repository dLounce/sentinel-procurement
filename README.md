# Sentinel Procurement

A privilege-separated multi-agent procurement negotiation system. A privileged
Buyer agent negotiates against several independent Vendor agents to source goods
under a fixed budget, while a quarantined Interpreter agent with no tool access
sits between untrusted vendor text and the only agent allowed to spend money.

## Security model

The design provides two distinct properties, and they are kept separate:

- **Architectural.** The Interpreter reads untrusted vendor text but has no
  capability or network path to act. The Buyer can act but never receives raw
  vendor text — it only ever sees validated `VendorOffer` objects. This
  separation is enforced by process, credential, and network isolation, not by
  prompts.
- **Empirical.** Deterministic guards (schema validation, budget, and price
  plausibility) reject malformed, ambiguous, out-of-policy, or anomalous offers
  before an order is placed. Their coverage is measured against defined
  adversarial cases, not proven complete.

Schema validation is necessary but not sufficient: a well-formed offer can still
carry a wrong value (the "$1 SUV"), so business guards run before any order.

## The `VendorOffer` contract

`schemas/vendor_offer.json` is the single object that crosses from the
Interpreter to the Buyer. It is strict (`additionalProperties: false`, every
field typed and enumerated) so free text, extra fields, and action-smuggling
fields cannot reach the Buyer's context. `guards/schema_validate.py` enforces it.

## Layout

```
schemas/   the VendorOffer contract
guards/    deterministic checks (not agents)
tests/     control-level tests
```

## Development

```
pip install -e ".[dev]"
pytest
```

## Status

Early development. Components are added incrementally; see the commit history for
what is currently implemented and tested.
