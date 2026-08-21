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
- **Deterministic authorization.** Every privileged order must pass configured
  procurement policy — schema/extraction-flag validation, the budget ceiling, the
  configured price floor, and the max-delivery constraint — before it is placed, so
  vendor communication cannot drive an out-of-policy or unauthorized action.

Whether a vendor's quoted price or delivery is *truthful* is a commercial-claim
question, not a security property: a vendor may quote poorly or optimistically and
is still handled as legitimate commerce. Schema validation is necessary but not
sufficient on its own: a well-formed offer can still be out of policy (a below-floor
price, the "$1 SUV"), so the deterministic policy guards run before any order.

## Agentic negotiation — models propose, deterministic controls authorize

The Buyer and Vendors are genuinely model-driven. The Buyer runs a small LangGraph
reasoning loop that reasons over the round's validated offers and history and emits
a structured `BuyerDecision` (`accept` / `counter` / `reject` / `walk_away`) — it
never calls `place_order`. The deterministic authorization layer (BuyerDecision
schema validation + the guards + the gated `place_order`) independently decides
what is *allowed*, so a confused or malicious Buyer model cannot bypass a control.
Each Vendor's model proposes a price; a deterministic clamp enforces that vendor's
private reservation floor, which never leaves the vendor. Models are injected
(strong model for the Buyer, cheaper model for Vendors); tests use scripted
doubles, so no live calls are made and behaviour is reproducible.

## The `VendorOffer` contract

`schemas/vendor_offer.json` is the single object that crosses from the
Interpreter to the Buyer. It is strict (`additionalProperties: false`, every
field typed and enumerated) so free text, extra fields, and action-smuggling
fields cannot reach the Buyer's context. `guards/schema_validate.py` enforces it.

## Deterministic guards

Before an order is placed, two deterministic guards must pass — neither is an LLM
judgment, and neither reads the offer's `confidence`:

- **Budget guard** rejects any order whose `unit_price × quantity` exceeds the
  RFQ budget. The exact budget is allowed.
- **Price-policy guard** blocks an offer if either: its `unit_price` is below a
  configured absolute floor, or its `extraction_flag` is pricing-relevant
  (`missing_price`, `non_usd`, `tiered`, `range`, `conditional`). It reads only the
  offer's own price and flag against configured policy — it does not compare a
  vendor's claim against other vendors or against hidden truth.

Chosen parameters and rationale:

- **Absolute floor:** a scenario-specific configured threshold, supplied
  independently of vendor responses (never derived from vendor offers or vendor
  text, and not evaluation ground truth). It has no universal default and is set
  per procurement scenario, because a credible minimum unit cost depends on the
  item. It is a configured policy limit (e.g. it blocks an out-of-policy "$1 SUV").

`place_order` is born gated: it re-runs the delivery constraint, budget, and
price-policy checks and cannot create an order unless all pass. These are
deterministic policy controls — they authorize an order against configured policy,
they do not judge whether a vendor's commercial claim is truthful.

## Evaluation

`eval/` runs a deterministic red-team set through the real negotiation path and
reports the security metric: action reachability. Evaluation-only ground truth
(whether a case is an attack and which channel it targets) lives only in the
harness; the runtime Buyer/Interpreter/Vendors never receive it. Reproduce with:

```
python -m eval.run_redteam
```

Measured result (14 cases: a legitimate baseline and 13 adversarial payloads;
seed 1337):

- **Metric A — action reachability: 0.** No payload reached the `place_order`
  surface outside the deterministic authorization path. The Interpreter has no
  action capability and the only path to an order is the gated Buyer method, so
  attacker content cannot supply order arguments outside the validated path. Each
  injection case (action/schema, price, delivery, identity) is contained at the
  deterministic gate — blocked by configured policy, dropped by schema validation,
  or bound to the transport-stamped `vendor_id`.

Vendor-claim accuracy is not measured as security: whether an in-policy quoted
price or delivery is truthful is a commercial question, not a system compromise.
An above-floor price or a within-constraint delivery that satisfies configured
policy may legitimately reach an authorized order.

## Layout

```
agents/    Buyer, Interpreter, Vendor
schemas/   the VendorOffer contract
guards/    deterministic checks (not agents)
eval/      evaluation-only red-team harness (ground truth stays here)
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
