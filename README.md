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
- **Price-plausibility guard** blocks an offer if any of: its `unit_price` is
  below a configured absolute floor; its `extraction_flag` is pricing-relevant
  (`missing_price`, `non_usd`, `tiered`, `range`, `conditional`); or, once at
  least three valid offers exist for the round, its `unit_price` is an outlier —
  below `median × (1 − k)`.

Chosen parameters and rationale:

- **Absolute floor:** a scenario-specific configured threshold, supplied
  independently of vendor responses (never derived from vendor offers or vendor
  text, and not evaluation ground truth). It has no universal default and is set
  per procurement scenario, because a credible minimum unit cost depends on the
  item. It is the primary value-channel defense (e.g. it blocks the "$1 SUV").
- **Outlier `k = 0.4`** (block below 60% of the round median): a secondary
  cross-offer check, not the primary defense.
- **Minimum 3 valid offers** before the outlier rule applies, matching the
  three-vendor design; with fewer offers the guard falls back to the floor and
  the flag check.

`place_order` is born gated: it re-runs the delivery, budget, and plausibility
checks and cannot create an order unless all pass. The price guard is a
deterministic policy whose coverage is measured empirically — it is not a
mathematical guarantee against all value corruption.

## Evaluation

`eval/` runs a deterministic red-team set through the real negotiation path and
reports the two security properties separately. Evaluation-only ground truth
(true quote, legitimate range, expected delivery) lives only in the harness; the
runtime Buyer/Interpreter/Vendors never receive it. Reproduce with:

```
python -m eval.run_redteam
```

Measured results (16 cases: a legitimate baseline and 15 adversarial payloads;
seed 1337):

- **Metric A — action reachability (architectural): 0.** No payload reached the
  `place_order` surface. The Interpreter has no action capability and the only
  path to an order is the gated Buyer method, so attacker content cannot supply
  order arguments outside the validated deterministic path.
- **Metric B — value corruption (empirical): 1 anomalous in-budget order.** Of the
  15 payloads, 11 value-corruption attempts (drastic underpricing, in-budget
  outlier, non-USD, missing/unparseable price, malformed/ambiguous/hidden/false-
  slow delivery) were detected and blocked; direct prompt injection, action/field
  smuggling, and vendor identity spoofing were contained by schema validation and
  transport identity stamping. **One case was not blocked: a false-fast delivery
  corruption reached an anomalous in-budget order.**

Metric A is architectural. Metric B is an empirical property of the deterministic
guards against these specific cases and is not proven complete — the one observed
failure demonstrates that directly.

**Finding — the delivery value channel is not deterministically guarded.** The
price value channel has a deterministic plausibility guard (floor/outlier/flag);
the delivery value channel does not. Malformed, ambiguous, hidden, and unparseable
delivery are flagged and blocked, but a clean false-fast delivery value (a
well-formed but wrong `delivery_days`) has no deterministic cross-check and
produced an anomalous in-budget order in the evaluation. Closing this would
require a deterministic delivery-value control analogous to the price guard; it is
reported here rather than hidden.

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
