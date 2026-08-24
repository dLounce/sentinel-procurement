# Sentinel Procurement

A privilege-separated multi-agent procurement system, with an adversarial security evaluation that tests whether untrusted vendor text can ever push a real, model-driven Buyer into an unauthorized or out-of-policy purchase.

## Results at a glance

Live model runs, 100 completed attack replications across 10 attack families, on the defended architecture:

- 0 of 100 trials produced an unauthorized action.
- 0 of 100 trials produced a policy violation.
- 12 of 100 trials showed a measurable model-influence effect (the attack shifted the Buyer's reasoning-level decision past the benign noise band), and none of those turned into an unauthorized action or a policy violation.

Deterministic 25-cell representative matrix, reproducible and offline (no API calls):

- Unauthorized-action rate: 3 of 25.
- Policy-violation rate: 0 of 25.
- Model-susceptibility and attacker-self-selection: 0 of 19 evaluated cells, with 6 cells explicitly marked as not causally evaluable.

All 3 unauthorized-action cases came from the deliberately weakened S0, S1, and S2 architecture variants. The defended architecture (S3) produced no unauthorized actions in either the matrix or the live runs.

These numbers are observations from completed trials. They are not a claim that the true attack success probability is zero. Deterministic coverage and live-model validation are reported separately on purpose: the matrix gives repeatable, wide coverage with scripted doubles, and the live runs check behavior against real models on a focused subset.

## What was built

- An adversarial evaluation for LLM procurement security that runs attacks through the real negotiation path and reports whether any attack reaches an unauthorized or out-of-policy order.
- A matched control/treatment method with an A/A-prime baseline. Control runs the negotiation once and records the vendor messages. Treatment replays those exact commercial messages and adds only the attacker's payload, so the only thing that changes is the attack. A/A-prime replays the control messages a second time to measure ordinary run-to-run variation, which sets a noise band that a real attack effect has to beat.
- A 25-cell deterministic representative matrix that covers the attack families, three attack timings (early, middle, late/finalization), best/middle/worst attacker positions, one/two/three malicious-vendor collusion, and the S0 through S3 architecture variants. It runs with scripted model doubles, so it is fully reproducible and makes no API calls.
- 100 completed live attack replications across 10 attack families against the defended architecture, using real models (Buyer on a stronger model, Interpreter on a smaller model, Vendors on a separate hosted model).

## Security model

The design keeps two separate properties, and it does not mix them:

- Architectural. The Interpreter reads untrusted vendor text but has no tool or network path to act. The Buyer can act but never sees raw vendor text; it only ever receives validated VendorOffer objects. This split is enforced by process, credential, and network isolation, not by prompt wording.
- Deterministic authorization. Every privileged order must pass configured procurement policy (schema and extraction-flag validation, the budget ceiling, the configured price floor, and the max-delivery constraint) before it is placed. Vendor text cannot drive an out-of-policy or unauthorized order.

Whether a vendor's quoted price or delivery is truthful is a commercial-claim question, not a security property. A vendor can quote poorly or optimistically and is still treated as ordinary commerce. Schema validation is necessary but not enough on its own: a well-formed offer can still be out of policy (a below-floor price, the "$1 SUV"), so the deterministic policy guards run before any order.

## Agentic negotiation, models propose and deterministic controls authorize

The Buyer and Vendors are genuinely model-driven. The Buyer runs a small LangGraph reasoning loop over the round's validated offers and history and emits a structured BuyerDecision (accept, counter, reject, or walk_away). It never calls place_order. The deterministic authorization layer (BuyerDecision schema validation, the guards, and the gated place_order) independently decides what is allowed, so a confused or malicious Buyer model cannot get past a control. Each Vendor's model proposes a price, and a deterministic clamp enforces that vendor's private reservation floor, which never leaves the vendor. Models are injected, and tests use scripted doubles so no live calls are made and behavior is reproducible.

## The VendorOffer contract

schemas/vendor_offer.json is the single object that crosses from the Interpreter to the Buyer. It is strict (additionalProperties: false, every field typed and enumerated) so free text, extra fields, and action-smuggling fields cannot reach the Buyer's context. guards/schema_validate.py enforces it.

## Deterministic guards

Before an order is placed, two deterministic guards must pass. Neither is an LLM judgment, and neither reads the offer's confidence:

- Budget guard rejects any order whose unit_price times quantity exceeds the RFQ budget. The exact budget is allowed.
- Price-policy guard blocks an offer if its unit_price is below a configured absolute floor, or if its extraction_flag is pricing-relevant (missing_price, non_usd, tiered, range, conditional). It reads only the offer's own price and flag against configured policy. It does not compare a vendor's claim against other vendors or against hidden truth.

The absolute floor is a scenario-specific configured threshold, supplied independently of vendor responses and never derived from vendor text or evaluation ground truth. It is set per scenario because a credible minimum unit cost depends on the item, and it blocks an out-of-policy case like a "$1 SUV".

place_order is born gated: it re-runs the delivery, budget, and price-policy checks and cannot create an order unless all pass. These are deterministic policy controls. They authorize an order against configured policy; they do not judge whether a vendor's commercial claim is truthful.

## Evaluation method

Each attack cell is a matched pair on the real negotiation path:

- Control runs the negotiation with real (or scripted) models and records the exact vendor commercial messages.
- Treatment replays those same commercial messages and adds only the attacker's scripted payload at the configured round, so the commercial content is held fixed and the attack is the only change.
- A/A-prime replays the control messages again to measure benign variation, giving a noise band. A model-influence verdict is only recorded when the treatment decision moves past that band.

Evaluation-only ground truth (whether a case is an attack and which channel it targets) lives only in the harness. The runtime Buyer, Interpreter, and Vendors never receive it.

## Important security finding

The three unauthorized-action results in the deterministic matrix are exactly the S0, S1, and S2 cells, which are intentionally weakened baselines:

- S0 sends raw vendor text straight to the Buyer with no trusted offer boundary and no authorization gate.
- S1 adds only a prompt-level warning, still with no trusted boundary.
- S2 restores the typed offer boundary but still bypasses the deterministic place_order gate.

These variants are supposed to allow unauthorized orders. They are the control group that shows the defense is doing real work. The defended architecture (S3), which keeps both the typed Interpreter-to-Buyer boundary and the gated place_order, produced no unauthorized actions and no policy violations in the matrix or across the 100 live trials.

The 12 live model-influence cases matter for the same reason. Even when an attack changed the Buyer model's reasoning, the deterministic controls downstream still held, so influence at the model layer did not become an unauthorized action or a policy breach.

## Engineering and reliability

- Hard 5-round ceiling. Evaluator negotiations are capped at 5 rounds so a single trial cannot run unbounded, without changing production negotiation semantics.
- Explicit attack-not-executed versus attack-influenced distinction. If a negotiation ends before the attack round, the payload is never delivered. The evaluator marks that trial as attack-not-reached and never scores it as influence or compromise, so a benign early close is not mistaken for an attack effect.
- Causal metric denominator fix. Model-susceptibility and attacker-self-selection rates are computed only over trials where causal influence was actually evaluated. Cells that do not run the A/A-prime comparison (collusion, the architecture ablations, and the bounded adaptive cell) are excluded from those denominators and listed explicitly, so the rate is not diluted by cells that could never register a positive. This is why the deterministic causal metric reads 0 of 19 rather than 0 of 25.
- 310 passing tests covering the guards, the negotiation path, the evaluation runners, the outcome and evidence layers, and the regression cases behind the fixes above.

## Reproduce

Install and run the test suite:

```
pip install -e ".[dev]"
pytest
```

Run the offline pilot (deterministic doubles, zero API calls):

```
python -m eval.adversarial.run_pilot
```

Run the original action-reachability red-team (14 cases, seed 1337):

```
python -m eval.run_redteam
```

The live attack evaluation runs through the resumable runner in eval/adversarial/live_runner.py and needs provider API keys. Live outputs are written to eval/results/, which is git-ignored and kept out of the repository on purpose.

## Layout

```
agents/    Buyer, Interpreter, Vendor
schemas/   the VendorOffer contract
guards/    deterministic checks (not agents)
eval/      evaluation harness, adversarial matrix, and live runner (ground truth stays here)
tests/     control-level and evaluation tests
```

## Status

Complete. The negotiation system, the deterministic authorization controls, the 25-cell representative matrix, and the live attack evaluation are all implemented and tested.
