import random
from dataclasses import dataclass

from agents.buyer import Buyer, Order, OrderRejected
from agents.interpreter import ExtractionError
from agents.vendor import Vendor, VendorProposalError
from guards.decision_guard import validate_decision
from guards.price_guard import (
    PRICING_RELEVANT_FLAGS,
    PlausibilityConfig,
    plausibility_block_reason,
)
from guards.schema_validate import SchemaValidationError
from rfq import RFQ


@dataclass
class NegotiationResult:
    outcome: str
    accepted_offer: dict | None
    log: list
    order: Order | None = None


def choose_injection(num_vendors: int, max_rounds: int, rng: random.Random):
    """Randomly pick which vendor injects and on which round, per run."""
    return rng.randrange(num_vendors), rng.randrange(max_rounds)


def _rfq_view(rfq: RFQ) -> dict:
    """Only what the Buyer is legitimately allowed to see — no reservation prices,
    no raw vendor text, no evaluation ground truth."""
    return {
        "item": rfq.item,
        "quantity": rfq.quantity,
        "budget": rfq.budget,
        "max_delivery_days": rfq.max_delivery_days,
    }


def negotiate(
    rfq: RFQ,
    vendors: list[Vendor],
    interpreter,
    buyer_decider,
    plausibility_config: PlausibilityConfig,
    *,
    vendor_model,
    rng: random.Random | None = None,
    injection=None,
) -> NegotiationResult:
    """Run the multi-vendor agentic negotiation.

    Each round every vendor's model proposes a quote (deterministically clamped to
    its private reservation); exactly one vendor on one randomized round injects.
    Every reply crosses the Interpreter boundary + schema validation. The price
    guard runs over the round's offers, then the Buyer's LangGraph/LLM decider
    (``buyer_decider``) emits a structured BuyerDecision — it never places an
    order. The deterministic authorization layer (schema-validated decision +
    gated place_order) independently authorizes any order. Terminates in
    closed_deal / closed_no_deal / closed_max_rounds.
    """
    if injection is None:
        injection = choose_injection(len(vendors), rfq.max_rounds, rng or random.Random())
    inject_vendor_index, inject_round = injection

    buyer = Buyer(rfq)
    rfq_view = _rfq_view(rfq)
    log: list = []
    history: list = []
    counter: float | None = None

    for round_index in range(rfq.max_rounds):
        offers = []
        for i, vendor in enumerate(vendors):
            inject = i == inject_vendor_index and round_index == inject_round
            try:
                raw = vendor.propose_quote(
                    rfq, round_index, counter, history, model=vendor_model, inject=inject
                )
            except VendorProposalError as exc:
                log.append({"event": "vendor_error", "round": round_index, "vendor_id": vendor.vendor_id, "reason": type(exc).__name__})
                continue

            log.append({"event": "vendor_message", "round": round_index, "vendor_id": vendor.vendor_id, "raw_text": raw, "injected": inject})
            try:
                offer = interpreter(raw, vendor_id=vendor.vendor_id, quantity=rfq.quantity)
            except (SchemaValidationError, ExtractionError) as exc:
                log.append({"event": "offer_dropped", "round": round_index, "vendor_id": vendor.vendor_id, "reason": type(exc).__name__})
                continue
            log.append({"event": "offer", "round": round_index, "offer": offer})
            offers.append(offer)

        round_unit_prices = [
            o["unit_price"] for o in offers if o["extraction_flag"] not in PRICING_RELEVANT_FLAGS
        ]

        plausible = []
        for offer in offers:
            reason = plausibility_block_reason(offer, round_unit_prices, plausibility_config)
            if reason is None:
                plausible.append(offer)
            else:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": offer["vendor_id"], "reason": reason})

        decision = validate_decision(
            buyer_decider(rfq_view, plausible, history, round_index, rfq.max_rounds)
        )
        log.append({"event": "decision", "round": round_index, "action": decision["action"], "vendor_id": decision.get("vendor_id"), "counter_price": decision.get("counter_price")})
        history.append({"round": round_index, "offers": plausible, "decision": decision})

        action = decision["action"]
        if action == "accept":
            chosen = next((o for o in plausible if o["vendor_id"] == decision["vendor_id"]), None)
            if chosen is None:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": decision.get("vendor_id"), "reason": "accept_unknown_vendor"})
                continue
            try:
                order = buyer.place_order(chosen, round_unit_prices, plausibility_config)
            except OrderRejected as rejected:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": chosen["vendor_id"], "reason": rejected.reason})
                continue
            log.append({"event": "order_placed", "round": round_index, "order": order})
            return NegotiationResult("closed_deal", chosen, log, order)

        if action in ("reject", "walk_away"):
            return NegotiationResult("closed_no_deal", None, log)

        counter = decision.get("counter_price")

    return NegotiationResult("closed_max_rounds", None, log)
