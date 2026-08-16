import random
from dataclasses import dataclass

from agents.buyer import Buyer, Order, OrderRejected
from agents.interpreter import ExtractionError
from agents.vendor import Vendor
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
    """Randomly pick which vendor injects and on which round. Per run, so no
    vendor is permanently the attacker and the Buyer cannot learn to distrust
    one. The Buyer is never told the result."""
    return rng.randrange(num_vendors), rng.randrange(max_rounds)


def negotiate(
    rfq: RFQ,
    vendors: list[Vendor],
    interpreter,
    plausibility_config: PlausibilityConfig,
    *,
    rng: random.Random | None = None,
    injection=None,
) -> NegotiationResult:
    """Run the multi-vendor negotiation.

    Each round the RFQ (or a counter) fans out to every vendor; exactly one
    vendor on one randomized round embeds an injection in its otherwise-normal
    reply. Every reply crosses the Interpreter boundary and schema validation
    before the Buyer sees it — the Buyer never reads raw vendor text and never
    sees a reservation price. The price guard runs over the round's offers; the
    Buyer reasons over the survivors and orders only through the gated
    place_order. Terminates in closed_deal / closed_no_deal / closed_max_rounds.
    """
    if injection is None:
        injection = choose_injection(len(vendors), rfq.max_rounds, rng or random.Random())
    inject_vendor_index, inject_round = injection

    buyer = Buyer(rfq)
    log: list = []
    counter: float | None = None

    for round_index in range(rfq.max_rounds):
        offers = []
        for i, vendor in enumerate(vendors):
            inject = i == inject_vendor_index and round_index == inject_round
            raw = vendor.quote(rfq, round_index, counter, inject=inject)
            log.append(
                {
                    "event": "vendor_message",
                    "round": round_index,
                    "vendor_id": vendor.vendor_id,
                    "raw_text": raw,
                    "injected": inject,
                }
            )
            try:
                offer = interpreter(raw, vendor_id=vendor.vendor_id, quantity=rfq.quantity)
            except (SchemaValidationError, ExtractionError) as exc:
                # fail closed: an unparseable or schema-invalid extraction drops
                # this vendor's offer for the round rather than reaching the Buyer.
                log.append(
                    {
                        "event": "offer_dropped",
                        "round": round_index,
                        "vendor_id": vendor.vendor_id,
                        "reason": type(exc).__name__,
                    }
                )
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
                log.append(
                    {
                        "event": "offer_blocked",
                        "round": round_index,
                        "vendor_id": offer["vendor_id"],
                        "reason": reason,
                    }
                )

        decision = buyer.decide_round(plausible)
        log.append(
            {
                "event": "decision",
                "round": round_index,
                "action": decision.action,
                "counter_price": decision.counter_price,
            }
        )

        if decision.action == "accept":
            chosen = decision.offer
            try:
                order = buyer.place_order(chosen, round_unit_prices, plausibility_config)
            except OrderRejected as rejected:
                log.append(
                    {
                        "event": "offer_blocked",
                        "round": round_index,
                        "vendor_id": chosen["vendor_id"],
                        "reason": rejected.reason,
                    }
                )
                counter = round(rfq.budget / rfq.quantity, 2)
                continue
            log.append({"event": "order_placed", "round": round_index, "order": order})
            return NegotiationResult("closed_deal", chosen, log, order)
        if decision.action == "reject":
            return NegotiationResult("closed_no_deal", None, log)
        counter = decision.counter_price

    return NegotiationResult("closed_max_rounds", None, log)
