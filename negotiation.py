from dataclasses import dataclass

from agents.buyer import Buyer, Order, OrderRejected
from agents.vendor import Vendor
from guards.price_guard import PlausibilityConfig
from rfq import RFQ


@dataclass
class NegotiationResult:
    outcome: str
    accepted_offer: dict | None
    log: list
    order: Order | None = None


def negotiate(
    rfq: RFQ,
    vendor: Vendor,
    interpreter,
    plausibility_config: PlausibilityConfig,
) -> NegotiationResult:
    """Run a single-vendor negotiation.

    ``interpreter`` is the injected Interpreter boundary
    ``(raw_text, *, vendor_id, quantity) -> VendorOffer``. Raw vendor text flows
    only through it; the Buyer sees the validated VendorOffer, never the raw
    message (kept in the audit log only). An accepted offer becomes a deal only if
    the deterministic order gate (delivery + budget + plausibility) passes; an
    accepted-but-anomalous offer is blocked and negotiation continues. Terminates
    in exactly one of closed_deal / closed_no_deal / closed_max_rounds.
    """
    buyer = Buyer(rfq)
    log: list = []
    counter: float | None = None

    for round_index in range(rfq.max_rounds):
        raw = vendor.quote(rfq, round_index, counter)
        log.append(
            {
                "event": "vendor_message",
                "round": round_index,
                "vendor_id": vendor.vendor_id,
                "raw_text": raw,
            }
        )

        offer = interpreter(raw, vendor_id=vendor.vendor_id, quantity=rfq.quantity)
        log.append({"event": "offer", "round": round_index, "offer": offer})

        decision = buyer.decide(offer)
        log.append(
            {
                "event": "decision",
                "round": round_index,
                "action": decision.action,
                "counter_price": decision.counter_price,
            }
        )

        if decision.action == "accept":
            try:
                order = buyer.place_order(offer, [offer["unit_price"]], plausibility_config)
            except OrderRejected as rejected:
                log.append(
                    {
                        "event": "offer_blocked",
                        "round": round_index,
                        "reason": rejected.reason,
                    }
                )
                counter = round(rfq.budget / rfq.quantity, 2)
                continue
            log.append({"event": "order_placed", "round": round_index, "order": order})
            return NegotiationResult("closed_deal", offer, log, order)
        if decision.action == "reject":
            return NegotiationResult("closed_no_deal", None, log)
        counter = decision.counter_price

    return NegotiationResult("closed_max_rounds", None, log)
