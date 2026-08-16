from dataclasses import dataclass

from agents.buyer import Buyer
from agents.interpreter import extract_offer
from agents.vendor import Vendor
from rfq import RFQ


@dataclass
class NegotiationResult:
    outcome: str
    accepted_offer: dict | None
    log: list


def negotiate(rfq: RFQ, vendor: Vendor) -> NegotiationResult:
    """Run a single-vendor negotiation.

    Raw vendor text flows only through the Interpreter boundary; the Buyer sees
    the validated VendorOffer, never the raw message (which is kept in the audit
    log only). Terminates in exactly one of closed_deal / closed_no_deal /
    closed_max_rounds.
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

        offer = extract_offer(raw, vendor_id=vendor.vendor_id, quantity=rfq.quantity)
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
            return NegotiationResult("closed_deal", offer, log)
        if decision.action == "reject":
            return NegotiationResult("closed_no_deal", None, log)
        counter = decision.counter_price

    return NegotiationResult("closed_max_rounds", None, log)
