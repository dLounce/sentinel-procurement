"""adversarial negotiation orchestrator.

Runs one negotiation using the EXACT production system under test — the injected
LLM Interpreter, the price-plausibility guard, the LangGraph Buyer, BuyerDecision
validation, and the gated place_order — in the production order. The only
differences from production negotiate() are that each vendor carries its own
adversarial model and there is NO fixed-injection path. There is exactly one order
authorization path (Buyer.place_order); the orchestrator never authorizes a
purchase itself.
"""

from agents.buyer import Buyer, OrderRejected
from agents.buyer_graph import make_buyer_decider
from agents.interpreter import ExtractionError, make_interpreter
from agents.vendor import VendorProposalError
from guards.decision_guard import validate_decision
from guards.price_guard import PRICING_RELEVANT_FLAGS, PlausibilityConfig, plausibility_block_reason
from guards.schema_validate import SchemaValidationError
from rfq import RFQ


def run_negotiation(scenario, vendors, buyer_model, interpreter_model,* ,raw_vendor_messages: bool = False, ) -> dict:
    rfq = RFQ(scenario.scenario_id, scenario.item, scenario.quantity, scenario.budget,
              scenario.max_delivery_days, scenario.max_rounds)
    config = PlausibilityConfig(absolute_floor=scenario.absolute_floor)
    interpreter = make_interpreter(interpreter_model)
    buyer_decider = make_buyer_decider(buyer_model)
    buyer = Buyer(rfq)
    rfq_view = {"item": rfq.item, "quantity": rfq.quantity, "budget": rfq.budget,
                "max_delivery_days": rfq.max_delivery_days}

    log = []
    history = []
    counter = None
    for round_index in range(rfq.max_rounds):
        offers = []
        for vendor in vendors:
            try:
                if raw_vendor_messages:
                    if not hasattr(vendor, "propose_raw_message"):
                        raise TypeError(
                            f"{type(vendor).__name__} does not support raw vendor messages"
                        )
                    raw = vendor.propose_raw_message(
                        rfq,
                        round_index,
                        counter,
                        history,
                        model=vendor.model,
                    )
                else:
                    raw = vendor.propose_quote(
                        rfq,
                        round_index,
                        counter,
                        history,
                        model=vendor.model,
                    )
            except (VendorProposalError, ValueError) as exc:
                log.append({"event": "vendor_error", "round": round_index, "vendor_id": vendor.vendor_id, "reason": type(exc).__name__})
                continue
            log.append({"event": "vendor_message", "round": round_index, "vendor_id": vendor.vendor_id, "raw_text": raw})
            try:
                offer = interpreter(raw, vendor_id=vendor.vendor_id, quantity=rfq.quantity)
            except (SchemaValidationError, ExtractionError) as exc:
                log.append({"event": "offer_dropped", "round": round_index, "vendor_id": vendor.vendor_id, "reason": type(exc).__name__})
                continue
            log.append({"event": "offer", "round": round_index, "offer": offer})
            offers.append(offer)

        round_unit_prices = [o["unit_price"] for o in offers if o["extraction_flag"] not in PRICING_RELEVANT_FLAGS]
        plausible = []
        for offer in offers:
            reason = plausibility_block_reason(offer, round_unit_prices, config)
            if reason is None:
                plausible.append(offer)
            else:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": offer["vendor_id"], "reason": reason})

        decision = validate_decision(buyer_decider(rfq_view, plausible, history, round_index, rfq.max_rounds))
        log.append({"event": "decision", "round": round_index, "action": decision["action"], "vendor_id": decision.get("vendor_id"), "counter_price": decision.get("counter_price")})
        history.append({"round": round_index, "offers": plausible, "decision": decision})

        action = decision["action"]
        if action == "accept":
            chosen = next((o for o in plausible if o["vendor_id"] == decision["vendor_id"]), None)
            if chosen is None:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": decision.get("vendor_id"), "reason": "accept_unknown_vendor"})
                continue
            try:
                order = buyer.place_order(chosen, round_unit_prices, config)
            except OrderRejected as rejected:
                log.append({"event": "offer_blocked", "round": round_index, "vendor_id": chosen["vendor_id"], "reason": rejected.reason})
                continue
            order_dict = {"vendor_id": order.vendor_id, "unit_price": order.unit_price, "quantity": order.quantity, "total": order.total, "delivery_days": chosen["delivery_days"]}
            log.append({"event": "order_placed", "round": round_index, "order": order_dict})
            return {"outcome": "closed_deal", "order": order_dict, "log": log}
        if action in ("reject", "walk_away"):
            return {"outcome": "closed_no_deal", "order": None, "log": log}
        counter = decision.get("counter_price")

    return {"outcome": "closed_max_rounds", "order": None, "log": log}
