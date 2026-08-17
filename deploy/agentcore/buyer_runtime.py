"""Buyer AgentCore Runtime entrypoint.

UNDEPLOYED / NOT YET VERIFIED.

Privileged runtime and the sole holder of the order capability. For one negotiation
round it runs the deterministic authorization: re-validate every offer at the
boundary (rejecting raw text / malformed input), run the price-plausibility guard,
run the LangGraph Buyer (real LLM) to obtain a structured BuyerDecision, and — only
for an accept that clears the gate — call the gated place_order. The Buyer LLM never
calls place_order; it only proposes a decision the deterministic layer authorizes.
The Buyer never receives raw vendor text, reservation prices, or evaluation ground
truth.

Fallback behaviour (identified explicitly): the LangGraph repair node substitutes a
safe deterministic BuyerDecision ONLY when the model returns malformed/invalid JSON.
A model call that fails (throttling, unset/invalid model id) raises and the
invocation errors — it does NOT silently fall back. When a fallback decision is
used it is surfaced in the response as ``fallback_used: true`` so it is never
silent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402

from agents.buyer import Buyer, OrderRejected  # noqa: E402
from agents.buyer_graph import make_buyer_decider  # noqa: E402
from deploy.agentcore.bedrock_model import make_bedrock_model  # noqa: E402
from guards.price_guard import (  # noqa: E402
    PRICING_RELEVANT_FLAGS,
    PlausibilityConfig,
    plausibility_block_reason,
)
from guards.schema_validate import validate_offer  # noqa: E402
from rfq import RFQ  # noqa: E402

app = BedrockAgentCoreApp()
_deciders = {}


def _decider_for(model_id):
    if model_id not in _deciders:
        _deciders[model_id] = make_buyer_decider(make_bedrock_model(model_id))
    return _deciders[model_id]


@app.entrypoint
def invoke(payload):
    model_id = os.environ.get("BUYER_MODEL_ID") or payload.get("buyer_model_id")
    if not model_id:
        return {"error": "model_id_not_configured"}

    rfq = RFQ(**payload["rfq"])
    config = PlausibilityConfig(**payload["plausibility_config"])
    # Boundary: only validated VendorOffers are accepted — a raw string or malformed
    # object is rejected here, so raw vendor text never reaches the Buyer.
    offers = [validate_offer(o) for o in payload["offers"]]
    history = payload.get("history", [])
    round_index = int(payload.get("round_index", 0))
    max_rounds = int(payload.get("max_rounds", rfq.max_rounds))

    round_unit_prices = [
        o["unit_price"] for o in offers if o["extraction_flag"] not in PRICING_RELEVANT_FLAGS
    ]
    plausible, blocked = [], []
    for offer in offers:
        reason = plausibility_block_reason(offer, round_unit_prices, config)
        if reason is None:
            plausible.append(offer)
        else:
            blocked.append({"vendor_id": offer["vendor_id"], "reason": reason})

    rfq_view = {
        "item": rfq.item,
        "quantity": rfq.quantity,
        "budget": rfq.budget,
        "max_delivery_days": rfq.max_delivery_days,
    }
    decision = _decider_for(model_id)(rfq_view, plausible, history, round_index, max_rounds)

    result = {
        "decision": decision,
        "blocked": blocked,
        "order": None,
        "fallback_used": decision.get("rationale") == "fallback",
    }
    if decision["action"] == "accept":
        chosen = next((o for o in plausible if o["vendor_id"] == decision["vendor_id"]), None)
        if chosen is None:
            result["blocked"].append({"vendor_id": decision.get("vendor_id"), "reason": "accept_unknown_vendor"})
            return result
        try:
            order = Buyer(rfq).place_order(chosen, round_unit_prices, config)
        except OrderRejected as rejected:
            result["blocked"].append({"vendor_id": chosen["vendor_id"], "reason": rejected.reason})
            return result
        result["order"] = {
            "vendor_id": order.vendor_id,
            "unit_price": order.unit_price,
            "quantity": order.quantity,
            "total": order.total,
        }
    return result


if __name__ == "__main__":
    app.run()
