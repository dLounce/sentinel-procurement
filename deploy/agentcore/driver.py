"""Local negotiation driver (client) for the deployed runtimes.

UNDEPLOYED / NOT YET VERIFIED. Runs on your machine, NOT as a runtime. It hosts the
Vendors (with their private reservation prices, kept local) and drives the
negotiation loop by invoking the two AgentCore Runtimes over the AWS SDK:

  Vendor model (local) -> raw text -> Interpreter Runtime -> validated VendorOffer
    -> Buyer Runtime (price guard + LangGraph Buyer + deterministic gate + order)

The driver never runs place_order itself and never sends raw vendor text to the
Buyer Runtime — it forwards only validated VendorOffers. Vendor reservation prices
never leave the local Vendor objects. This is the caller identity that holds
bedrock-agentcore:InvokeAgentRuntime; the runtimes' own execution roles do not.
"""

import json
import os
import uuid

import boto3

from agents.vendor import Vendor, VendorProposalError
from deploy.agentcore.bedrock_model import make_bedrock_model

INTERPRETER_ARN = os.environ.get("INTERPRETER_RUNTIME_ARN")
BUYER_ARN = os.environ.get("BUYER_RUNTIME_ARN")
INTERPRETER_MODEL_ID = os.environ.get("INTERPRETER_MODEL_ID")
BUYER_MODEL_ID = os.environ.get("BUYER_MODEL_ID")
VENDOR_MODEL_ID = os.environ.get("VENDOR_MODEL_ID")

_client = boto3.client("bedrock-agentcore")
_vendor_model = make_bedrock_model(VENDOR_MODEL_ID)  # a cheaper Bedrock model, local caller creds


def _invoke_runtime(arn: str, payload: dict) -> dict:
    response = _client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=str(uuid.uuid4()),
        payload=json.dumps(payload).encode(),
        qualifier="DEFAULT",
    )
    chunks = [c.decode("utf-8") for c in response.get("response", [])]
    return json.loads("".join(chunks))


def negotiate_remote(rfq_dict: dict, vendors, plausibility_config: dict, max_rounds: int):
    history = []
    counter = None
    for round_index in range(max_rounds):
        offers = []
        for vendor in vendors:
            try:
                raw = vendor.propose_quote(_Rfq(rfq_dict), round_index, counter, history, model=_vendor_model)
            except VendorProposalError:
                continue
            extracted = _invoke_runtime(INTERPRETER_ARN, {"raw_text": raw, "vendor_id": vendor.vendor_id, "quantity": rfq_dict["quantity"], "model_id": INTERPRETER_MODEL_ID})
            if "offer" in extracted:
                offers.append(extracted["offer"])

        result = _invoke_runtime(
            BUYER_ARN,
            {
                "rfq": rfq_dict,
                "offers": offers,
                "plausibility_config": plausibility_config,
                "history": history,
                "round_index": round_index,
                "max_rounds": max_rounds,
                "buyer_model_id": BUYER_MODEL_ID,
            },
        )
        decision = result["decision"]
        history.append({"round": round_index, "offers": offers, "decision": decision})
        if result.get("order"):
            return {"outcome": "closed_deal", "order": result["order"], "history": history}
        if decision["action"] in ("reject", "walk_away"):
            return {"outcome": "closed_no_deal", "order": None, "history": history}
        counter = decision.get("counter_price")
    return {"outcome": "closed_max_rounds", "order": None, "history": history}


class _Rfq:
    """Minimal RFQ view for the local Vendor (matches the fields Vendor reads)."""

    def __init__(self, d):
        self.item = d["item"]
        self.quantity = d["quantity"]
        self.budget = d["budget"]
        self.max_delivery_days = d["max_delivery_days"]
        self.max_rounds = d.get("max_rounds", 8)


if __name__ == "__main__":
    rfq = {"rfq_id": "rfq-1", "item": "widgets", "quantity": 200, "budget": 9000, "max_delivery_days": 10, "max_rounds": 6}
    vendors = [
        Vendor("vendor_a", reservation_price=40, opening_price=60, delivery_days=8),
        Vendor("vendor_b", reservation_price=42, opening_price=62, delivery_days=7),
        Vendor("vendor_c", reservation_price=44, opening_price=64, delivery_days=9),
    ]
    print(json.dumps(negotiate_remote(rfq, vendors, {"absolute_floor": 20.0}, rfq["max_rounds"]), default=str, indent=2))
