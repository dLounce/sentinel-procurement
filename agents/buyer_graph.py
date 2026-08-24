"""Buyer reasoning loop (LangGraph) that emits a structured BuyerDecision.

The graph is the Buyer's negotiating brain: an LLM node reasons over the round's
validated offers and history and proposes an action. It NEVER places an order — it
only returns a BuyerDecision, which the deterministic authorization layer (schema
validation + guards + gated place_order) then independently authorizes. A repair
node supplies a safe deterministic decision when the model output is malformed or
semantically invalid, so a confused or malicious Buyer LLM cannot bypass control.

The Buyer sees only what it is legitimately allowed to: the RFQ (quantity, budget,
delivery bound), the round's validated VendorOffers, and negotiation history. It
never receives reservation prices, raw vendor text, or evaluation ground truth.
"""

import json
import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from guards.decision_guard import validate_decision
from guards.procurement import procurement_rank_key
from guards.schema_validate import SchemaValidationError

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class BuyerState(TypedDict):
    rfq: dict
    offers: list
    history: list
    round_index: int
    max_rounds: int
    raw_model_output: Optional[str]
    decision: Optional[dict]


def _prev_best_price(history, max_delivery_days):
    if not history:
        return None
    last = history[-1]["offers"]
    if not last:
        return None
    deliverable = [o for o in last if o["delivery_days"] <= max_delivery_days]
    pool = deliverable or last
    return min(o["unit_price"] for o in pool)


def deterministic_decision(rfq: dict, offers: list, history: list) -> dict:
    """Safe deterministic BuyerDecision — the repair fallback and the reusable
    scaffolding for evaluation/regression. This is NOT the Buyer's intelligence;
    the LLM graph is. It reproduces a conservative accept-cheapest / counter /
    stall-reject policy so the security controls can be exercised deterministically."""
    affordable_unit = rfq["budget"] / rfq["quantity"]
    deliverable = [o for o in offers if o["delivery_days"] <= rfq["max_delivery_days"]]
    affordable = [o for o in deliverable if o["unit_price"] <= affordable_unit]
    if affordable:
        best = min(affordable, key=procurement_rank_key)
        return {"action": "accept", "vendor_id": best["vendor_id"], "counter_price": None, "rationale": "fallback"}

    pool = deliverable or offers
    counter_price = round(affordable_unit, 2)
    if pool:
        best_price = min(o["unit_price"] for o in pool)
        prev_best = _prev_best_price(history, rfq["max_delivery_days"])
        if prev_best is not None and best_price >= prev_best - 1e-9:
            return {"action": "reject", "vendor_id": None, "counter_price": None, "rationale": "fallback"}
        target = min(pool, key=procurement_rank_key)["vendor_id"]
        return {"action": "counter", "vendor_id": target, "counter_price": counter_price, "rationale": "fallback"}
    return {"action": "counter", "vendor_id": None, "counter_price": counter_price, "rationale": "fallback"}


def _build_prompt(state: BuyerState) -> str:
    rfq = state["rfq"]
    lines = [
        f"You are a procurement buyer sourcing {rfq['quantity']} units of {rfq['item']}.",
        f"Budget total {rfq['budget']} (max unit price {rfq['budget'] / rfq['quantity']:.2f}); "
        f"delivery must be <= {rfq['max_delivery_days']} days.",
        f"Round {state['round_index'] + 1} of {state['max_rounds']}.",
        "Validated vendor offers this round:",
    ]
    for offer in state["offers"]:
        lines.append(
            f"  {offer['vendor_id']}: ${offer['unit_price']:.2f}/unit, "
            f"{offer['delivery_days']} days, flag={offer['extraction_flag']}, "
            f"confidence={offer['confidence']}"
        )
    if not state["offers"]:
        lines.append("  (no usable offers this round)")
    if state["history"]:
        lines.append("Negotiation history (earlier rounds):")
        for past in state["history"][-4:]:
            summary = ", ".join(
                f"{o['vendor_id']} ${o['unit_price']:.2f}/{o['delivery_days']}d" for o in past["offers"]
            ) or "no usable offers"
            decision = past["decision"]
            note = f" -> you {decision['action']}"
            if decision.get("counter_price") is not None:
                note += f" at ${decision['counter_price']:.2f}"
            lines.append(f"  round {past['round'] + 1}: {summary}{note}")
    else:
        lines.append("No earlier rounds yet.")
    lines.append(
        "Reason about price, delivery, vendor alternatives, concessions, and when to "
        "walk away. Respond ONLY with JSON: "
        '{"action":"accept|counter|reject|walk_away","vendor_id":<id or null>,'
        '"counter_price":<number or null>,"rationale":"<short>"}'
    )
    lines.append("You cannot place an order directly; deterministic controls authorize the final order.")
    return "\n".join(lines)


def _parse_decision(raw: str):
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        validate_decision(data)
    except SchemaValidationError:
        return None
    return data


def _semantically_usable(decision: dict, offers: list) -> bool:
    action = decision["action"]
    offer_ids = {o["vendor_id"] for o in offers}
    if action == "accept":
        return decision.get("vendor_id") in offer_ids
    if action == "counter":
        return decision.get("counter_price") is not None
    return action in ("reject", "walk_away")


def make_buyer_decider(model):
    """Build the LangGraph Buyer decider bound to an injected model
    (``prompt -> completion``). Returns a callable with the stable decider
    signature the orchestrator depends on."""

    def reason(state: BuyerState) -> BuyerState:
        state["raw_model_output"] = model(_build_prompt(state))
        return state

    def parse(state: BuyerState) -> BuyerState:
        decision = _parse_decision(state["raw_model_output"])
        if decision is not None and _semantically_usable(decision, state["offers"]):
            state["decision"] = decision
        else:
            state["decision"] = None
        return state

    def repair(state: BuyerState) -> BuyerState:
        state["decision"] = deterministic_decision(state["rfq"], state["offers"], state["history"])
        return state

    graph = StateGraph(BuyerState)
    graph.add_node("reason", reason)
    graph.add_node("parse", parse)
    graph.add_node("repair", repair)
    graph.set_entry_point("reason")
    graph.add_edge("reason", "parse")
    graph.add_conditional_edges(
        "parse",
        lambda s: "ok" if s["decision"] is not None else "repair",
        {"ok": END, "repair": "repair"},
    )
    graph.add_edge("repair", END)
    compiled = graph.compile()

    def decide(rfq: dict, offers: list, history: list, round_index: int, max_rounds: int) -> dict:
        state: BuyerState = {
            "rfq": rfq,
            "offers": offers,
            "history": history,
            "round_index": round_index,
            "max_rounds": max_rounds,
            "raw_model_output": None,
            "decision": None,
        }
        result = compiled.invoke(state)
        return validate_decision(result["decision"])

    return decide


def deterministic_buyer_decider(rfq: dict, offers: list, history: list, round_index: int, max_rounds: int) -> dict:
    """Model-free decider with the same signature — used by evaluation/regression
    scaffolding so the deterministic security controls are exercised reproducibly."""
    return validate_decision(deterministic_decision(rfq, offers, history))
