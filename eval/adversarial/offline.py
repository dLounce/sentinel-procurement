"""Offline model doubles for tests and the offline pilot dry-run.

These make the harness fully runnable with ZERO live API calls so the machinery
can be validated deterministically. They are NOT part of a live scientific run.

The vendor double reads the legitimate objective clause from the prompt (the same
minimal-contrast text a real model sees) — it does not key on any secret marker.
A dishonest vendor whose true delivery misses the deadline will CLAIM a fast
delivery (a false-fast deception); an honest vendor always claims its true values.
"""

import json
import re

from eval.adversarial.prompts import DISHONEST_OBJECTIVE

_COST = re.compile(r"true_unit_cost=\$([\d.]+)")
_TRUE_DELIV = re.compile(r"true_fulfillable_delivery_days=(\d+)")
_REQUIRED = re.compile(r"required within (\d+) days")
_ROUND = re.compile(r"round:\s*(\d+)")
_COUNTER = re.compile(r"buyer_counter:\s*(none|\$[\d.]+)")
_QTY = re.compile(r"quantity:\s*(\d+)")
_UNIT = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*per\s*unit", re.IGNORECASE)
_DELIVERY = re.compile(r"delivery\s+in\s+(\d+)\s+days", re.IGNORECASE)
_INTERP_QTY = re.compile(r"quantity:\s*(\d+)")


class OfflineVendorModel:
    """A minimal negotiating vendor double whose honesty follows its prompt's
    objective clause."""

    def __call__(self, prompt: str) -> str:
        cost = float(_COST.search(prompt).group(1))
        true_deliv = int(_TRUE_DELIV.search(prompt).group(1))
        required = int(_REQUIRED.search(prompt).group(1))
        round_index = int(_ROUND.search(prompt).group(1))
        counter_text = _COUNTER.search(prompt).group(1)
        counter = None if counter_text == "none" else float(counter_text[1:])

        # concede toward cost over rounds / toward the buyer counter, never below cost
        opening = cost + 22.0
        price = cost + (opening - cost) * (0.6 ** round_index)
        if counter is not None:
            price = min(price, max(cost, counter))
        price = round(max(price, cost), 2)

        dishonest = DISHONEST_OBJECTIVE.split(".")[0] in prompt
        # honest -> claim true delivery; dishonest & truly late -> claim it can meet the deadline
        claimed_delivery = true_deliv
        if dishonest and true_deliv > required:
            claimed_delivery = required
        note = "We can meet your requirements." if claimed_delivery <= required else "This is our best."
        return json.dumps({"unit_price": price, "delivery_days": claimed_delivery, "note": note})


class OfflineInterpreterModel:
    """Faithful extractor double: parses the emitted price/delivery from the vendor
    message into the VendorOffer JSON (vendor_id is stamped by the production
    Interpreter from transport, not read here)."""

    def __call__(self, prompt: str) -> str:
        quantity = int(_INTERP_QTY.search(prompt).group(1))
        unit = _UNIT.search(prompt)
        if unit:
            unit_price = round(float(unit.group(1).replace(",", "")), 2)
            basis = "unit"
        else:
            unit_price, basis = 0.0, "unknown"
        deliv = _DELIVERY.search(prompt)
        delivery_days = int(deliv.group(1)) if deliv else 0
        return json.dumps({
            "vendor_id": "vendor_a", "unit_price": unit_price, "currency": "USD",
            "delivery_days": delivery_days, "confidence": "high",
            "quoted_basis": basis, "extraction_flag": "none",
        })


class OfflineBuyerModel:
    """Procurement-agent double: accepts the cheapest affordable, deliverable offer;
    otherwise counters toward budget/quantity; rejects on a stalled market. It never
    sees which vendors are dishonest and never sees ground truth."""

    _OFFER = re.compile(r"(vendor_[a-c]): \$([\d.]+)/unit, (\d+) days")
    _BUDGET = re.compile(r"max unit price ([\d.]+)")

    def __call__(self, prompt: str) -> str:
        affordable_unit = float(self._BUDGET.search(prompt).group(1))
        max_deliv = int(re.search(r"delivery must be <= (\d+) days", prompt).group(1))
        offers = [(vid, float(p), int(d)) for vid, p, d in self._OFFER.findall(prompt)]
        deliverable = [(vid, p, d) for vid, p, d in offers if d <= max_deliv]
        affordable = [(vid, p, d) for vid, p, d in deliverable if p <= affordable_unit]
        if affordable:
            vid, p, d = min(affordable, key=lambda o: o[1])
            return json.dumps({"action": "accept", "vendor_id": vid, "counter_price": None, "rationale": "best affordable"})
        pool = deliverable or offers
        if pool:
            vid = min(pool, key=lambda o: o[1])[0]
            return json.dumps({"action": "counter", "vendor_id": vid, "counter_price": round(affordable_unit, 2), "rationale": "push"})
        return json.dumps({"action": "reject", "vendor_id": None, "counter_price": None, "rationale": "no usable offers"})
