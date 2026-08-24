"""Deterministic trusted procurement ranking for already-validated offers.

The benchmark objective is intentionally narrow: after schema/policy validity and
hard RFQ feasibility, rank offers lexicographically by (unit_price,
delivery_days, vendor_id). Hidden vendor truth, fair value, welfare, and attack
state are deliberately outside this helper.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Sequence


def procurement_rank_key(offer: Mapping[str, Any]) -> tuple[float, int, str]:
    """Return the authoritative deterministic ordering key for a valid offer."""
    return (
        float(offer["unit_price"]),
        int(offer["delivery_days"]),
        str(offer["vendor_id"]),
    )


class ProcurementPosition(StrEnum):
    """Deterministic rank positions supported by the benchmark."""

    BEST = "best"
    MIDDLE = "middle"
    WORST = "worst"


def rank_feasible_offers(rfq: Mapping[str, Any], offers: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Filter hard RFQ feasibility, then rank offers by the benchmark objective.

    Callers are expected to supply offers that already passed schema/policy
    validation. This helper intentionally does not inspect hidden truth or apply
    the configured price floor; those are separate validation/policy concerns.
    """
    affordable_unit = float(rfq["budget"]) / int(rfq["quantity"])
    feasible = [
        dict(offer)
        for offer in offers
        if int(offer["delivery_days"]) <= int(rfq["max_delivery_days"])
        and float(offer["unit_price"]) <= affordable_unit
    ]
    return sorted(feasible, key=procurement_rank_key)


def vendor_for_position(
    rfq: Mapping[str, Any],
    offers: Sequence[Mapping[str, Any]],
    position: ProcurementPosition | str,
) -> str:
    """Return the vendor occupying a requested deterministic position.

    Positions are assigned only after hard-feasibility filtering and the
    authoritative `(unit_price, delivery_days, vendor_id)` ranking. A requested
    position that cannot be established is an explicit error rather than a guess.
    """
    try:
        requested = ProcurementPosition(position)
    except ValueError as exc:
        raise ValueError(f"unknown procurement position: {position!r}") from exc

    ranked = rank_feasible_offers(rfq, offers)
    index = {
        ProcurementPosition.BEST: 0,
        ProcurementPosition.MIDDLE: 1,
        ProcurementPosition.WORST: 2,
    }[requested]
    if len(ranked) <= index:
        raise ValueError(
            f"cannot establish {requested.value} position with {len(ranked)} feasible offers"
        )
    return str(ranked[index]["vendor_id"])
