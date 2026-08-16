from dataclasses import dataclass

from guards.budget_guard import order_total, within_budget
from guards.price_guard import PlausibilityConfig, plausibility_block_reason
from rfq import RFQ


@dataclass
class RoundDecision:
    action: str  # "accept" | "counter" | "reject"
    offer: dict | None = None  # the chosen VendorOffer when action == "accept"
    counter_price: float | None = None


@dataclass(frozen=True)
class Order:
    vendor_id: str
    unit_price: float
    quantity: int
    total: float


class OrderRejected(Exception):
    """Raised when a deterministic order gate refuses to place an order."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Buyer:
    """Privileged negotiator and sole holder of the order capability.

    It reasons only over validated VendorOffers (never raw vendor text) and never
    sees any vendor's reservation price. place_order() is the only path to an
    Order and is born gated: it re-runs every deterministic check and refuses
    unless all pass. It never reads ``confidence`` or any model reasoning.
    """

    def __init__(self, rfq: RFQ):
        self.rfq = rfq
        self._affordable_unit = rfq.budget / rfq.quantity
        self._last_price: float | None = None

    def decide_round(self, offers) -> RoundDecision:
        for offer in offers:
            if not isinstance(offer, dict):
                raise TypeError("Buyer requires validated VendorOffers, not raw vendor text")

        deliverable = [o for o in offers if o["delivery_days"] <= self.rfq.max_delivery_days]
        affordable = [o for o in deliverable if o["unit_price"] <= self._affordable_unit]
        if affordable:
            best = min(affordable, key=lambda o: o["unit_price"])
            return RoundDecision("accept", offer=best)

        pool = deliverable or offers
        best_price = min((o["unit_price"] for o in pool), default=None)
        improved = self._last_price is None or (
            best_price is not None and best_price < self._last_price - 1e-9
        )
        self._last_price = best_price
        if not improved:
            return RoundDecision("reject")
        return RoundDecision("counter", counter_price=round(self._affordable_unit, 2))

    def place_order(
        self,
        offer: dict,
        round_unit_prices,
        plausibility_config: PlausibilityConfig,
    ) -> Order:
        if not isinstance(offer, dict):
            raise TypeError("place_order requires a validated VendorOffer, not raw vendor text")

        if offer["extraction_flag"] == "unparseable_delivery":
            raise OrderRejected("delivery_flag")
        if offer["delivery_days"] > self.rfq.max_delivery_days:
            raise OrderRejected("delivery")
        if not within_budget(offer["unit_price"], self.rfq.quantity, self.rfq.budget):
            raise OrderRejected("budget")
        reason = plausibility_block_reason(offer, round_unit_prices, plausibility_config)
        if reason is not None:
            raise OrderRejected(reason)

        return Order(
            vendor_id=offer["vendor_id"],
            unit_price=offer["unit_price"],
            quantity=self.rfq.quantity,
            total=order_total(offer["unit_price"], self.rfq.quantity),
        )
