from dataclasses import dataclass

from guards.budget_guard import order_total, within_budget
from guards.price_guard import PlausibilityConfig, plausibility_block_reason
from rfq import RFQ


@dataclass
class Decision:
    action: str
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

    decide() accepts only a validated VendorOffer (a dict), never raw vendor
    text. place_order() is the only path to an Order and is born gated: it
    re-runs every deterministic check and refuses unless all pass. It never reads
    ``confidence`` or any model reasoning — those cannot authorize an order.
    """

    def __init__(self, rfq: RFQ):
        self.rfq = rfq
        self._affordable_unit = rfq.budget / rfq.quantity
        self._last_price: float | None = None

    def decide(self, offer: dict) -> Decision:
        if not isinstance(offer, dict):
            raise TypeError("Buyer requires a validated VendorOffer, not raw vendor text")

        price = offer["unit_price"]
        delivery = offer["delivery_days"]

        if price <= self._affordable_unit and delivery <= self.rfq.max_delivery_days:
            return Decision("accept")

        improved = self._last_price is None or price < self._last_price - 1e-9
        self._last_price = price
        if not improved:
            return Decision("reject")
        return Decision("counter", counter_price=round(self._affordable_unit, 2))

    def place_order(
        self,
        offer: dict,
        round_unit_prices,
        plausibility_config: PlausibilityConfig,
    ) -> Order:
        if not isinstance(offer, dict):
            raise TypeError("place_order requires a validated VendorOffer, not raw vendor text")

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
