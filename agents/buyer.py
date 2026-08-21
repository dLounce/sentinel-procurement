from dataclasses import dataclass

from guards.budget_guard import order_total, within_budget
from guards.price_guard import PlausibilityConfig, plausibility_block_reason
from rfq import RFQ


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
    """Privileged actor and sole holder of the order capability.

    The negotiating brain is a separate LangGraph/LLM decider (agents/buyer_graph)
    that only ever emits a structured BuyerDecision. This class is the
    deterministic authorization layer: place_order() is the only path to an Order
    and is born gated — it re-runs every deterministic check and refuses unless all
    pass, regardless of what the Buyer LLM proposed. It never reads a reservation
    price, raw vendor text, ``confidence``, or any model reasoning.
    """

    def __init__(self, rfq: RFQ):
        self.rfq = rfq

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
        reason = plausibility_block_reason(
            offer,
            round_unit_prices,
            plausibility_config,
        )
        if reason is not None:
            raise OrderRejected(reason)

        return Order(
            vendor_id=offer["vendor_id"],
            unit_price=offer["unit_price"],
            quantity=self.rfq.quantity,
            total=order_total(offer["unit_price"], self.rfq.quantity),
        )
