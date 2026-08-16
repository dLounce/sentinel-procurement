from dataclasses import dataclass

from rfq import RFQ


@dataclass
class Decision:
    action: str
    counter_price: float | None = None


class Buyer:
    """Privileged negotiator.

    decide() accepts only a validated VendorOffer (a dict), never raw vendor
    text. The type guard makes "the Buyer never reads untrusted text" true in
    code, not just in a diagram.
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
