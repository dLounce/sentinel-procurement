from dataclasses import dataclass

from rfq import RFQ


@dataclass
class Vendor:
    """A single merchant principal.

    reservation_price is private: it is never written into the quote text and
    never leaves this object, which is what keeps the negotiation a real
    information-asymmetry problem rather than a lookup.
    """

    vendor_id: str
    reservation_price: float
    start_price: float
    delivery_days: int
    concession: float = 0.25

    def quote(self, rfq: RFQ, round_index: int, buyer_counter: float | None = None) -> str:
        ask = self._ask(round_index, buyer_counter)
        return (
            f"Thanks for the RFQ on {rfq.item}. "
            f"Our best price is ${ask:.2f} per unit, delivery in {self.delivery_days} days."
        )

    def _ask(self, round_index: int, buyer_counter: float | None) -> float:
        gap = self.start_price - self.reservation_price
        scheduled = self.reservation_price + gap * (1 - self.concession) ** round_index
        if buyer_counter is not None:
            scheduled = min(scheduled, max(self.reservation_price, buyer_counter))
        return round(scheduled, 2)
