import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RFQ:
    rfq_id: str
    item: str
    quantity: int
    budget: float
    max_delivery_days: int
    max_rounds: int

    def __post_init__(self):
        # Boundary validation for the procurement request. These are business/task
        # inputs; the check rejects malformed requests, it does not interpret or
        # choose any security policy.
        if not self.rfq_id:
            raise ValueError("rfq_id must be non-empty")
        if not self.item:
            raise ValueError("item must be non-empty")
        if not all(
            math.isfinite(v)
            for v in (self.quantity, self.budget, self.max_delivery_days, self.max_rounds)
        ):
            raise ValueError("RFQ numeric fields must be finite")
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.budget < 0:
            raise ValueError("budget must be >= 0")
        if self.max_delivery_days < 0:
            raise ValueError("max_delivery_days must be >= 0")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
