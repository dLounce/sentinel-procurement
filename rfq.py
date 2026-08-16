from dataclasses import dataclass


@dataclass(frozen=True)
class RFQ:
    rfq_id: str
    item: str
    quantity: int
    budget: float
    max_delivery_days: int
    max_rounds: int
