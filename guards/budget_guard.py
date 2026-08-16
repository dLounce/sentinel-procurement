def order_total(unit_price: float, quantity: int) -> float:
    return unit_price * quantity


def within_budget(unit_price: float, quantity: int, budget: float) -> bool:
    """Deterministic hard budget check. The exact budget is allowed; anything
    above it is rejected."""
    return order_total(unit_price, quantity) <= budget
