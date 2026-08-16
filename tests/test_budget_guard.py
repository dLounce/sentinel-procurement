from guards.budget_guard import order_total, within_budget


def test_valid_in_budget_order():
    assert within_budget(unit_price=40.0, quantity=200, budget=9000) is True


def test_exact_budget_boundary_allowed():
    # 45 * 200 == 9000 exactly
    assert order_total(45.0, 200) == 9000
    assert within_budget(unit_price=45.0, quantity=200, budget=9000) is True


def test_budget_overrun_rejected():
    assert within_budget(unit_price=45.01, quantity=200, budget=9000) is False


def test_just_over_budget_rejected():
    assert within_budget(unit_price=45.005, quantity=200, budget=9000) is False
