import pytest

from rfq import RFQ


def valid(**over):
    args = dict(
        rfq_id="rfq-1",
        item="widgets",
        quantity=200,
        budget=9000.0,
        max_delivery_days=10,
        max_rounds=8,
    )
    args.update(over)
    return RFQ(**args)


def test_valid_rfq_constructs():
    rfq = valid()
    assert rfq.rfq_id == "rfq-1"
    assert rfq.quantity == 200
    assert rfq.budget == 9000.0


def test_empty_rfq_id_rejected():
    with pytest.raises(ValueError):
        valid(rfq_id="")


def test_empty_item_rejected():
    with pytest.raises(ValueError):
        valid(item="")


def test_zero_quantity_rejected():
    with pytest.raises(ValueError):
        valid(quantity=0)


def test_negative_quantity_rejected():
    with pytest.raises(ValueError):
        valid(quantity=-5)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        valid(budget=-1.0)


def test_zero_budget_allowed():
    assert valid(budget=0).budget == 0


def test_negative_max_delivery_rejected():
    with pytest.raises(ValueError):
        valid(max_delivery_days=-1)


def test_zero_max_delivery_allowed():
    assert valid(max_delivery_days=0).max_delivery_days == 0


def test_zero_max_rounds_rejected():
    with pytest.raises(ValueError):
        valid(max_rounds=0)


def test_negative_max_rounds_rejected():
    with pytest.raises(ValueError):
        valid(max_rounds=-1)


def test_one_max_round_allowed():
    assert valid(max_rounds=1).max_rounds == 1


def test_non_finite_budget_rejected():
    with pytest.raises(ValueError):
        valid(budget=float("inf"))
    with pytest.raises(ValueError):
        valid(budget=float("nan"))


def test_evaluator_scenario_rfqs_remain_valid():
    # the deterministic evaluation set must still construct cleanly under validation
    from eval.cases import build_cases

    cases = build_cases()
    assert cases
    for c in cases:
        assert c.rfq.quantity > 0
        assert c.rfq.budget >= 0
        assert c.rfq.max_rounds >= 1
