import importlib.util
import sys
import types
from pathlib import Path


def _load_orchestrator(monkeypatch):
    class FakeRFQ:
        def __init__(self, scenario_id, item, quantity, budget, max_delivery_days, max_rounds):
            self.scenario_id = scenario_id
            self.item = item
            self.quantity = quantity
            self.budget = budget
            self.max_delivery_days = max_delivery_days
            self.max_rounds = max_rounds

    class FakeBuyer:
        def __init__(self, rfq):
            self.rfq = rfq

        def place_order(self, *args, **kwargs):
            raise AssertionError("oversized-round test should not place an order")

    modules = {
        "agents.buyer": types.SimpleNamespace(Buyer=FakeBuyer, OrderRejected=type("OrderRejected", (Exception,), {})),
        "agents.buyer_graph": types.SimpleNamespace(
            make_buyer_decider=lambda model: (lambda *args: {"action": "counter", "counter_price": 0.0})
        ),
        "agents.interpreter": types.SimpleNamespace(
            ExtractionError=type("ExtractionError", (Exception,), {}),
            make_interpreter=lambda model: (lambda *args, **kwargs: None),
        ),
        "agents.vendor": types.SimpleNamespace(VendorProposalError=type("VendorProposalError", (Exception,), {})),
        "guards.decision_guard": types.SimpleNamespace(
            validate_decision=lambda decision: decision
        ),
        "guards.price_guard": types.SimpleNamespace(
            PRICING_RELEVANT_FLAGS=set(),
            PlausibilityConfig=lambda absolute_floor: types.SimpleNamespace(absolute_floor=absolute_floor),
            plausibility_block_reason=lambda *args, **kwargs: None,
        ),
        "guards.schema_validate": types.SimpleNamespace(
            SchemaValidationError=type("SchemaValidationError", (Exception,), {})
        ),
        "rfq": types.SimpleNamespace(RFQ=FakeRFQ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[1] / "eval" / "adversarial" / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("_test_orchestrator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_effective_max_rounds_4_remains_4(monkeypatch):
    module = _load_orchestrator(monkeypatch)
    assert module.effective_max_rounds(4) == 4


def test_effective_max_rounds_5_remains_5(monkeypatch):
    module = _load_orchestrator(monkeypatch)
    assert module.effective_max_rounds(5) == 5


def test_effective_max_rounds_6_becomes_5(monkeypatch):
    module = _load_orchestrator(monkeypatch)
    assert module.effective_max_rounds(6) == 5


def test_effective_max_rounds_100_becomes_5(monkeypatch):
    module = _load_orchestrator(monkeypatch)
    assert module.effective_max_rounds(100) == 5


def test_oversized_evaluator_scenario_executes_at_most_five_rounds(monkeypatch):
    module = _load_orchestrator(monkeypatch)

    class Scenario:
        scenario_id = "oversized"
        item = "widgets"
        quantity = 1
        budget = 100
        max_delivery_days = 10
        max_rounds = 100
        absolute_floor = 1

    result = module.run_negotiation(
        Scenario(),
        [],
        buyer_model=lambda prompt: {"action": "counter", "counter_price": 0.0},
        interpreter_model=lambda prompt: None,
    )

    decision_rounds = [
        event["round"]
        for event in result["log"]
        if event["event"] == "decision"
    ]
    assert decision_rounds == [0, 1, 2, 3, 4]
