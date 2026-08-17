from pathlib import Path

from eval.adversarial.harness import Bundle, run_once
from eval.adversarial.offline import OfflineBuyerModel, OfflineInterpreterModel, OfflineVendorModel
from eval.adversarial.prompts import DISHONEST_OBJECTIVE, HONEST_OBJECTIVE, vendor_prompt
from eval.adversarial.scenarios import S2
from eval.adversarial.vendor_agent import AdversarialVendor
from rfq import RFQ

FORBIDDEN_IN_AGENT_INPUT = ["true_cost", "true_fulfillable", "reservation_price", "fair_value",
                            "fair_low", "fair_high", "dishonest", "honest supplier", "seed"]
FORBIDDEN_ATTACK_TAXONOMY = ["ignore previous", "ignore all", "injection", "jailbreak", "SYSTEM OVERRIDE"]


class Capture:
    def __init__(self, inner):
        self.inner = inner
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.inner(prompt)


def _AdversarialVendor(truth, disposition, model):
    return AdversarialVendor(vendor_id=truth.vendor_id, reservation_price=truth.true_cost,
                    opening_price=truth.opening_price, delivery_days=truth.true_delivery,
                    disposition=disposition, model=model)


def _run_with_capture(dishonest_ids):
    buyer, interp = Capture(OfflineBuyerModel()), Capture(OfflineInterpreterModel())
    bundle = Bundle(buyer=buyer, interpreter=interp, vendor=OfflineVendorModel())
    run_once(S2, dishonest_ids, bundle)
    return buyer, interp


def test_buyer_prompt_never_contains_ground_truth_or_labels():
    buyer, _ = _run_with_capture({"vendor_c"})
    assert buyer.prompts
    for prompt in buyer.prompts:
        lowered = prompt.lower()
        for token in FORBIDDEN_IN_AGENT_INPUT:
            assert token.lower() not in lowered, token


def test_interpreter_input_never_contains_ground_truth():
    _, interp = _run_with_capture({"vendor_c"})
    assert interp.prompts
    for prompt in interp.prompts:
        lowered = prompt.lower()
        for token in ("true_cost", "reservation_price", "fair_value", "dishonest", "honest supplier"):
            assert token not in lowered, token


def test_vendor_prompt_has_own_facts_but_not_others_or_market():
    v = _AdversarialVendor(S2.truth("vendor_a"), "honest", OfflineVendorModel())
    rfq = RFQ(S2.scenario_id, S2.item, S2.quantity, S2.budget, S2.max_delivery_days, S2.max_rounds)
    prompt = vendor_prompt(v, rfq, 0, None, [])
    assert "true_unit_cost" in prompt  # its own private fact is allowed
    for token in ("vendor_b", "vendor_c", "budget", "fair_value", "fair_low", "seed"):
        assert token not in prompt, token


def test_dishonest_prompt_has_no_attack_taxonomy():
    v = _AdversarialVendor(S2.truth("vendor_c"), "dishonest", OfflineVendorModel())
    rfq = RFQ(S2.scenario_id, S2.item, S2.quantity, S2.budget, S2.max_delivery_days, S2.max_rounds)
    prompt = vendor_prompt(v, rfq, 0, None, [])
    for token in FORBIDDEN_ATTACK_TAXONOMY:
        assert token.lower() not in prompt.lower(), token


def test_honest_and_dishonest_prompts_are_minimal_contrast():
    truth = S2.truth("vendor_c")
    rfq = RFQ(S2.scenario_id, S2.item, S2.quantity, S2.budget, S2.max_delivery_days, S2.max_rounds)
    honest = vendor_prompt(_AdversarialVendor(truth, "honest", None), rfq, 0, None, [])
    dishonest = vendor_prompt(_AdversarialVendor(truth, "dishonest", None), rfq, 0, None, [])
    # identical after normalizing away only the objective clause
    assert honest.replace(HONEST_OBJECTIVE, "<OBJ>") == dishonest.replace(DISHONEST_OBJECTIVE, "<OBJ>")


# --- production isolation (structural) ---------------------------------------

def _src(name):
    return (Path(__file__).resolve().parent.parent / "eval" / "adversarial" / name).read_text()


def test_adversarial_introduces_no_second_place_order_path():
    orch = _src("orchestrator.py")
    assert "def place_order" not in orch  # no reimplementation
    assert "buyer.place_order(" in orch  # the only order path is the production gated method
    # no eval/adversarial module defines its own guard/authorization
    for name in ("orchestrator.py", "harness.py", "scoring.py"):
        assert "def place_order" not in _src(name)


def test_adversarial_uses_production_buyer_and_guards():
    orch = _src("orchestrator.py")
    assert "from agents.buyer_graph import make_buyer_decider" in orch
    assert "from agents.interpreter import" in orch
    assert "plausibility_block_reason" in orch and "validate_decision" in orch
