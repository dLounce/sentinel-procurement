import json

from eval.adversarial.adaptive import (
    AdaptiveRule,
    BoundedAdaptivePlan,
    BuyerObservation,
    canonical_adaptive_plan,
    run_bounded_adaptive_counterfactual,
)
from eval.adversarial.attacks import AttackFamily, AttackStep, AttackStyle
from eval.adversarial.harness import Bundle
from eval.adversarial.offline import OfflineInterpreterModel
from eval.adversarial.scenarios import S1


class ContinueBuyer:
    def __init__(self, first_action="counter"):
        self.first_action = first_action
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            action = self.first_action
        elif self.calls >= 4:
            action = "accept"
        else:
            action = "counter"
        return json.dumps({
            "action": action,
            "vendor_id": "vendor_a" if action == "accept" else None,
            "counter_price": None if action == "accept" else 40,
            "rationale": "test",
        })


class FixedVendor:
    def __call__(self, prompt):
        return json.dumps({"unit_price": 40, "delivery_days": 8, "note": "commercial terms"})


def bundle(first_action="counter"):
    return Bundle(
        buyer=ContinueBuyer(first_action),
        interpreter=OfflineInterpreterModel(),
        vendor=FixedVendor(),
    )


def test_adaptive_policy_uses_only_normalized_observation():
    plan = canonical_adaptive_plan("vendor_a")
    counter_step = plan.next_step(BuyerObservation("counter", None, 40), 2, 0)
    reject_step = plan.next_step(BuyerObservation("reject", None, None), 2, 0)
    assert counter_step is not None
    assert reject_step is not None
    assert counter_step.family is AttackFamily.ADAPTIVE
    assert reject_step.family is AttackFamily.ADAPTIVE
    assert counter_step.payload != reject_step.payload


def test_adaptive_rule_never_accepts_hidden_information():
    rule = AdaptiveRule("counter", AttackStep(1, AttackFamily.ADAPTIVE, AttackStyle.CONTEXTUAL, "bounded"))
    plan = BoundedAdaptivePlan("vendor_a", 1, (rule,))
    assert plan.next_step(BuyerObservation("counter", None, 40), 2, 0) is not None
    assert plan.next_step(None, 2, 0) is None


def test_counter_response_selects_deterministic_adaptive_payload():
    result = run_bounded_adaptive_counterfactual(S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=7)
    rows = result["attack"]["messages"]
    assert rows
    assert rows[0]["round"] == 2
    assert rows[0]["attack_family"] == AttackFamily.ADAPTIVE.value


def test_reject_response_selects_different_adaptive_payload():
    plan = canonical_adaptive_plan("vendor_a")
    step = plan.next_step(BuyerObservation("reject", None, None), 2, 0)
    assert step is not None
    assert "authority" in step.payload.lower()


def test_adaptive_attack_is_bounded():
    result = run_bounded_adaptive_counterfactual(S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=7)
    assert len(result["attack"]["messages"]) <= 2


def test_adaptive_replays_control_commercial_messages_exactly():
    result = run_bounded_adaptive_counterfactual(S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=9)
    control = {(e["round"], e["vendor_id"]): e["raw_text"] for e in result["control"]["log"] if e["event"] == "vendor_message"}
    treatment = {(e["round"], e["vendor_id"]): e["raw_text"] for e in result["treatment"]["log"] if e["event"] == "vendor_message"}
    for key, control_message in control.items():
        vendor_id = key[1]
        row = next((r for r in result["attack"]["messages"] if (r["round"] - 1, r["attacker_vendor"]) == key), None)
        if vendor_id == "vendor_a" and row is not None:
            assert control_message in treatment[key]
        else:
            assert treatment[key] == control_message


def test_adaptive_execution_is_deterministic():
    first = run_bounded_adaptive_counterfactual(S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=11)
    second = run_bounded_adaptive_counterfactual(S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=11)
    assert first == second


def test_adaptive_outcomes_have_unevaluated_causal_fields():
    result = run_bounded_adaptive_counterfactual(
        S1, canonical_adaptive_plan("vendor_a"), lambda seed: bundle("counter"), seed=7
    )
    assert result["outcomes"]
    assert all(o.model_influenced is None for o in result["outcomes"])
    assert all(o.causal_effect_class is None for o in result["outcomes"])
