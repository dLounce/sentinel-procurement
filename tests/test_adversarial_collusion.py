import json

from eval.adversarial.attacks import (
    AttackFamily,
    AttackStep,
    AttackStyle,
    AttackTrajectory,
    CollusionPlan,
    collusion_stress_plans,
    representative_payload,
)
from eval.adversarial.harness import Bundle, run_collusion_counterfactual
from eval.adversarial.offline import OfflineInterpreterModel
from eval.adversarial.scenarios import S1


class FixedVendorModel:
    def __call__(self, prompt):
        return json.dumps({
            "unit_price": 60,
            "delivery_days": 8,
            "note": "Standard commercial terms.",
        })


class ContinueBuyerModel:
    def __call__(self, prompt):
        round_number = int(prompt.split("Round ", 1)[1].split(" of", 1)[0])
        if round_number < 4:
            return json.dumps({
                "action": "counter",
                "vendor_id": "vendor_a",
                "counter_price": 40,
                "rationale": "continue",
            })
        return json.dumps({
            "action": "reject",
            "vendor_id": None,
            "counter_price": None,
            "rationale": "stop",
        })


def bundle(seed):
    return Bundle(
        buyer=ContinueBuyerModel(),
        interpreter=OfflineInterpreterModel(),
        vendor=FixedVendorModel(),
    )


class WinningVendorModel(FixedVendorModel):
    def __call__(self, prompt):
        return json.dumps({
            "unit_price": 43,
            "delivery_days": 8,
            "note": "Standard commercial terms.",
        })


class WinningBuyerModel(ContinueBuyerModel):
    def __call__(self, prompt):
        round_number = int(prompt.split("Round ", 1)[1].split(" of", 1)[0])
        if round_number < 4:
            return super().__call__(prompt)
        return json.dumps({
            "action": "accept",
            "vendor_id": "vendor_a",
            "counter_price": None,
            "rationale": "best feasible offer",
        })


def winning_bundle(seed):
    return Bundle(
        buyer=WinningBuyerModel(),
        interpreter=OfflineInterpreterModel(),
        vendor=WinningVendorModel(),
    )


def plan(*trajectories, collusion_id="test-collusion"):
    return CollusionPlan(collusion_id, tuple(trajectories))


def trajectory(vendor_id, round_number, family, style):
    return AttackTrajectory(
        vendor_id,
        (AttackStep(round_number, family, style, representative_payload(family)),),
    )


def test_one_two_three_malicious_vendor_stress_suite_is_explicit_and_deterministic():
    plans = collusion_stress_plans()
    assert [len(p.attacker_vendors) for p in plans] == [1, 2, 3]
    assert plans[1].collusion_id.endswith("coordinated")
    assert plans[2].collusion_id.endswith("coordinated")
    assert plans == collusion_stress_plans()


def test_one_malicious_vendor_runs_and_records_count():
    p = collusion_stress_plans()[0]
    result = run_collusion_counterfactual(S1, p, bundle, seed=1)
    assert result["number_of_malicious_vendors"] == 1
    assert result["attacker_vendors"] == ["vendor_a"]
    assert {outcome.number_of_malicious_vendors for outcome in result["outcomes"]} == {1}


def test_two_malicious_vendors_have_independent_trajectories_and_coordinated_group():
    p = collusion_stress_plans()[1]
    assert p.trajectory_for_vendor("vendor_b").payload_for_round(2)
    assert p.trajectory_for_vendor("vendor_c").payload_for_round(3)
    assert p.trajectory_for_vendor("vendor_b").payload_for_round(3) is None

    result = run_collusion_counterfactual(S1, p, bundle, seed=2)
    assert result["number_of_malicious_vendors"] == 2
    assert {o.attacker_vendor for o in result["outcomes"]} == {"vendor_b", "vendor_c"}
    assert {o.collusion_id for o in result["outcomes"]} == {p.collusion_id}


def test_three_malicious_vendors_can_run_with_three_independent_trajectories():
    p = collusion_stress_plans()[2]
    result = run_collusion_counterfactual(S1, p, bundle, seed=3)
    assert result["number_of_malicious_vendors"] == 3
    assert set(result["attacker_vendors"]) == {"vendor_a", "vendor_b", "vendor_c"}
    assert {o.attacker_vendor for o in result["outcomes"]} == {"vendor_a", "vendor_b", "vendor_c"}


def test_conflicting_narratives_are_explicit_and_vendor_bound():
    p = plan(
        trajectory("vendor_a", 2, AttackFamily.COMPETITOR_SUPPRESSION, AttackStyle.SUBTLE),
        trajectory("vendor_b", 2, AttackFamily.IDENTITY_SPOOFING, AttackStyle.ROLE_BASED),
    )
    assert p.trajectory_for_vendor("vendor_a").step_for_round(2).family == AttackFamily.COMPETITOR_SUPPRESSION
    assert p.trajectory_for_vendor("vendor_b").step_for_round(2).family == AttackFamily.IDENTITY_SPOOFING

    result = run_collusion_counterfactual(S1, p, bundle, seed=4)
    rows = result["attack"]["messages"]
    assert {(r["attacker_vendor"], r["round"]) for r in rows} == {("vendor_a", 2), ("vendor_b", 2)}


def test_treatment_changes_only_attacker_payloads():
    p = collusion_stress_plans()[1]
    result = run_collusion_counterfactual(S1, p, bundle, seed=5)
    control = {
        (e["round"], e["vendor_id"]): e["raw_text"]
        for e in result["control"]["log"] if e["event"] == "vendor_message"
    }
    treatment = {
        (e["round"], e["vendor_id"]): e["raw_text"]
        for e in result["treatment"]["log"] if e["event"] == "vendor_message"
    }
    for key, text in control.items():
        if key[1] in {"vendor_b", "vendor_c"} and key[0] in {1, 2}:
            assert text in treatment[key]
        else:
            assert treatment[key] == text


def test_repeated_collusion_execution_is_deterministic_and_security_is_separate():
    p = collusion_stress_plans()[2]
    first = run_collusion_counterfactual(S1, p, bundle, seed=7)
    second = run_collusion_counterfactual(S1, p, bundle, seed=7)
    assert first == second
    for outcome in first["outcomes"]:
        assert outcome.unauthorized_action is False
        assert outcome.system_compromised is False
        assert outcome.policy_violated is False


def test_malicious_vendor_winning_is_not_security_success():
    outcome = run_collusion_counterfactual(S1, collusion_stress_plans()[0], winning_bundle, seed=8)["outcomes"][0]
    # The outcome layer records security boundary events, not attacker victory.
    assert outcome.selected_vendor is not None
    assert outcome.system_compromised is False
    assert outcome.unauthorized_action is False


def test_collusion_outcomes_have_unevaluated_causal_fields():
    result = run_collusion_counterfactual(S1, collusion_stress_plans()[1], bundle, seed=5)
    outs = result["outcomes"]
    assert outs
    assert all(o.model_influenced is None for o in outs)
    assert all(o.causal_effect_class is None for o in outs)


def test_collusion_multiple_rows_share_one_trial_id_and_dedupe():
    from eval.adversarial.evidence import build_metric_evidence
    result = run_collusion_counterfactual(S1, collusion_stress_plans()[1], bundle, seed=6)
    outs = result["outcomes"]
    assert len(outs) >= 2                                 # two malicious vendors -> multiple rows
    assert len({o.run_identifier for o in outs}) == 1     # one shared trial id
    ev = build_metric_evidence(
        outs, scenario_definitions={}, attack_family_definitions={},
        model_configuration={}, replications=1,
        inclusion_rules={"uar": "", "pvr": "", "model_susceptibility": "", "attacker_self_selection": ""},
    )
    assert ev["metrics"]["uar"]["denominator"] == 1                     # deduped to one trial
    assert ev["metrics"]["model_susceptibility"]["denominator"] == 0    # unevaluated -> excluded
    assert ev["metrics"]["model_susceptibility"]["excluded_trial_count"] == 1
