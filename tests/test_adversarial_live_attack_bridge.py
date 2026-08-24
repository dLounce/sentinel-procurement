import json

from eval.adversarial.attacks import (
    ATTACKS_BY_FAMILY,
    AttackFamily,
    AttackStep,
    AttackStyle,
    AttackTrajectory,
)
from eval.adversarial.harness import Bundle
from eval.adversarial.live_runner import run_live_attack_counterfactuals
from eval.adversarial.offline import OfflineInterpreterModel
from eval.adversarial.orchestrator import HARD_MAX_ROUNDS, effective_max_rounds
from eval.adversarial.scenarios import S3


PILOT_FAMILIES = (
    AttackFamily.DIRECT_GOAL_PROMPT_HIJACKING,
    AttackFamily.POLICY_OVERRIDE,
    AttackFamily.NEGOTIATION_CONTEXT_POISONING,
    AttackFamily.STRUCTURED_OUTPUT_ACTION_FIELD_INJECTION,
    AttackFamily.TOOL_USE_INJECTION,
)


def trajectories():
    return tuple(
        AttackTrajectory(
            "vendor_a",
            (
                AttackStep(
                    3,
                    family,
                    ATTACKS_BY_FAMILY[family].style,
                    ATTACKS_BY_FAMILY[family].payload,
                ),
            ),
        )
        for family in PILOT_FAMILIES
    )


class FixedVendorModel:
    def __call__(self, prompt):
        return '{"unit_price": 60, "delivery_days": 8, "note": "Standard commercial terms."}'


class ContinueBuyerModel:
    def __call__(self, prompt):
        import re
        match = re.search(r"Round (\d+) of (\d+)\.", prompt)
        round_number = int(match.group(1))
        max_rounds = int(match.group(2))
        if round_number < max_rounds:
            return '{"action": "counter", "vendor_id": "vendor_a", "counter_price": 40, "rationale": "continue"}'
        return '{"action": "reject", "vendor_id": null, "counter_price": null, "rationale": "stop"}'


def bundle(seed):
    return Bundle(
        buyer=ContinueBuyerModel(),
        interpreter=OfflineInterpreterModel(),
        vendor=FixedVendorModel(),
    )


def run_bridge(tmp_path, run_id="bridge_test", **kwargs):
    return run_live_attack_counterfactuals(
        [S3],
        trajectories(),
        bundle,
        k=2,
        run_id=run_id,
        max_trials=10,
        target_vendor="vendor_a",
        aa_repetitions=2,
        results_dir=tmp_path,
        **kwargs,
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_exact_five_requested_families_are_selected(tmp_path):
    summary = run_bridge(tmp_path)
    records = _records(tmp_path / "bridge_test.jsonl")
    manifest = json.loads((tmp_path / "bridge_test.manifest.json").read_text(encoding="utf-8"))

    assert summary["planned_trials"] == 10
    assert manifest["attack_families"] == sorted(family.value for family in PILOT_FAMILIES)
    assert {record["attack_family"][0] for record in records} == {
        family.value for family in PILOT_FAMILIES
    }


def test_two_replications_per_family_produce_exactly_ten_attack_trials(tmp_path):
    summary = run_bridge(tmp_path)
    records = _records(tmp_path / "bridge_test.jsonl")

    assert summary["executed_this_call"] == 10
    assert len(records) == 10
    counts = {}
    for record in records:
        counts[record["attack_family"][0]] = counts.get(record["attack_family"][0], 0) + 1
    assert counts == {family.value: 2 for family in PILOT_FAMILIES}


def test_each_trial_carries_the_correct_attack_family(tmp_path):
    run_bridge(tmp_path)
    records = _records(tmp_path / "bridge_test.jsonl")

    for record in records:
        family = AttackFamily(record["attack_family"][0])
        assert record["attack_trajectory"]["steps"][0]["family"] == family.value
        attack_round = record["attack_trajectory"]["steps"][0]["target_round"]
        attack_row = next(
            row for row in record["result"]["attack"]["messages"]
            if row["round"] == attack_round
        )
        assert attack_row["attack_family"] == family.value


def test_attacker_vendor_is_bound_to_vendor_a(tmp_path):
    run_bridge(tmp_path)
    records = _records(tmp_path / "bridge_test.jsonl")

    assert all(record["attacker_vendor"] == "vendor_a" for record in records)
    assert all(record["target_vendor"] == "vendor_a" for record in records)
    assert all(record["result"]["target_vendor"] == "vendor_a" for record in records)


def test_aa_repetitions_are_retained(tmp_path):
    run_bridge(tmp_path)
    records = _records(tmp_path / "bridge_test.jsonl")

    assert all(record["aa_repetitions"] == 2 for record in records)
    assert all(len(record["result"]["aa_control"]["repetitions"]) == 2 for record in records)
    assert all("aa_comparison" in record["result"] for record in records)


def test_hard_round_cap_remains_five():
    assert HARD_MAX_ROUNDS == 5
    assert effective_max_rounds(4) == 4
    assert effective_max_rounds(5) == 5
    assert effective_max_rounds(6) == 5
    assert effective_max_rounds(100) == 5


def test_live_result_persistence_is_jsonl_manifest_compatible(tmp_path):
    summary = run_bridge(tmp_path)
    jsonl_path = tmp_path / "bridge_test.jsonl"
    manifest_path = tmp_path / "bridge_test.manifest.json"

    assert summary["results_file"] == str(jsonl_path)
    assert summary["manifest_file"] == str(manifest_path)
    assert jsonl_path.exists()
    assert manifest_path.exists()

    records = _records(jsonl_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(record["status"] == "completed" for record in records)
    assert manifest["kind"] == "live_attack_counterfactual"
    assert manifest["planned_trials"] == 10
    assert manifest["max_rounds"] == 5


def test_repeated_execution_has_stable_trial_ids(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    run_bridge(first_dir, run_id="first")
    run_bridge(second_dir, run_id="second")

    first_ids = [record["trial_id"] for record in _records(first_dir / "first.jsonl")]
    second_ids = [record["trial_id"] for record in _records(second_dir / "second.jsonl")]
    assert first_ids == second_ids


def test_provider_failure_is_recorded_without_security_outcome(tmp_path):
    class FailingVendor:
        def __call__(self, prompt):
            raise RuntimeError("provider unavailable")

    def failing_bundle(seed):
        return Bundle(
            buyer=ContinueBuyerModel(),
            interpreter=OfflineInterpreterModel(),
            vendor=FailingVendor(),
        )

    summary = run_live_attack_counterfactuals(
        [S3],
        trajectories()[:1],
        failing_bundle,
        k=1,
        run_id="provider_failure",
        max_trials=1,
        target_vendor="vendor_a",
        aa_repetitions=2,
        results_dir=tmp_path,
    )

    record = _records(tmp_path / "provider_failure.jsonl")[0]
    assert summary["failed_this_call"] == 1
    assert record["status"] == "failed"
    assert "result" not in record
    assert "unauthorized_action" not in record
    assert "model_influenced" not in record
    assert "policy_violated" not in record
    assert "system_compromised" not in record
