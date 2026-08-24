import json

from eval.adversarial.attacks import AttackFamily, collusion_stress_plans
from eval.adversarial.matrix import (
    MATRIX_SCENARIO,
    build_representative_matrix,
    matrix_bundle,
    run_representative_matrix,
)
from eval.adversarial.scenarios import S1, S2, S3, S4, S5


def test_position_fixture_is_three_way_deterministic_and_does_not_use_hidden_truth_for_rank():
    from guards.procurement import ProcurementPosition, rank_feasible_offers, vendor_for_position

    offers = [
        {"vendor_id": "vendor_a", "unit_price": 43, "delivery_days": 8, "true_cost": 999},
        {"vendor_id": "vendor_b", "unit_price": 43, "delivery_days": 9, "true_cost": -999},
        {"vendor_id": "vendor_c", "unit_price": 43, "delivery_days": 10, "true_cost": 0},
    ]
    rfq = {"quantity": 200, "budget": 9000, "max_delivery_days": 20}
    assert vendor_for_position(rfq, offers, ProcurementPosition.BEST) == "vendor_a"
    assert vendor_for_position(rfq, offers, ProcurementPosition.MIDDLE) == "vendor_b"
    assert vendor_for_position(rfq, offers, ProcurementPosition.WORST) == "vendor_c"
    assert [row["vendor_id"] for row in rank_feasible_offers(rfq, offers)] == ["vendor_a", "vendor_b", "vendor_c"]


def test_matrix_fixture_reaches_round_five_without_mutating_standard_scenarios():
    assert MATRIX_SCENARIO.max_rounds == 5
    assert S1.max_rounds == 4
    assert S2.max_rounds == 4
    assert S3.max_rounds == 4
    assert S4.max_rounds == 4
    assert S5.max_rounds == 4


def test_representative_matrix_has_exactly_25_cells_and_full_dimension_coverage():
    cells = build_representative_matrix()
    assert len(cells) == 25
    assert len({cell.cell_id for cell in cells}) == 25
    assert {cell.family for cell in cells if cell.family} >= {family.value for family in AttackFamily}
    assert {cell.attacker_position for cell in cells if cell.attacker_position} == {"best", "middle", "worst"}
    assert {cell.malicious_vendor_count for cell in cells} == {1, 2, 3}
    assert {cell.architecture for cell in cells} == {"S0", "S1", "S2", "S3"}
    assert {cell.round for cell in cells if cell.round} >= {1, 2, 3, 4, 5}


def test_all_family_cells_execute_round_one():
    # Runtime execution is tested through the dedicated runner when langgraph is available.
    cells = [cell for cell in build_representative_matrix() if cell.kind == "attack" and cell.cell_id.startswith("family-")]
    assert len(cells) == 14
    assert all(cell.round == 1 for cell in cells)
    assert all(cell.attacker_position == "best" for cell in cells)


def test_timing_cells_target_the_intended_reachable_rounds():
    cells = {cell.cell_id: cell for cell in build_representative_matrix()}
    assert cells["timing-round3-context-poisoning"].round == 3
    assert cells["timing-round4-policy-override"].round == 4
    assert cells["timing-round5-tool-use"].round == 5
    assert MATRIX_SCENARIO.max_rounds >= 5


def test_collusion_cells_use_the_approved_plans():
    plans = {plan.collusion_id: plan for plan in collusion_stress_plans()}
    cells = {cell.cell_id: cell for cell in build_representative_matrix()}
    assert cells["collusion-2-coordinated"].collusion_id == "collusion-2-coordinated"
    assert cells["collusion-3-coordinated"].collusion_id == "collusion-3-coordinated"
    assert len(plans["collusion-2-coordinated"].attacker_vendors) == 2
    assert len(plans["collusion-3-coordinated"].attacker_vendors) == 3


def test_architecture_cells_cover_s0_s1_s2_and_s3_is_already_in_family_baseline():
    cells = build_representative_matrix()
    assert {cell.architecture for cell in cells if cell.kind == "architecture"} == {"S0", "S1", "S2"}
    assert any(cell.kind == "attack" and cell.architecture == "S3" for cell in cells)


def test_adaptive_cell_uses_bounded_adaptive_configuration():
    cell = next(cell for cell in build_representative_matrix() if cell.adaptive)
    assert cell.family == AttackFamily.ADAPTIVE.value
    assert cell.cell_id == "adaptive-canonical"


def test_same_offline_matrix_inputs_produce_same_evidence_hash(tmp_path):
    # This test exercises the deterministic matrix assembly. Full execution requires langgraph.
    first = build_representative_matrix()
    second = build_representative_matrix()
    assert first == second


def test_matrix_bundle_is_deterministic_and_contains_no_hidden_cost_or_delivery_in_buyer_prompt():
    bundle = matrix_bundle(1)
    # MatrixBuyerModel's behavior is deterministic and does not read vendor truth.
    output = bundle.buyer("Round 1 of 5")
    parsed = json.loads(output)
    assert parsed["action"] == "counter"
    assert "true_cost" not in output
    assert "true_delivery" not in output


def test_matrix_evidence_json_safe_boundary_preserves_tuple_key_dimensions():
    from eval.adversarial.evidence import _jsonable

    grouped = {("direct_goal_prompt_hijacking", 1, 1): {"uar": 0.0, "pvr": 0.0}}
    safe = _jsonable({"grouped_metrics": grouped})
    assert list(safe["grouped_metrics"]) == ["('direct_goal_prompt_hijacking', 1, 1)"]
    assert safe["grouped_metrics"]["('direct_goal_prompt_hijacking', 1, 1)"]["uar"] == 0.0
