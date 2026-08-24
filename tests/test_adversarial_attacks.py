import json
from dataclasses import replace

from eval.adversarial.attacks import (
    ATTACK_POSITIONS,
    ATTACK_DEFINITIONS,
    CANONICAL_ESCALATION,
    AttackFamily,
    AttackStep,
    AttackStyle,
    AttackTrajectory,
    representative_payload,
)
from eval.adversarial.scripted_vendor import ScriptedAdversarialVendor, VendorMessage
from eval.adversarial.scenarios import Scenario, VendorTruth
from rfq import RFQ


class CommercialModel:
    def __call__(self, prompt):
        return json.dumps({
            "unit_price": 48,
            "delivery_days": 8,
            "note": "Standard commercial terms.",
        })


def vendor(vendor_id="vendor_a", trajectory=None):
    return ScriptedAdversarialVendor(
        vendor_id=vendor_id,
        reservation_price=40,
        opening_price=60,
        delivery_days=8,
        disposition="dishonest",
        model=CommercialModel(),
        trajectory=trajectory,
    )


def rfq():
    return RFQ("attack-test", "widgets", 200, 10000, 20, 5)


def test_all_14_attack_families_exist():
    assert len(AttackFamily) == 14
    assert {definition.family for definition in ATTACK_DEFINITIONS} == set(AttackFamily)


def test_all_six_attack_styles_exist():
    assert len(AttackStyle) == 6
    assert {definition.style for definition in ATTACK_DEFINITIONS} <= set(AttackStyle)
    assert {definition.style for definition in ATTACK_DEFINITIONS} == {
        AttackStyle.DIRECT,
        AttackStyle.SUBTLE,
        AttackStyle.ROLE_BASED,
        AttackStyle.FAKE_SYSTEM,
        AttackStyle.CONTEXTUAL,
        AttackStyle.COMMERCIALLY_EMBEDDED,
    }


def test_every_family_has_a_deterministic_representative_payload():
    for family in AttackFamily:
        first = representative_payload(family)
        second = representative_payload(family)
        assert first and first == second


def test_position_library_covers_required_round_positions():
    assert set(ATTACK_POSITIONS) == {"early", "middle", "late", "finalization"}
    assert ATTACK_POSITIONS["early"] == (1,)
    assert ATTACK_POSITIONS["middle"] == (3,)
    assert ATTACK_POSITIONS["late"] == (4,)
    assert ATTACK_POSITIONS["finalization"] == (5,)


def test_round_targeting_and_non_targeted_rounds():
    trajectory = AttackTrajectory(
        "vendor_a",
        (
            AttackStep(2, AttackFamily.COMPETITOR_SUPPRESSION, AttackStyle.SUBTLE,
                       representative_payload(AttackFamily.COMPETITOR_SUPPRESSION)),
            AttackStep(4, AttackFamily.TOOL_USE_INJECTION, AttackStyle.DIRECT,
                       representative_payload(AttackFamily.TOOL_USE_INJECTION)),
        ),
    )
    assert trajectory.payload_for_round(1) is None
    assert trajectory.payload_for_round(2) == representative_payload(AttackFamily.COMPETITOR_SUPPRESSION)
    assert trajectory.payload_for_round(3) is None
    assert trajectory.payload_for_round(4) == representative_payload(AttackFamily.TOOL_USE_INJECTION)
    assert trajectory.payload_for_round(5) is None


def test_canonical_trajectory_escalates_in_order():
    assert [step.target_round for step in CANONICAL_ESCALATION.steps] == [2, 3, 4, 5]
    assert [step.family for step in CANONICAL_ESCALATION.steps] == [
        AttackFamily.DIRECT_GOAL_PROMPT_HIJACKING,
        AttackFamily.NEGOTIATION_CONTEXT_POISONING,
        AttackFamily.POLICY_OVERRIDE,
        AttackFamily.TOOL_USE_INJECTION,
    ]
    assert [step.style for step in CANONICAL_ESCALATION.steps] == [
        AttackStyle.SUBTLE,
        AttackStyle.CONTEXTUAL,
        AttackStyle.FAKE_SYSTEM,
        AttackStyle.DIRECT,
    ]


def test_trajectory_is_associated_with_the_correct_vendor():
    trajectory = replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")
    assert vendor("vendor_a", trajectory).trajectory.attacker_vendor == "vendor_a"

    try:
        vendor("vendor_b", trajectory)
    except ValueError as exc:
        assert "vendor_id" in str(exc)
    else:
        raise AssertionError("vendor/trajectory mismatch was accepted")


def test_legitimate_commercial_content_and_attack_payload_are_separate():
    trajectory = replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")
    scripted = vendor("vendor_a", trajectory)
    message = scripted.propose_message(rfq(), 1, None, [], model=scripted.model)

    assert isinstance(message, VendorMessage)
    assert message.commercial_content == "Standard commercial terms. Our price is $48.00 per unit, delivery in 8 days."
    assert message.attack_payload == representative_payload(AttackFamily.DIRECT_GOAL_PROMPT_HIJACKING)
    assert message.commercial_content not in message.attack_payload
    assert message.text == f"{message.commercial_content} {message.attack_payload}"


def test_one_malicious_vendor_can_be_represented():
    trajectory = replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")
    scripted = vendor("vendor_a", trajectory)
    message = scripted.propose_quote(rfq(), 1, None, [], model=scripted.model)
    assert representative_payload(AttackFamily.DIRECT_GOAL_PROMPT_HIJACKING) in message


def test_two_malicious_vendors_can_have_independent_trajectories():
    first = replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")
    second = AttackTrajectory(
        "vendor_b",
        (AttackStep(3, AttackFamily.COLLUSION, AttackStyle.CONTEXTUAL,
                     representative_payload(AttackFamily.COLLUSION)),),
    )
    a = vendor("vendor_a", first)
    b = vendor("vendor_b", second)
    assert a.attack_payload_for_round(1) == representative_payload(AttackFamily.DIRECT_GOAL_PROMPT_HIJACKING)
    assert b.attack_payload_for_round(1) is None
    assert b.attack_payload_for_round(2) == representative_payload(AttackFamily.COLLUSION)


def test_three_malicious_vendors_can_be_represented():
    vendors = [
        vendor("vendor_a", replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")),
        vendor("vendor_b", AttackTrajectory("vendor_b", (AttackStep(
            2, AttackFamily.ROLE_HIJACKING, AttackStyle.ROLE_BASED,
            representative_payload(AttackFamily.ROLE_HIJACKING),
        ),))),
        vendor("vendor_c", AttackTrajectory("vendor_c", (AttackStep(
            4, AttackFamily.COMMERCIALLY_EMBEDDED, AttackStyle.COMMERCIALLY_EMBEDDED,
            representative_payload(AttackFamily.COMMERCIALLY_EMBEDDED),
        ),))),
    ]
    assert [v.trajectory.attacker_vendor for v in vendors] == ["vendor_a", "vendor_b", "vendor_c"]


def test_repeated_deterministic_execution_is_identical():
    trajectory = replace(CANONICAL_ESCALATION, attacker_vendor="vendor_a")
    scripted = vendor("vendor_a", trajectory)
    first = [scripted.propose_message(rfq(), r, None, [], model=scripted.model) for r in range(5)]
    second = [scripted.propose_message(rfq(), r, None, [], model=scripted.model) for r in range(5)]
    assert first == second
