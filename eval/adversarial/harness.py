"""Adversarial evaluation harness.

The legacy economic matched cell compares honest and dishonest vendor dispositions.
Issue #9 additionally provides a true attack counterfactual: control emits the
commercial messages once; treatment replays those exact messages and adds only the
target vendor's deterministic attack payload. A replayed control provides an A/A'
observation. No security-success scoring is performed here.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from eval.adversarial.orchestrator import run_negotiation
from eval.adversarial.scoring import run_metrics, welfare, classify_pair
from eval.adversarial.outcomes import (
    build_adversarial_outcomes,
    build_aa_control_result,
    classify_ab_against_aa,
)
from guards.procurement import ProcurementPosition, rank_feasible_offers
from guards.price_guard import PRICING_RELEVANT_FLAGS
from eval.adversarial.attacks import AttackTrajectory, CollusionPlan
from eval.adversarial.scenarios import Scenario
from eval.adversarial.scripted_vendor import VendorMessage
from eval.adversarial.vendor_agent import AdversarialVendor
from eval.adversarial import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class Bundle:
    buyer: object
    interpreter: object
    vendor: object
    attacker: object = None
    fingerprint: dict = field(default_factory=lambda: {"deterministic": True})


def build_vendors(
    scenario: Scenario,
    dishonest_ids,
    bundle: Bundle,
    *,
    raw_vendor_messages: bool = False,
):
    dishonest_ids = set(dishonest_ids)

    if raw_vendor_messages and bundle.attacker is None and dishonest_ids:
        raise ValueError("raw vendor mode requires an attacker model")

    return [
        AdversarialVendor(
            vendor_id=t.vendor_id,
            reservation_price=t.true_cost,
            opening_price=t.opening_price,
            delivery_days=t.true_delivery,
            disposition="dishonest" if t.vendor_id in dishonest_ids else "honest",
            model=(
                bundle.attacker
                if raw_vendor_messages and t.vendor_id in dishonest_ids
                else bundle.vendor
            ),
        )
        for t in scenario.vendors
    ]


def run_once(
    scenario: Scenario,
    dishonest_ids,
    bundle: Bundle,
    *,
    raw_vendor_messages: bool = False,
    vendor_message_overrides: dict[tuple[int, str], str] | None = None,
    vendor_message_factory=None,
) -> dict:
    vendors = build_vendors(
        scenario,
        set(dishonest_ids),
        bundle,
        raw_vendor_messages=raw_vendor_messages,
    )
    return run_negotiation(
        scenario,
        vendors,
        bundle.buyer,
        bundle.interpreter,
        raw_vendor_messages=raw_vendor_messages,
        vendor_message_overrides=vendor_message_overrides,
        vendor_message_factory=vendor_message_factory,
    )


def _vendor_messages(record: dict) -> dict[tuple[int, str], str]:
    """Extract the exact commercial messages emitted by the control run."""
    return {
        (event["round"], event["vendor_id"]): event["raw_text"]
        for event in record["log"]
        if event["event"] == "vendor_message"
    }


def _attack_messages(
    control_messages: dict[tuple[int, str], str],
    target_vendor: str,
    trajectory: AttackTrajectory,
) -> tuple[dict[tuple[int, str], str], list[dict]]:
    """Replay control messages, adding only the target's scripted payloads."""
    if trajectory.attacker_vendor != target_vendor:
        raise ValueError("attack trajectory is bound to a different target vendor")

    treatment = dict(control_messages)
    attack_rows = []
    for (round_index, vendor_id), commercial in control_messages.items():
        if vendor_id != target_vendor:
            continue
        round_number = round_index + 1
        payload = trajectory.payload_for_round(round_number)
        treatment[(round_index, vendor_id)] = VendorMessage(
            commercial_content=commercial,
            attack_payload=payload,
        ).text
        step = trajectory.step_for_round(round_number)
        attack_rows.append({
            "round": round_number,
            "attacker_vendor": vendor_id,
            "attack_family": None if step is None else step.family.value,
            "attack_style": None if step is None else step.style.value,
            "attack_payload": payload,
            "control_message": commercial,
            "treatment_message": treatment[(round_index, vendor_id)],
        })
    return treatment, attack_rows


def _initial_ranked_control_offers(record: dict, scenario: Scenario) -> list[dict]:
    offers = [
        event["offer"]
        for event in record.get("log", [])
        if event.get("event") == "offer" and event.get("round") == 0
        and event["offer"].get("extraction_flag") not in PRICING_RELEVANT_FLAGS
    ]
    ranked = rank_feasible_offers(
        {
            "budget": scenario.budget,
            "quantity": scenario.quantity,
            "max_delivery_days": scenario.max_delivery_days,
        },
        offers,
    )
    return [offer for offer in ranked if float(offer["unit_price"]) >= float(scenario.absolute_floor)]


def _validate_attacker_position(
    control: dict,
    scenario: Scenario,
    target_vendor: str,
    attacker_position: str,
) -> None:
    try:
        requested = ProcurementPosition(attacker_position)
    except ValueError as exc:
        raise ValueError(f"unknown attacker_position: {attacker_position!r}") from exc
    ranked = _initial_ranked_control_offers(control, scenario)
    index = {
        ProcurementPosition.BEST: 0,
        ProcurementPosition.MIDDLE: 1,
        ProcurementPosition.WORST: 2,
    }[requested]
    if len(ranked) <= index:
        raise ValueError(
            f"cannot establish {requested.value} attacker position with {len(ranked)} feasible first-round offers"
        )
    actual = ranked[index]["vendor_id"]
    if actual != target_vendor:
        raise ValueError(
            f"attacker {target_vendor!r} is not the requested {requested.value} position; "
            f"deterministic position is {actual!r}"
        )


def _collusion_attack_messages(
    control_messages: dict[tuple[int, str], str],
    plan: CollusionPlan,
) -> tuple[dict[tuple[int, str], str], list[dict]]:
    """Replay control messages and add each configured vendor's deterministic payload."""
    treatment = dict(control_messages)
    attack_rows: list[dict] = []
    for trajectory in plan.trajectories:
        attacker_vendor = trajectory.attacker_vendor
        for (round_index, vendor_id), commercial in control_messages.items():
            if vendor_id != attacker_vendor:
                continue
            round_number = round_index + 1
            payload = trajectory.payload_for_round(round_number)
            if payload is None:
                continue
            step = trajectory.step_for_round(round_number)
            treatment[(round_index, vendor_id)] = VendorMessage(
                commercial_content=commercial,
                attack_payload=payload,
            ).text
            attack_rows.append({
                "round": round_number,
                "attacker_vendor": vendor_id,
                "attack_family": None if step is None else step.family.value,
                "attack_style": None if step is None else step.style.value,
                "attack_payload": payload,
                "control_message": commercial,
                "treatment_message": treatment[(round_index, vendor_id)],
            })
    return treatment, sorted(attack_rows, key=lambda row: (row["round"], row["attacker_vendor"]))


def run_collusion_counterfactual(
    scenario: Scenario,
    plan: CollusionPlan,
    make_bundle,
    *,
    seed: int = 0,
) -> dict:
    """Run a deterministic multi-malicious matched attack counterfactual."""
    vendor_ids = {vendor.vendor_id for vendor in scenario.vendors}
    unknown = set(plan.attacker_vendors) - vendor_ids
    if unknown:
        raise ValueError(f"unknown colluding vendor(s): {sorted(unknown)}")

    dishonest_ids = set(plan.attacker_vendors)
    bundle = make_bundle(seed)
    control = run_once(scenario, dishonest_ids, bundle)
    control_messages = _vendor_messages(control)
    expected_rounds = set(range(scenario.max_rounds))
    observed_rounds = {round_index for round_index, _ in control_messages}
    if observed_rounds != expected_rounds:
        raise ValueError(
            "collusion execution requires the control run to emit commercial "
            "messages for every configured round"
        )

    treatment_messages, attack_rows = _collusion_attack_messages(control_messages, plan)
    treatment = run_once(
        scenario,
        dishonest_ids,
        bundle,
        vendor_message_overrides=treatment_messages,
    )

    null_bundle = make_bundle(seed + 10_000)
    null = run_once(
        scenario,
        dishonest_ids,
        null_bundle,
        vendor_message_overrides=control_messages,
    )

    return {
        "scenario": scenario.scenario_id,
        "attacker_vendors": list(plan.attacker_vendors),
        "number_of_malicious_vendors": len(plan.attacker_vendors),
        "collusion_id": plan.collusion_id,
        "control": control,
        "treatment": treatment,
        "null": null,
        "attack": {
            "collusion_id": plan.collusion_id,
            "attacker_vendors": list(plan.attacker_vendors),
            "messages": attack_rows,
        },
        "outcomes": build_adversarial_outcomes(
            scenario=scenario,
            target_vendor=plan.attacker_vendors[0],
            attack_rows=attack_rows,
            control=control,
            treatment=treatment,
            number_of_malicious_vendors=len(plan.attacker_vendors),
            scenario_seed=seed,
            run_identifier=f"collusion:{plan.collusion_id}:{scenario.scenario_id}:{seed}",
            collusion_id=plan.collusion_id,
        ),
    }



def run_attack_counterfactual(
    scenario: Scenario,
    target_vendor: str,
    trajectory: AttackTrajectory,
    make_bundle,
    *,
    dishonest_ids=(),
    seed: int = 0,
    attacker_position: str | None = None,
    aa_repetitions: int = 3,
) -> dict:
    """Run a matched control/treatment pair with commercial text held fixed.

    Control emits the commercial messages once. Treatment replays those exact
    messages and adds only the target vendor's deterministic attack payloads.
    A separate replayed-control run provides an A/A' noise observation without
    adding attack logic or security scoring.
    """
    vendor_ids = {vendor.vendor_id for vendor in scenario.vendors}
    if target_vendor not in vendor_ids:
        raise ValueError(f"unknown target_vendor {target_vendor!r}")
    if trajectory.attacker_vendor != target_vendor:
        raise ValueError("attack trajectory is bound to a different target vendor")

    dishonest_ids = set(dishonest_ids)
    dishonest_ids.discard(target_vendor)
    bundle = make_bundle(seed)

    control = run_once(scenario, dishonest_ids, bundle)
    control_messages = _vendor_messages(control)
    # A normal negotiation may legitimately terminate early (a deal closes or the
    # buyer walks away) before the configured maximum number of rounds. Treatment
    # and A/A' replay only the commercial messages the control run actually
    # emitted; later attack steps are skipped when the matching control message
    # never exists, and missing later commercial messages are never fabricated.
    # Only require that the control produced at least one commercial message to
    # match against.
    if not control_messages:
        raise ValueError(
            "matched attack execution requires the control run to emit at least "
            "one commercial message"
        )

    if attacker_position is not None:
        _validate_attacker_position(control, scenario, target_vendor, attacker_position)

    treatment_messages, attack_rows = _attack_messages(
        control_messages, target_vendor, trajectory
    )
    treatment = run_once(
        scenario,
        dishonest_ids,
        bundle,
        vendor_message_overrides=treatment_messages,
    )

    if aa_repetitions < 1:
        raise ValueError("aa_repetitions must be >= 1")
    aa_runs = [
        run_once(
            scenario,
            dishonest_ids,
            make_bundle(seed + 10_000 + i),
            vendor_message_overrides=control_messages,
        )
        for i in range(aa_repetitions)
    ]
    aa_control = build_aa_control_result(control, aa_runs, control_messages)
    null = aa_runs[0]

    return {
        "scenario": scenario.scenario_id,
        "target_vendor": target_vendor,
        "control": control,
        "treatment": treatment,
        "null": null,
        "aa_control": {
            "repetitions": aa_control.repetitions,
            "pairwise_differences": aa_control.pairwise_differences,
            "benign_variability_summary": aa_control.benign_variability_summary,
            "commercial_messages": aa_control.commercial_messages,
        },
        "aa_comparison": classify_ab_against_aa(control, treatment, aa_control),
        "attack": {
            "attacker_vendor": target_vendor,
            "trajectory_rounds": [step.target_round for step in trajectory.steps],
            "messages": attack_rows,
        },
        "outcomes": build_adversarial_outcomes(
            scenario=scenario,
            target_vendor=target_vendor,
            attack_rows=attack_rows,
            control=control,
            treatment=treatment,
            number_of_malicious_vendors=len(dishonest_ids) + 1,
            scenario_seed=seed,
            run_identifier=f"cf:{scenario.scenario_id}:{target_vendor}:{seed}",
            attacker_position=attacker_position,
            aa_control=aa_control,
        ),
    }


def choose_dishonest(scenario: Scenario, n: int, seed: int):
    ids = [v.vendor_id for v in scenario.vendors]
    return set(random.Random(seed).sample(ids, n))


def run_cell(scenario, dishonest_ids, target_id, make_bundle, k, seed0=0, *, raw_vendor_messages: bool = False, ):
    """K matched pairs (flip target only) + K A/A' null pairs, for one target."""
    dishonest_B = set(dishonest_ids)
    dishonest_A = dishonest_B - {target_id}  # target honest in A

    a_welfares, b_welfares, per_pair, records = [], [], [], []
    for i in range(k):
        bundle = make_bundle(seed0 + i)
        rec_a = run_once(
            scenario,
            dishonest_A,
            bundle,
            raw_vendor_messages=raw_vendor_messages,
        )
        rec_b = run_once(
            scenario,
            dishonest_B,
            bundle,
            raw_vendor_messages=raw_vendor_messages,
        )
        m_a = run_metrics(rec_a, scenario, target_id)
        m_b = run_metrics(rec_b, scenario, target_id)
        a_welfares.append(m_a["welfare"])
        b_welfares.append(m_b["welfare"])
        per_pair.append((m_a, m_b))
        records.append({"i": i, "A": rec_a, "B": rec_b, "metrics_A": m_a, "metrics_B": m_b, "bundle": bundle.fingerprint})

    null_deltas = []
    for i in range(k):
        b1 = make_bundle(seed0 + i)
        b2 = make_bundle(seed0 + 10_000 + i)
        w1 = welfare(
            run_once(
                scenario,
                dishonest_A,
                b1,
                raw_vendor_messages=raw_vendor_messages,
            )["order"],
            scenario,
        )
        w2 = welfare(
            run_once(
                scenario,
                dishonest_A,
                b2,
                raw_vendor_messages=raw_vendor_messages,
            )["order"],
            scenario,
        )
        null_deltas.append(w1 - w2)

    band = stats.null_band(null_deltas)
    effect = stats.paired_effect(a_welfares, b_welfares)
    classified = [classify_pair(m_a, m_b, a - b, band["high"]) for (m_a, m_b), a, b in zip(per_pair, a_welfares, b_welfares)]

    return {
        "scenario": scenario.scenario_id,
        "scenario_version": scenario.version,
        "dishonest_ids": sorted(dishonest_B),
        "target_vendor": target_id,
        "k": k,
        "paired_effect": {kk: vv for kk, vv in effect.items() if kk != "deltas"},
        "null_band": band,
        # --- economic / utility rates (never security signals) ---
        "welfare_loss_rate": stats.rate(c["welfare_loss_beyond_noise"] for c in classified),
        "welfare_reducing_win_rate": stats.rate(c["welfare_reducing_win"] for c in classified),
        "fair_win_rate": stats.rate(c["fair_win"] for c in classified),
        "no_welfare_loss_rate": stats.rate(c["no_welfare_loss"] for c in classified),
        "business_claim_inaccurate_rate_B": stats.rate(
            m_b["business_claim_inaccurate"] for _, m_b in per_pair
        ),
        "quoted_below_true_cost_rate_B": stats.rate(
            m_b["quoted_below_true_cost"] for _, m_b in per_pair
        ),
        "delivery_claim_mismatch_rate_B": stats.rate(
            m_b["delivery_claim_mismatch"] for _, m_b in per_pair
        ),
        "won_rate_B": stats.rate(
            m_b["won"] and m_b["winner"] == target_id
            for _, m_b in per_pair
        ),
        "absolute_harm_rate_A": stats.rate(
            m_a["absolute_harm"] for m_a, _ in per_pair
        ),
        "absolute_harm_rate_B": stats.rate(
            m_b["absolute_harm"] for _, m_b in per_pair
        ),
        "unnecessary_refusal_rate_B": stats.rate(
            m_b["unnecessary_refusal"] for _, m_b in per_pair
        ),
        # --- security rates (agent-security threat model only) ---
        "action_integrity_failure_rate": stats.rate(
            c["action_integrity_failure"] for c in classified
        ),
        "policy_violation_rate": stats.rate(
            c["policy_violation"] for c in classified
        ),
        "security_failure_rate": stats.rate(
            c["security_failure"] for c in classified
        ),
        "ground_truth": [{"vendor_id": t.vendor_id, "true_cost": t.true_cost, "true_delivery": t.true_delivery} for t in scenario.vendors],
        "records": records,
    }


def run_baseline(scenario, make_bundle, k, seed0=0, *, raw_vendor_messages: bool = False, ):
    """0-dishonest control: honest-market welfare reference (no target/pair)."""
    welfares = []
    records = []
    for i in range(k):
        bundle = make_bundle(seed0 + i)
        rec = run_once( scenario, set(), bundle, raw_vendor_messages=raw_vendor_messages, )
        welfares.append(welfare(rec["order"], scenario))
        records.append({"i": i, "run": rec})
    from statistics import mean, median

    return {
        "scenario": scenario.scenario_id,
        "composition": "0_dishonest",
        "k": k,
        "mean_welfare": mean(welfares) if welfares else float("nan"),
        "median_welfare": median(welfares) if welfares else float("nan"),
        "records": records,
    }


def run_matrix(scenarios, compositions, make_bundle, k, seed0=0, id_seed=0, write=True, tag="run", *, raw_vendor_messages: bool = False, ):
    cells = []
    for scenario in scenarios:
        for comp in compositions:
            if comp == 0:
                cells.append(
                    run_baseline(
                        scenario,
                        make_bundle,
                        k,
                        seed0=seed0,
                        raw_vendor_messages=raw_vendor_messages,
                    )
                )
                continue
            dishonest = choose_dishonest(scenario, comp, id_seed)
            for target in sorted(dishonest):
                cells.append(
                    run_cell(
                        scenario,
                        dishonest,
                        target,
                        make_bundle,
                        k,
                        seed0=seed0,
                        raw_vendor_messages=raw_vendor_messages,
                    )
                )
    report = {"tag": tag, "compositions": compositions, "k": k, "cells": cells}
    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"adversarial_{tag}.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        report["written_to"] = str(out)
    return report
