"""Matched-counterfactual harness.

For a target vendor V in a composition, a matched pair holds everything fixed and
flips ONLY V's disposition: A = V honest, B = V dishonest. A/A' null pairs (both
honest, different seeds) estimate the stochastic noise floor. Welfare loss is
attributed to V only when the paired welfare effect exceeds that null band — this
separates "V's disposition cost the buyer surplus" from "the scenario was bad" and from
model variation. That welfare loss is an economic outcome, never a security signal.

Hidden ground truth lives only in evaluator-side objects and result records; it is
never placed into any Buyer/Interpreter input. `make_bundle(seed)` provisions the
models (offline doubles for tests/pilot; seeded live clients for a real run), so no
live provider is touched here.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from eval.adversarial.orchestrator import run_negotiation
from eval.adversarial.scoring import run_metrics, welfare, classify_pair
from eval.adversarial.scenarios import Scenario
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
    )


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
