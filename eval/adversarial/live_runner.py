"""Resumable live adversarial-evaluation runner.

The runner treats one matched replication as the checkpoint unit:
    A = target vendor honest
    B = target vendor dishonest
    A/A' = null-noise comparison

Each completed replication is written immediately to JSONL so an interrupted
process can resume without losing completed work.

This module does not choose API keys, models, or budgets. The caller supplies
a ``make_bundle(seed)`` factory and an explicit USD budget ceiling.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from eval.adversarial.harness import (
    choose_dishonest,
    run_once,
    run_attack_counterfactual,
)
from eval.adversarial.attacks import AttackTrajectory
from eval.adversarial.orchestrator import HARD_MAX_ROUNDS

from eval.adversarial.scenarios import Scenario
from eval.adversarial.scoring import run_metrics, welfare
from eval.adversarial import stats

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

@dataclass(frozen=True)
class TrialSpec:
    """One resumable matched-replication job."""

    trial_id: str
    scenario_id: str
    composition: int
    dishonest_ids: tuple[str, ...]
    target_vendor: str | None
    replication_index: int
    seed: int
    raw_vendor_messages: bool = False


def make_run_id(prefix: str = "live") -> str:
    """Create a filesystem-safe unique run identifier."""

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _load_completed(path: Path) -> set[str]:
    """Return trial IDs already durably written to the results JSONL."""

    completed: set[str] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trial_id = record.get("trial_id")
            if trial_id and record.get("status") == "completed":
                completed.add(trial_id)

    return completed

def _load_spend(path: Path) -> float:
    """Sum completed trial costs already persisted in the JSONL."""
    total = 0.0

    if not path.exists():
        return total

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("status") != "completed":
                continue

            cost = record.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                total += float(cost)

    return total


def _append_record(path: Path, record: dict) -> None:
    """Append one result and flush it immediately."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _snapshot_fingerprint(bundle) -> dict:
    """Capture provider fingerprints/usage without requiring provider classes."""

    result = {}

    for role_name in ("buyer", "interpreter", "vendor", "attacker"):
        client = getattr(bundle, role_name, None)
        fingerprint = getattr(client, "fingerprint", None)
        if callable(fingerprint):
            try:
                result[role_name] = fingerprint()
            except Exception as exc:
                result[role_name] = {
                    "fingerprint_error": type(exc).__name__,
                }

    return result


def _usage_total(bundle) -> dict[str, int]:
    """Sum cumulative token usage exposed by provider clients."""

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for role_name in ("buyer", "interpreter", "vendor", "attacker"):
        client = getattr(bundle, role_name, None)
        fingerprint = getattr(client, "fingerprint", None)
        if not callable(fingerprint):
            continue

        try:
            metadata = fingerprint()
        except Exception:
            continue

        usage = metadata.get("total_usage")
        if not isinstance(usage, dict):
            continue

        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value

    return totals


def _estimate_cost_usd(fingerprints: dict, pricing: dict) -> float | None:
    """Estimate USD cost from per-role cumulative token usage.

    Pricing format:
        {
            "buyer": {"input_per_million": 0.40, "output_per_million": 1.60},
            "interpreter": {"input_per_million": 0.15, "output_per_million": 0.60},
            "vendor": {"input_per_million": 0.075, "output_per_million": 0.30},
        }

    Returns None when required usage/pricing is unavailable.
    """

    total = 0.0
    found = False

    for role, role_pricing in pricing.items():
        if not isinstance(role_pricing, dict):
            continue

        metadata = fingerprints.get(role)
        if not isinstance(metadata, dict):
            continue

        usage = metadata.get("total_usage")
        if not isinstance(usage, dict):
            continue

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if not isinstance(prompt_tokens, int) or not isinstance(
            completion_tokens, int
        ):
            continue

        input_price = role_pricing.get("input_per_million")
        output_price = role_pricing.get("output_per_million")

        if not isinstance(input_price, (int, float)) or not isinstance(
            output_price, (int, float)
        ):
            continue

        total += (prompt_tokens / 1_000_000) * float(input_price)
        total += (completion_tokens / 1_000_000) * float(output_price)
        found = True

    return total if found else None


def _run_matched_replication(
    scenario: Scenario,
    dishonest_ids: set[str],
    target_vendor: str,
    make_bundle: Callable[[int], object],
    replication_index: int,
    seed: int,
    *,
    raw_vendor_messages: bool,
) -> dict:
    """Execute and score one A/B matched replication plus A/A' null pair."""

    dishonest_a = set(dishonest_ids) - {target_vendor}

    bundle_ab = make_bundle(seed)

    print("  A started", flush=True)
    record_a = run_once(
        scenario,
        dishonest_a,
        bundle_ab,
        raw_vendor_messages=raw_vendor_messages,
    )
    print("  A completed", flush=True)
    metrics_a = run_metrics(record_a, scenario, target_vendor)

    print("  B started", flush=True)
    record_b = run_once(
        scenario,
        dishonest_ids,
        bundle_ab,
        raw_vendor_messages=raw_vendor_messages,
    )
    print("  B completed", flush=True)
    metrics_b = run_metrics(record_b, scenario, target_vendor)

    bundle_null_a = make_bundle(seed)
    bundle_null_b = make_bundle(seed + 10_000)
    print("  null A started", flush=True)
    welfare_null_a = welfare(
        run_once(
            scenario,
            dishonest_a,
            bundle_null_a,
            raw_vendor_messages=raw_vendor_messages,
        )["order"],
        scenario,
    )
    print("  null A completed", flush=True)
    print("  null B started", flush=True)
    welfare_null_b = welfare(
        run_once(
            scenario,
            dishonest_a,
            bundle_null_b,
            raw_vendor_messages=raw_vendor_messages,
        )["order"],
        scenario,
    )
    print("  null B completed", flush=True)
    null_delta = welfare_null_a - welfare_null_b
    band = stats.null_band([null_delta])

    delta = metrics_a["welfare"] - metrics_b["welfare"]

    return {
        "replication_index": replication_index,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "scenario_version": scenario.version,
        "composition": len(dishonest_ids),
        "dishonest_ids": sorted(dishonest_ids),
        "target_vendor": target_vendor,
        "raw_vendor_messages": raw_vendor_messages,
        "A": {
            "record": record_a,
            "metrics": metrics_a,
        },
        "B": {
            "record": record_b,
            "metrics": metrics_b,
        },
        "null": {
            "delta": null_delta,
            "band": band,
        },
        "paired_delta_welfare": delta,
        "provider_fingerprint_AB": _snapshot_fingerprint(bundle_ab),
        "provider_fingerprint_null_A": _snapshot_fingerprint(bundle_null_a),
        "provider_fingerprint_null_B": _snapshot_fingerprint(bundle_null_b),
        "provider_usage_AB": _usage_total(bundle_ab),
        "provider_usage_null_A": _usage_total(bundle_null_a),
        "provider_usage_null_B": _usage_total(bundle_null_b),
    }



def _jsonable(value):
    """Convert evaluator dataclasses/enums into JSON-compatible values."""

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _jsonable(asdict(value))
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _attack_trial_id(
    scenario: Scenario,
    trajectory: AttackTrajectory,
    replication_index: int,
    seed: int,
    *,
    aa_repetitions: int,
    target_vendor: str,
) -> str:
    """Build a stable attack-counterfactual trial identifier."""

    steps = ",".join(
        f"{step.target_round}:{step.family.value}"
        for step in trajectory.steps
    )
    return (
        f"{scenario.scenario_id}|attack_cf|target={target_vendor}|"
        f"vendor={trajectory.attacker_vendor}|steps={steps}|"
        f"rep={replication_index}|seed={seed}|aa={aa_repetitions}"
    )


def _captured_provider_cost(
    bundles: list[tuple[int, object]],
    pricing: dict,
) -> tuple[float | None, list[dict]]:
    """Estimate one trial's live cost across all bundles used by the counterfactual."""

    fingerprints = []
    total = 0.0
    found = False
    for seed, bundle in bundles:
        fingerprint = _snapshot_fingerprint(bundle)
        fingerprints.append({"seed": seed, "fingerprint": fingerprint})
        group_cost = _estimate_cost_usd(fingerprint, pricing)
        if group_cost is not None:
            total += group_cost
            found = True
    return (total if found else None), fingerprints


def run_live_attack_counterfactuals(
    scenarios: Iterable[Scenario],
    attack_trajectories: Iterable[AttackTrajectory],
    make_bundle: Callable[[int], object],
    *,
    k: int,
    run_id: str,
    max_trials: int,
    target_vendor: str,
    aa_repetitions: int = 2,
    max_usd: float | None = None,
    pricing: dict | None = None,
    seed0: int = 0,
    results_dir: Path | None = None,
) -> dict:
    """Run selected attack counterfactual trajectories with live models.

    This is a persistence bridge only: attack execution, A/A' controls, typed
    outcomes, and round limiting remain owned by the existing harness and
    orchestrator.
    """

    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if aa_repetitions < 1:
        raise ValueError("aa_repetitions must be >= 1")

    scenario_list = list(scenarios)
    trajectory_list = list(attack_trajectories)
    if not scenario_list:
        raise ValueError("at least one scenario is required")
    if not trajectory_list:
        raise ValueError("at least one attack trajectory is required")

    vendor_ids = {vendor.vendor_id for scenario in scenario_list for vendor in scenario.vendors}
    if target_vendor not in vendor_ids:
        raise ValueError(f"unknown target_vendor {target_vendor!r}")

    selected_families = {
        step.family.value
        for trajectory in trajectory_list
        for step in trajectory.steps
    }
    for trajectory in trajectory_list:
        if trajectory.attacker_vendor != target_vendor:
            raise ValueError("attack trajectory is bound to a different target vendor")

    root = results_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / f"{run_id}.jsonl"
    manifest_path = root / f"{run_id}.manifest.json"

    specs = []
    for scenario in scenario_list:
        for trajectory in trajectory_list:
            for replication_index in range(k):
                seed = seed0 + replication_index
                specs.append(
                    {
                        "trial_id": _attack_trial_id(
                            scenario,
                            trajectory,
                            replication_index,
                            seed,
                            aa_repetitions=aa_repetitions,
                            target_vendor=target_vendor,
                        ),
                        "scenario_id": scenario.scenario_id,
                        "scenario": scenario,
                        "trajectory": trajectory,
                        "replication_index": replication_index,
                        "seed": seed,
                    }
                )

    completed = _load_completed(jsonl_path)
    estimated_spend_usd = _load_spend(jsonl_path)

    manifest = {
        "run_id": run_id,
        "kind": "live_attack_counterfactual",
        "k": k,
        "aa_repetitions": aa_repetitions,
        "max_trials": max_trials,
        "max_rounds": HARD_MAX_ROUNDS,
        "target_vendor": target_vendor,
        "scenario_ids": [scenario.scenario_id for scenario in scenario_list],
        "attack_families": sorted(selected_families),
        "seed0": seed0,
        "planned_trials": len(specs),
        "completed_before_start": len(completed),
        "max_usd": max_usd,
        "estimated_spend_usd": estimated_spend_usd,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    executed = 0
    skipped = 0
    failed = 0

    for spec in specs:
        if spec["trial_id"] in completed:
            skipped += 1
            continue
        if executed >= max_trials:
            break
        if max_usd is not None and estimated_spend_usd >= max_usd:
            break

        started = time.time()
        captured_bundles: list[tuple[int, object]] = []

        def tracked_make_bundle(seed: int):
            bundle = make_bundle(seed)
            captured_bundles.append((seed, bundle))
            return bundle

        try:
            result = run_attack_counterfactual(
                spec["scenario"],
                target_vendor,
                spec["trajectory"],
                tracked_make_bundle,
                seed=spec["seed"],
                aa_repetitions=aa_repetitions,
            )

            outcome_list = []
            for outcome in result.get("outcomes", []):
                outcome_list.append(
                    replace(
                        outcome,
                        run_identifier=f"live:{spec['trial_id']}",
                    )
                )
            result["outcomes"] = outcome_list

            trial_cost_usd = None
            provider_fingerprints = []
            if pricing is not None:
                trial_cost_usd, provider_fingerprints = _captured_provider_cost(
                    captured_bundles,
                    pricing,
                )
            else:
                provider_fingerprints = [
                    {"seed": seed, "fingerprint": _snapshot_fingerprint(bundle)}
                    for seed, bundle in captured_bundles
                ]

            if trial_cost_usd is not None:
                estimated_spend_usd += trial_cost_usd

            trial_result = {
                "trial_id": spec["trial_id"],
                "status": "completed",
                "kind": "attack_counterfactual",
                "scenario_id": spec["scenario_id"],
                "target_vendor": target_vendor,
                "attacker_vendor": spec["trajectory"].attacker_vendor,
                "attack_family": [step.family.value for step in spec["trajectory"].steps],
                "attack_trajectory": _jsonable(spec["trajectory"]),
                "replication_index": spec["replication_index"],
                "seed": spec["seed"],
                "aa_repetitions": aa_repetitions,
                "max_rounds": HARD_MAX_ROUNDS,
                "result": _jsonable(result),
                "provider_fingerprints": provider_fingerprints,
                "estimated_cost_usd": trial_cost_usd,
                "cumulative_estimated_spend_usd": estimated_spend_usd,
                "elapsed_seconds": time.time() - started,
            }
            _append_record(jsonl_path, trial_result)
            completed.add(spec["trial_id"])
            executed += 1
        except Exception as exc:
            _append_record(
                jsonl_path,
                {
                    "trial_id": spec["trial_id"],
                    "status": "failed",
                    "kind": "attack_counterfactual",
                    "scenario_id": spec["scenario_id"],
                    "target_vendor": target_vendor,
                    "attack_family": [step.family.value for step in spec["trajectory"].steps],
                    "replication_index": spec["replication_index"],
                    "seed": spec["seed"],
                    "aa_repetitions": aa_repetitions,
                    "max_rounds": HARD_MAX_ROUNDS,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.time() - started,
                },
            )
            failed += 1
            break

    summary = {
        "run_id": run_id,
        "planned_trials": len(specs),
        "completed_records": len(completed),
        "executed_this_call": executed,
        "skipped_already_completed": skipped,
        "failed_this_call": failed,
        "results_file": str(jsonl_path),
        "manifest_file": str(manifest_path),
        "max_usd": max_usd,
        "estimated_spend_usd": estimated_spend_usd,
        "budget_exhausted": max_usd is not None and estimated_spend_usd >= max_usd,
    }
    manifest["last_run"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def build_trial_specs(
    scenarios: Iterable[Scenario],
    compositions: Iterable[int],
    *,
    k: int,
    seed0: int = 0,
    id_seed: int = 0,
    raw_vendor_messages: bool = False,
    target_vendor: str | None = None,
) -> list[TrialSpec]:
    """Build deterministic trial IDs so interrupted runs can resume safely."""

    specs: list[TrialSpec] = []

    for scenario in scenarios:
        for composition in compositions:
            if composition == 0:
                for i in range(k):
                    trial_id = (
                        f"{scenario.scenario_id}|0|baseline|rep={i}|"
                        f"seed={seed0 + i}|raw={int(raw_vendor_messages)}"
                    )
                    specs.append(
                        TrialSpec(
                            trial_id=trial_id,
                            scenario_id=scenario.scenario_id,
                            composition=0,
                            dishonest_ids=(),
                            target_vendor=None,
                            replication_index=i,
                            seed=seed0 + i,
                            raw_vendor_messages=raw_vendor_messages,
                        )
                    )
                continue

            if target_vendor is not None and composition == 1:
                if target_vendor not in {v.vendor_id for v in scenario.vendors}:
                    raise ValueError(
                        f"unknown target_vendor {target_vendor!r} for {scenario.scenario_id}"
                    )
                dishonest_ids = {target_vendor}
            else:
                dishonest_ids = choose_dishonest(scenario, composition, id_seed)

            for target_vendor in sorted(dishonest_ids):
                for i in range(k):
                    trial_id = (
                        f"{scenario.scenario_id}|{composition}|"
                        f"{','.join(sorted(dishonest_ids))}|target={target_vendor}|"
                        f"rep={i}|seed={seed0 + i}|raw={int(raw_vendor_messages)}"
                    )
                    specs.append(
                        TrialSpec(
                            trial_id=trial_id,
                            scenario_id=scenario.scenario_id,
                            composition=composition,
                            dishonest_ids=tuple(sorted(dishonest_ids)),
                            target_vendor=target_vendor,
                            replication_index=i,
                            seed=seed0 + i,
                            raw_vendor_messages=raw_vendor_messages,
                        )
                    )

    return specs


def run_live(
    scenarios: Iterable[Scenario],
    compositions: Iterable[int],
    make_bundle: Callable[[int], object],
    *,
    k: int,
    run_id: str,
    max_trials: int,
    max_usd: float | None = None,
    pricing: dict | None = None,
    raw_vendor_messages: bool = False,
    seed0: int = 0,
    id_seed: int = 0,
    target_vendor: str | None = None,
    results_dir: Path | None = None,
) -> dict:
    """Run/resume a live evaluation with one-record-at-a-time persistence.

    The runner never retries a completed trial ID. A failed trial is recorded
    as ``status=failed`` and the runner stops so the failure can be inspected
    rather than silently consuming more credits.
    """

    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    root = results_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)

    jsonl_path = root / f"{run_id}.jsonl"
    manifest_path = root / f"{run_id}.manifest.json"

    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    specs = build_trial_specs(
        scenarios_by_id.values(),
        compositions,
        k=k,
        seed0=seed0,
        id_seed=id_seed,
        raw_vendor_messages=raw_vendor_messages,
        target_vendor=target_vendor,
    )

    completed = _load_completed(jsonl_path)

    estimated_spend_usd = _load_spend(jsonl_path)

    manifest = {
        "run_id": run_id,
        "k": k,
        "max_trials": max_trials,
        "raw_vendor_messages": raw_vendor_messages,
        "seed0": seed0,
        "id_seed": id_seed,
        "planned_trials": len(specs),
        "completed_before_start": len(completed),
        "max_usd": max_usd,
        "estimated_spend_usd": estimated_spend_usd,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    executed = 0
    skipped = 0
    failed = 0

    for spec in specs:
        if spec.trial_id in completed:
            skipped += 1
            continue

        if executed >= max_trials:
            break

        if max_usd is not None and estimated_spend_usd >= max_usd:
            break

        scenario = scenarios_by_id[spec.scenario_id]

        print(
            f"starting trial {spec.replication_index + 1}/{k} "
            f"| {spec.scenario_id} "
            f"| target={spec.target_vendor} "
            f"| raw={int(spec.raw_vendor_messages)}",
            flush=True,
        )

        started = time.time()

        try:
            if spec.composition == 0:
                bundle = make_bundle(spec.seed)
                record = run_once(
                    scenario,
                    set(),
                    bundle,
                    raw_vendor_messages=raw_vendor_messages,
                )
                trial_result = {
                    "trial_id": spec.trial_id,
                    "status": "completed",
                    "kind": "baseline",
                    "scenario_id": spec.scenario_id,
                    "replication_index": spec.replication_index,
                    "seed": spec.seed,
                    "record": record,
                    "welfare": welfare(record["order"], scenario),
                    "provider_fingerprint": _snapshot_fingerprint(bundle),
                    "provider_usage": _usage_total(bundle),
                    "elapsed_seconds": time.time() - started,
                }

                trial_cost_usd = None

                if pricing is not None:
                    trial_cost_usd = _estimate_cost_usd(
                    _snapshot_fingerprint(bundle),
                    pricing,
                )

                if trial_cost_usd is not None:
                    estimated_spend_usd += trial_cost_usd

                trial_result["estimated_cost_usd"] = trial_cost_usd
                trial_result["cumulative_estimated_spend_usd"] = estimated_spend_usd
            else:
                trial_result = _run_matched_replication(
                    scenario,
                    set(spec.dishonest_ids),
                    spec.target_vendor,
                    make_bundle,
                    spec.replication_index,
                    spec.seed,
                    raw_vendor_messages=raw_vendor_messages,
                )
                trial_result.update(
                    {
                        "trial_id": spec.trial_id,
                        "status": "completed",
                        "elapsed_seconds": time.time() - started,
                    }
                )

                trial_cost_usd = None

                if pricing is not None:
                    trial_cost_usd = 0.0
                    cost_found = False

                    for fingerprint_group in (
                    trial_result["provider_fingerprint_AB"],
                    trial_result["provider_fingerprint_null_A"],
                    trial_result["provider_fingerprint_null_B"],
                    ):
                        group_cost = _estimate_cost_usd(fingerprint_group, pricing)
                        if group_cost is not None:
                            trial_cost_usd += group_cost
                            cost_found = True

                    if not cost_found:
                        trial_cost_usd = None

                if trial_cost_usd is not None:
                    estimated_spend_usd += trial_cost_usd

                trial_result["estimated_cost_usd"] = trial_cost_usd
                trial_result["cumulative_estimated_spend_usd"] = estimated_spend_usd

            _append_record(jsonl_path, trial_result)
            print(
                f"trial {spec.replication_index + 1} checkpointed "
                f"| cost=${estimated_spend_usd:.4f} "
                f"| cumulative=${estimated_spend_usd:.4f}",
                flush=True,
            )
            completed.add(spec.trial_id)
            executed += 1

        except Exception as exc:
            failure = {
                "trial_id": spec.trial_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.time() - started,
            }
            _append_record(jsonl_path, failure)
            failed += 1
            break

    summary = {
        "run_id": run_id,
        "planned_trials": len(specs),
        "completed_records": len(completed),
        "executed_this_call": executed,
        "skipped_already_completed": skipped,
        "failed_this_call": failed,
        "results_file": str(jsonl_path),
        "manifest_file": str(manifest_path),
        "max_usd": max_usd,
        "estimated_spend_usd": estimated_spend_usd,
        "budget_exhausted": (
         max_usd is not None and estimated_spend_usd >= max_usd
        ),
    }

    manifest["last_run"] = summary
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return summary