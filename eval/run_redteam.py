"""Run the deterministic red-team evaluation and record results.

Usage: python -m eval.run_redteam
Writes a JSON report to eval/results/ and prints the two-part headline.
"""

import json
from pathlib import Path

from eval.harness import run

SEED = 1337
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    report = run(seed=SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"redteam_seed_{SEED}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"cases: {report['n_cases']}  attack payloads: {report['n_attack_payloads']}")
    print(f"Metric A (action reachability): {report['metric_a_action_reachability']}")
    print(f"Metric B (value corruption, defined set): {report['metric_b_value_corruption']}")
    print(f"known-limitation probe (delivery, unguarded): {report['metric_b_known_limitation_probe']}")
    print(report["headline"])
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
