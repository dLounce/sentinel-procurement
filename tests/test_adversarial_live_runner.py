import json
import pytest
from eval.adversarial.harness import Bundle
from eval.adversarial.live_runner import (
    _estimate_cost_usd,
    _load_spend,
    run_live,
)
from eval.adversarial.offline import (
    OfflineBuyerModel,
    OfflineInterpreterModel,
    OfflineVendorModel,
)
from eval.adversarial.scenarios import S1


PRICING = {
    "buyer": {
        "input_per_million": 1.0,
        "output_per_million": 1.0,
    },
    "interpreter": {
        "input_per_million": 1.0,
        "output_per_million": 1.0,
    },
    "vendor": {
        "input_per_million": 1.0,
        "output_per_million": 1.0,
    },
}

class UsageModel:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, prompt):
        return self.inner(prompt)

    def fingerprint(self):
        return {
            "provider": "test",
            "model": "offline",
            "total_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
        }

def bundle(seed):
    return Bundle(
        buyer=UsageModel(OfflineBuyerModel()),
        interpreter=UsageModel(OfflineInterpreterModel()),
        vendor=UsageModel(OfflineVendorModel()),
    )

def test_estimate_cost_usd_uses_per_role_usage():
    fingerprints = {
        "buyer": {
            "total_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        },
        "interpreter": {
            "total_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        },
        "vendor": {
            "total_usage": {
                "prompt_tokens": 300,
                "completion_tokens": 150,
                "total_tokens": 450,
            }
        },
    }

    assert _estimate_cost_usd(fingerprints, PRICING) == 0.0009


def test_load_spend_sums_completed_trials_only(tmp_path):
    path = tmp_path / "run.jsonl"

    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trial_id": "a",
                        "status": "completed",
                        "estimated_cost_usd": 0.10,
                    }
                ),
                json.dumps(
                    {
                        "trial_id": "b",
                        "status": "failed",
                        "estimated_cost_usd": 0.50,
                    }
                ),
                json.dumps(
                    {
                        "trial_id": "c",
                        "status": "completed",
                        "estimated_cost_usd": 0.20,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _load_spend(path) == pytest.approx(0.30)


def test_budget_stops_before_next_trial(tmp_path):
    result = run_live(
        [S1],
        [0],
        bundle,
        k=3,
        run_id="budget_stop",
        max_trials=3,
        max_usd=0.0,
        pricing=PRICING,
        results_dir=tmp_path,
    )

    assert result["executed_this_call"] == 0
    assert result["estimated_spend_usd"] == 0.0
    assert result["budget_exhausted"] is True


def test_resume_reloads_persisted_spend_and_skips_completed(tmp_path):
    run_id = "resume_budget"

    first = run_live(
        [S1],
        [0],
        bundle,
        k=1,
        run_id=run_id,
        max_trials=1,
        max_usd=10.0,
        pricing=PRICING,
        results_dir=tmp_path,
    )

    assert first["executed_this_call"] == 1
    assert first["estimated_spend_usd"] > 0.0

    second = run_live(
        [S1],
        [0],
        bundle,
        k=2,
        run_id=run_id,
        max_trials=1,
        max_usd=first["estimated_spend_usd"] + 0.000001,
        pricing=PRICING,
        results_dir=tmp_path,
    )

    assert second["skipped_already_completed"] == 1
    assert second["executed_this_call"] == 1
    assert second["estimated_spend_usd"] > first["estimated_spend_usd"]

def test_resume_continues_after_multiple_completed_trials(tmp_path):
    run_id = "resume_multiple"

    first = run_live(
        [S1],
        [0],
        bundle,
        k=3,
        run_id=run_id,
        max_trials=2,
        max_usd=10.0,
        pricing=PRICING,
        results_dir=tmp_path,
    )

    assert first["executed_this_call"] == 2

    second = run_live(
        [S1],
        [0],
        bundle,
        k=3,
        run_id=run_id,
        max_trials=1,
        max_usd=10.0,
        pricing=PRICING,
        results_dir=tmp_path,
    )

    assert second["skipped_already_completed"] == 2
    assert second["executed_this_call"] == 1
