import json

from eval.adversarial.capability import classify_level, measure_capability


class CorrectProbeModel:
    def __call__(self, prompt):
        if "exactly these keys" in prompt:
            return json.dumps({"a": 7, "b": "ok"})
        if "highest per-unit" in prompt:
            return json.dumps({"max_unit_price": 50})
        if "stating a delivery time" in prompt:
            return json.dumps({"message": "We deliver in 9 days."})
        if "actual delivery in days" in prompt:
            return json.dumps({"delivery_days": 21})
        return "{}"


class WeakProbeModel:
    def __call__(self, prompt):
        return "I cannot produce JSON."


def test_capability_measurement_scores_a_correct_model_high():
    result = measure_capability(CorrectProbeModel())
    assert result["capability_index"] == 1.0
    assert all(v == 1.0 for v in result["probe_scores"].values())


def test_capability_measurement_scores_a_weak_model_low():
    result = measure_capability(WeakProbeModel())
    assert result["capability_index"] == 0.0


def test_levels_are_derived_from_measured_indices_not_names():
    assert classify_level(0.4, 0.9) == "weaker"
    assert classify_level(0.9, 0.4) == "stronger"
    assert classify_level(0.8, 0.8) == "comparable"
