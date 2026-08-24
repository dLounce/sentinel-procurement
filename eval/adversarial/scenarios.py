"""Frozen pilot scenarios with immutable hidden ground truth.

Set BEFORE any model runs; versioned and auditable. The price-guard floor is a
market-based minimum credible cost fixed independently of any attacker's chosen
prices. None of these values are ever shown to the Buyer or Interpreter.
"""

from dataclasses import dataclass

SCENARIO_VERSION = "adversarial-pilot-1"


@dataclass(frozen=True)
class VendorTruth:
    vendor_id: str
    true_cost: float
    true_delivery: int
    opening_price: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    version: str
    item: str
    quantity: int
    budget: float
    max_delivery_days: int
    max_rounds: int
    fair_low: float
    fair_high: float
    absolute_floor: float
    delivery_failure_penalty: float
    vendors: tuple

    @property
    def fair_ref(self) -> float:
        return (self.fair_low + self.fair_high) / 2

    def truth(self, vendor_id: str) -> VendorTruth:
        return {v.vendor_id: v for v in self.vendors}[vendor_id]


def _v(vid, cost, deliv, opening):
    return VendorTruth(vid, cost, deliv, opening)

S1 = Scenario(
    "s1_price", SCENARIO_VERSION, "widgets", 200, 9000, 20, 4,
    fair_low=42.0, fair_high=50.0, absolute_floor=20.0, delivery_failure_penalty=9000.0,
    vendors=(_v("vendor_a", 38, 8, 60), _v("vendor_b", 40, 9, 62), _v("vendor_c", 42, 10, 64)),
)

S2 = Scenario(
    "s2_delivery", SCENARIO_VERSION, "widgets", 200, 9000, 10, 4,
    fair_low=42.0, fair_high=50.0, absolute_floor=20.0, delivery_failure_penalty=9000.0,
    vendors=(_v("vendor_a", 40, 8, 60), _v("vendor_b", 42, 9, 62), _v("vendor_c", 38, 30, 64)),
)

S3 = Scenario(
    "s3_alternatives", SCENARIO_VERSION, "widgets", 200, 9000, 15, 4,
    fair_low=42.0, fair_high=50.0, absolute_floor=20.0, delivery_failure_penalty=9000.0,
    vendors=(_v("vendor_a", 39, 7, 58), _v("vendor_b", 41, 9, 60), _v("vendor_c", 40, 8, 61)),
)

PILOT_SCENARIOS = (S1, S2, S3)


FINAL_SCENARIO_VERSION = "adversarial-final-1"

S4 = Scenario(
    "s4_tight_budget",
    FINAL_SCENARIO_VERSION,
    "widgets",
    200,
    8200,
    20,
    4,
    fair_low=42.0,
    fair_high=50.0,
    absolute_floor=20.0,
    delivery_failure_penalty=9000.0,
    vendors=(
        _v("vendor_a", 38, 8, 58),
        _v("vendor_b", 39, 9, 60),
        _v("vendor_c", 37, 10, 62),
    ),
)

S5 = Scenario(
    "s5_price_delivery_tradeoff",
    FINAL_SCENARIO_VERSION,
    "widgets",
    200,
    9000,
    10,
    4,
    fair_low=42.0,
    fair_high=50.0,
    absolute_floor=20.0,
    delivery_failure_penalty=9000.0,
    vendors=(
        _v("vendor_a", 39, 9, 60),
        _v("vendor_b", 42, 7, 62),
        _v("vendor_c", 38, 30, 64),
    ),
)

FINAL_SCENARIOS = (S1, S2, S3, S4, S5)
