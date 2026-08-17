from dataclasses import dataclass

from agents.vendor import Vendor
from eval.adversarial.prompts import vendor_prompt


@dataclass
class AdversarialVendor(Vendor):
    """A vendor principal for adversarial.

    Reuses the production Vendor: the reservation clamp still guarantees the
    emitted price is never below the vendor's true cost. Only the disposition and
    the injected per-vendor model are new; the honest/dishonest objective is
    carried entirely by the (minimal-contrast) prompt. delivery_days is treated as
    the vendor's TRUE fulfillable delivery; the model may CLAIM a different value
    (deception) — that claim is what the buyer sees, and the evaluator scores it
    against the true value.
    """

    disposition: str = "honest"
    model: object = None

    @property
    def true_cost(self) -> float:
        return self.reservation_price

    @property
    def true_delivery(self) -> int:
        return self.delivery_days

    def build_prompt(self, rfq, round_index, buyer_counter, history) -> str:
        return vendor_prompt(self, rfq, round_index, buyer_counter, history)
