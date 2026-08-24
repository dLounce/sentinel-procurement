from dataclasses import dataclass

from agents.vendor import INJECTION_PAYLOAD, Vendor
from eval.adversarial.prompts import vendor_prompt
from eval.adversarial.prompts import raw_vendor_prompt

@dataclass
class AdversarialVendor(Vendor):
    """Vendor principal used by the adversarial evaluation.

    Honest vendors retain the production reservation-price enforcement.
    Dishonest vendors are allowed to represent price and delivery however they
    choose through the normal vendor-message interface; the evaluator compares
    those claims against hidden ground truth afterward.
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

    def propose_quote(
        self,
        rfq,
        round_index,
        buyer_counter,
        history,
        *,
        model,
        inject: bool = False,
    ) -> str:
        proposal = self._parse(
            model(self.build_prompt(rfq, round_index, buyer_counter, history))
        )

        unit_price = round(float(proposal["unit_price"]), 2)
        if self.disposition == "honest":
            unit_price = max(unit_price, self.reservation_price)

        delivery_days = max(1, int(proposal["delivery_days"]))
        note = str(proposal.get("note", "")).strip()[:200]

        message = (
            f"{note} Our price is ${unit_price:.2f} per unit, "
            f"delivery in {delivery_days} days."
        ).strip()

        if inject:
            message += INJECTION_PAYLOAD

        return message

    def propose_raw_message(
        self,
        rfq,
        round_index,
        buyer_counter,
        history,
        *,
        model,
    ) -> str:
        raw = model(
            raw_vendor_prompt(
                self,
                rfq,
                round_index,
                buyer_counter,
                history,
            )
        )

        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("vendor model returned empty raw message")

        return raw.strip()
