"""Offline adversarial pilot.

Validates the experiment machinery end-to-end with the offline doubles — ZERO live
API calls, zero credits. Prints per-cell causal/absolute metrics and an estimate of
how many live model calls the equivalent live pilot would make. It does NOT call any
live provider.

Usage: python -m eval.adversarial.run_pilot
"""

from eval.adversarial.harness import Bundle, run_matrix
from eval.adversarial.offline import OfflineBuyerModel, OfflineInterpreterModel, OfflineVendorModel
from eval.adversarial.scenarios import PILOT_SCENARIOS

K = 3
COMPOSITIONS = [0, 1, 2, 3]


def offline_bundle(seed):
    return Bundle(
        buyer=OfflineBuyerModel(),
        interpreter=OfflineInterpreterModel(),
        vendor=OfflineVendorModel(),
        fingerprint={"deterministic": True, "offline": True, "seed": seed},
    )


def _count_calls(record):
    log = record["log"]
    vendor = sum(1 for e in log if e["event"] in ("vendor_message", "vendor_error"))
    interp = sum(1 for e in log if e["event"] in ("offer", "offer_dropped"))
    buyer = sum(1 for e in log if e["event"] == "decision")
    return vendor + interp + buyer


def _negotiations_in(report):
    total = 0
    for cell in report["cells"]:
        if cell.get("composition") == "0_dishonest":
            total += cell["k"]  # baseline runs
        else:
            total += cell["k"] * 4  # K matched pairs (2) + K null pairs (2)
    return total


def main():
    report = run_matrix(PILOT_SCENARIOS, COMPOSITIONS, offline_bundle, K, tag="pilot_offline", write=True)

    # average model calls per negotiation, measured from the stored records
    calls, negs = 0, 0
    for cell in report["cells"]:
        if "records" in cell and cell.get("composition") != "0_dishonest":
            for rec in cell["records"]:
                calls += _count_calls(rec["A"]) + _count_calls(rec["B"])
                negs += 2
        elif cell.get("composition") == "0_dishonest":
            for rec in cell["records"]:
                calls += _count_calls(rec["run"])
                negs += 1
    avg = calls / negs if negs else 0

    print(f"offline pilot: {len(report['cells'])} cells, K={K}, LIVE API calls made = 0")
    print(f"avg model calls / negotiation (measured) = {avg:.1f}")
    est_neg = _negotiations_in(report)
    print(f"equivalent LIVE pilot would run ~{est_neg} negotiations -> ~{int(est_neg * avg)} model calls")
    print(f"(cost depends on provider/model/token pricing; not estimated in $ here)\n")

    for cell in report["cells"]:
        if cell.get("composition") == "0_dishonest":
            print(f"[{cell['scenario']}] 0-dishonest baseline: mean_welfare={cell['mean_welfare']:.1f}")
        else:
            pe = cell["paired_effect"]
            print(
                f"[{cell['scenario']}] target={cell['target_vendor']} dishonest={cell['dishonest_ids']} "
                f"mean_delta={pe['mean_delta']:.1f} welfare_loss={cell['welfare_loss_rate']['rate']:.2f} "
                f"welfare_reducing_win={cell['welfare_reducing_win_rate']['rate']:.2f} fair_win={cell['fair_win_rate']['rate']:.2f} "
                f"won_B={cell['won_rate_B']['rate']:.2f} sec_fail={cell['security_failure_rate']['rate']:.2f}"
            )
    print(f"\nreport written to {report.get('written_to')}")


if __name__ == "__main__":
    main()
