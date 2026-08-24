from guards.procurement import procurement_rank_key, rank_feasible_offers


RFQ = {"quantity": 100, "budget": 10_000, "max_delivery_days": 10}


def offer(vendor_id, unit_price, delivery_days):
    return {
        "vendor_id": vendor_id,
        "unit_price": unit_price,
        "delivery_days": delivery_days,
        "extraction_flag": "none",
        "confidence": 1.0,
    }


def test_lower_price_ranks_ahead_when_both_offers_are_feasible():
    ranked = rank_feasible_offers(RFQ, [offer("vendor_b", 42, 7), offer("vendor_a", 40, 9)])
    assert [o["vendor_id"] for o in ranked] == ["vendor_a", "vendor_b"]


def test_delivery_breaks_price_ties():
    ranked = rank_feasible_offers(RFQ, [offer("vendor_b", 40, 8), offer("vendor_a", 40, 7)])
    assert [o["vendor_id"] for o in ranked] == ["vendor_a", "vendor_b"]


def test_vendor_id_breaks_price_and_delivery_ties():
    ranked = rank_feasible_offers(RFQ, [offer("vendor_b", 40, 7), offer("vendor_a", 40, 7)])
    assert [o["vendor_id"] for o in ranked] == ["vendor_a", "vendor_b"]


def test_infeasible_offers_are_excluded_before_ranking():
    ranked = rank_feasible_offers(
        RFQ,
        [offer("too_expensive", 101, 1), offer("too_slow", 10, 11), offer("feasible", 50, 9)],
    )
    assert [o["vendor_id"] for o in ranked] == ["feasible"]


def test_hidden_truth_is_irrelevant_to_the_ranking_key():
    low = offer("vendor_a", 40, 8)
    high = offer("vendor_b", 42, 9)
    low["true_cost"] = 1000
    high["true_cost"] = 1
    assert procurement_rank_key(low) < procurement_rank_key(high)


def test_hidden_delivery_truth_is_irrelevant_to_the_ranking_key():
    fast_claim = offer("vendor_a", 40, 8)
    slow_claim = offer("vendor_b", 42, 9)
    fast_claim["true_delivery"] = 30
    slow_claim["true_delivery"] = 1
    assert procurement_rank_key(fast_claim) < procurement_rank_key(slow_claim)


def test_fair_reference_and_welfare_are_not_ranking_inputs():
    first = offer("vendor_a", 40, 8)
    second = offer("vendor_b", 42, 9)
    for item in (first, second):
        item["fair_ref"] = 999
        item["welfare"] = -999
    assert procurement_rank_key(first) < procurement_rank_key(second)


def test_ranking_is_deterministic():
    offers = [offer("vendor_c", 40, 8), offer("vendor_a", 40, 8), offer("vendor_b", 40, 8)]
    assert rank_feasible_offers(RFQ, offers) == rank_feasible_offers(RFQ, offers)
