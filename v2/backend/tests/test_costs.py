from decimal import Decimal

from v2_auditory_learning.costs import estimate_completion_cost_usd


def test_estimate_completion_cost_usd_uses_gpt_5_4_mini_price() -> None:
    cost = estimate_completion_cost_usd("gpt-5.4-mini", 1000, 2000)
    assert cost == Decimal("0.009750")


def test_estimate_completion_cost_usd_returns_zero_for_unknown_model() -> None:
    assert estimate_completion_cost_usd("unknown-model", 1000, 2000) == Decimal("0")
