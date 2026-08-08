from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

_EMBEDDING_PRICES_USD_PER_MILLION_TOKENS: dict[str, Decimal] = {
    "text-embedding-3-large": Decimal("0.13"),
    "text-embedding-3-small": Decimal("0.02"),
    "text-embedding-ada-002": Decimal("0.10"),
}

_COMPLETION_PRICES_USD_PER_MILLION_TOKENS: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-5.6": (Decimal("5.00"), Decimal("30.00")),
    "gpt-5.6-sol": (Decimal("5.00"), Decimal("30.00")),
    "gpt-5.6-terra": (Decimal("2.00"), Decimal("12.00")),
    "gpt-5.6-luna": (Decimal("0.20"), Decimal("1.20")),
    "gpt-5": (Decimal("1.25"), Decimal("10.00")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("2.00")),
    "gpt-5-nano": (Decimal("0.05"), Decimal("0.40")),
    "gpt-5.1": (Decimal("1.25"), Decimal("10.00")),
    "gpt-5.1-mini": (Decimal("0.25"), Decimal("2.00")),
    "gpt-5.1-nano": (Decimal("0.05"), Decimal("0.40")),
    "gpt-5.2": (Decimal("1.75"), Decimal("14.00")),
    "gpt-5.2-mini": (Decimal("0.25"), Decimal("2.00")),
    "gpt-5.2-nano": (Decimal("0.05"), Decimal("0.40")),
    "gpt-5.4": (Decimal("1.75"), Decimal("14.00")),
    "gpt-5.4-mini": (Decimal("0.75"), Decimal("4.50")),
    "gpt-5.4-nano": (Decimal("0.20"), Decimal("1.25")),
    "gpt-4.1": (Decimal("2.00"), Decimal("8.00")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1-nano": (Decimal("0.10"), Decimal("0.40")),
    "gpt-4o": (Decimal("5.00"), Decimal("15.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
}

SUPPORTED_COMPLETION_MODEL_NAMES = frozenset(_COMPLETION_PRICES_USD_PER_MILLION_TOKENS)


def _to_decimal(value: int | float | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize_usd(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def estimate_embedding_cost_usd(model_name: str, tokens: int | None) -> Decimal:
    if tokens is None or tokens <= 0:
        return Decimal("0")
    price_per_million = _EMBEDDING_PRICES_USD_PER_MILLION_TOKENS.get(model_name)
    if price_per_million is None:
        return Decimal("0")
    return _quantize_usd(_to_decimal(tokens) * price_per_million / Decimal("1000000"))


def estimate_completion_cost_usd(model_name: str, input_tokens: int | None, output_tokens: int | None) -> Decimal:
    if input_tokens is None and output_tokens is None:
        return Decimal("0")
    price = _COMPLETION_PRICES_USD_PER_MILLION_TOKENS.get(model_name)
    if price is None:
        return Decimal("0")
    input_price, output_price = price
    input_cost = _to_decimal(input_tokens or 0) * input_price / Decimal("1000000")
    output_cost = _to_decimal(output_tokens or 0) * output_price / Decimal("1000000")
    return _quantize_usd(input_cost + output_cost)
