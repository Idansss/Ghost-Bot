from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.nlu import Intent, parse_message


@settings(max_examples=300, deadline=None)
@given(st.text(min_size=0, max_size=500))
def test_parse_message_never_crashes(text: str) -> None:
    parsed = parse_message(text)
    assert parsed.intent in Intent
    assert isinstance(parsed.entities, dict)
    assert isinstance(parsed.requires_followup, bool)


@settings(max_examples=200, deadline=None)
@given(
    st.one_of(
        st.tuples(
            st.sampled_from(["BTC", "ETH", "SOL"]),
            st.sampled_from(["long", "short"]),
        ).map(lambda t: f"{t[0]} {t[1]}"),
        st.tuples(
            st.sampled_from(["BTC", "ETH", "SOL"]),
            st.integers(min_value=1, max_value=1_000_000).map(str),
            st.sampled_from(["above", "below", "cross", ""]),
        ).map(lambda t: f"alert {t[0]} {t[1]} {t[2]}".strip()),
        st.tuples(
            st.sampled_from(["1h", "4h", "1d"]),
            st.sampled_from(["overbought", "oversold"]),
        ).map(lambda t: f"rsi {t[0]} {t[1]}"),
        st.tuples(
            st.integers(min_value=2, max_value=500),
            st.sampled_from(["1h", "4h", "1d"]),
        ).map(lambda t: f"ema {t[0]} {t[1]} top 10"),
    )
)
def test_nlu_required_entities_present_for_common_commands(text: str) -> None:
    parsed = parse_message(text)

    if parsed.intent == Intent.ANALYSIS:
        assert parsed.entities.get("symbol")
    if parsed.intent == Intent.ALERT_CREATE:
        assert parsed.entities.get("symbol")
        assert parsed.entities.get("target_price") is not None
    if parsed.intent == Intent.RSI_SCAN:
        assert parsed.entities.get("timeframe")
        assert parsed.entities.get("mode") in {"overbought", "oversold"}
    if parsed.intent == Intent.EMA_SCAN:
        assert parsed.entities.get("timeframe")
        assert parsed.entities.get("ema_length") is not None

