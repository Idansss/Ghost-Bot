from app.core.knowledge import extract_definition_term, try_answer_definition


def test_extract_definition_term_basic() -> None:
    assert extract_definition_term("what is SMC") == "smc"
    assert extract_definition_term("Define fair value gap") == "fair value gap"


def test_try_answer_definition_matches_aliases() -> None:
    out = try_answer_definition("what is fvg?")
    assert out is not None
    assert "fvg" in out.lower()

