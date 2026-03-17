from app.core.howto import try_answer_howto


def test_try_answer_howto_alerts() -> None:
    out = try_answer_howto("how do i create an alert")
    assert out is not None
    assert "set an alert" in out.lower()


def test_try_answer_howto_not_responding() -> None:
    out = try_answer_howto("the bot is not responding")
    assert out is not None
    assert "not responding" in out.lower()

