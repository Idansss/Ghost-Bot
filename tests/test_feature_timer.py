from __future__ import annotations

from app.core.metrics import FEATURE_DURATION_SECONDS, FeatureTimer


def test_feature_timer_observes_duration() -> None:
    before = float(FEATURE_DURATION_SECONDS.labels(feature="unit_test")._sum.get())  # type: ignore[attr-defined]
    with FeatureTimer("unit_test"):
        pass
    after = float(FEATURE_DURATION_SECONDS.labels(feature="unit_test")._sum.get())  # type: ignore[attr-defined]
    assert after >= before

