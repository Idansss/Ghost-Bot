from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Messages:
    """Tiny i18n/message-catalog scaffold.

    Keep user-facing strings centralized over time so:
    - UX stays consistent
    - future localization is feasible
    - accessibility tweaks (wording, clarity) are easy
    """

    busy_generic: str = "still working — try again in a few seconds."
    unauthorized: str = "Unauthorized."


DEFAULT_MESSAGES = Messages()

