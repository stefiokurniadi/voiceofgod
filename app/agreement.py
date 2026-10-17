"""Local agreement over successive speech hypotheses.

A word is confirmed when it occupies the same position in two consecutive
transcripts. Confirmed words may be finalised. The unconfirmed tail stays
provisional and is allowed to change.
"""

from __future__ import annotations

import re


def normalise_token(token: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE).casefold()


def common_prefix_len(previous: list[str], current: list[str]) -> int:
    count = 0
    for left, right in zip(previous, current):
        if normalise_token(left) != normalise_token(right) or not normalise_token(left):
            break
        count += 1
    return count


class LocalAgreement:
    def __init__(self) -> None:
        self.previous: list[str] | None = None

    def reset(self) -> None:
        self.previous = None

    def update(self, words: list[str]) -> int:
        """Return how many leading words are confirmed by this hypothesis."""
        if self.previous is None:
            self.previous = list(words)
            return 0
        confirmed = common_prefix_len(self.previous, words)
        self.previous = list(words)
        return confirmed


def ends_sentence(token: str) -> bool:
    stripped = token.strip()
    return stripped.endswith((".", "?", "!", "…")) or stripped in {".", "?", "!"}
