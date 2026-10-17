"""Word error rate and chrF, implemented here so the eval harness has no extra dependency."""

from __future__ import annotations

from collections import Counter


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    previous = list(range(len(hyp) + 1))
    for row, token in enumerate(ref, start=1):
        current = [row]
        for col, other in enumerate(hyp, start=1):
            cost = 0 if token == other else 1
            current.append(min(previous[col] + 1, current[col - 1] + 1, previous[col - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def chrf(reference: str, hypothesis: str, max_order: int = 6, beta: float = 2.0) -> float | None:
    """Character n-gram F-score. None when there is no translation to score."""
    if reference == "" and hypothesis == "":
        return None
    if reference == "" or hypothesis == "":
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        ref_grams = Counter(_ngrams(reference, order))
        hyp_grams = Counter(_ngrams(hypothesis, order))
        if not ref_grams or not hyp_grams:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        overlap = sum(min(count, ref_grams[gram]) for gram, count in hyp_grams.items())
        precisions.append(overlap / sum(hyp_grams.values()))
        recalls.append(overlap / sum(ref_grams.values()))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _ngrams(text: str, order: int) -> list[tuple[str, ...]]:
    chars = list(text)
    if len(chars) < order:
        return []
    return [tuple(chars[index : index + order]) for index in range(len(chars) - order + 1)]
