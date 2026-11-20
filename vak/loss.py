"""The held-out loss: how surprised a model is, on average, by each next token of the stories it never trained on."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class HeldOutLoss:
    mean: float  # the mean loss per token, in nats: natural-log units
    low: float  # the 95% interval runs from low to high
    high: float
    tokens: int
    stories: int


def _per_token(totals: np.ndarray, counts: np.ndarray) -> float:
    """The mean loss per token, for one draw of stories."""
    return np.sum(totals) / np.sum(counts)


def held_out_loss(losses_per_story: list[list[float]], seed: int) -> HeldOutLoss:
    """The mean loss over every held-out token. The 95% interval comes from a bootstrap: draw the stories again at
    random, with repeats, many times, and see how far the mean moves. It draws whole stories, because the tokens of
    one story depend on each other."""
    totals = []
    counts = []
    for losses in losses_per_story:
        totals.append(sum(losses))
        counts.append(len(losses))
    mean = sum(totals) / sum(counts)
    result = stats.bootstrap((totals, counts), _per_token, paired=True, vectorized=False, confidence_level=0.95,
                             rng=seed)
    interval = result.confidence_interval
    return HeldOutLoss(mean, float(interval.low), float(interval.high), sum(counts), len(totals))
