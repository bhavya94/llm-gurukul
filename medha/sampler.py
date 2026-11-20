"""The sampler and the decode loop. The sampler turns one row of scores into one token: temperature, then top-k,
then top-p, then a random draw. The loop does that again and again, one token at a time, to write text."""

from dataclasses import dataclass
from typing import Protocol

import torch


class Model(Protocol):
    """What the loop needs from a model: one score, a logit, per vocabulary token, for the token after ids."""

    def next_logits(self, ids: list[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class Step:
    token: int  # the id the sampler drew
    chance: float  # the chance it had, after the sampler's settings


def chances(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
            top_p: float | None = None) -> torch.Tensor:
    """The chances the sampler draws from. In order: divide the logits by the temperature, keep the top k tokens,
    keep the smallest set whose chances add up to top_p, and rescale the kept chances to add up to 1.
    Temperature 0 means greedy: the top token gets all the chance."""
    if temperature == 0:
        greedy = torch.zeros_like(logits)
        greedy[torch.argmax(logits)] = 1.0
        return greedy
    scores = logits / temperature  # below 1 sharpens the chances, above 1 flattens them
    if top_k is not None:
        best = torch.topk(scores, min(top_k, len(scores))).indices
        kept = torch.full_like(scores, float("-inf"))  # a score of minus infinity gets a chance of 0
        kept[best] = scores[best]
        scores = kept
    probs = torch.softmax(scores, dim=-1)
    if top_p is not None:
        sorted_probs, order = torch.sort(probs, descending=True, stable=True)  # on a tie, the lower id first
        above = torch.cumsum(sorted_probs, dim=-1) - sorted_probs  # the chance of all the tokens ranked above
        probs[order[above >= top_p]] = 0.0  # the tokens above already reach top_p, so drop these
        probs = probs / probs.sum()
    return probs


def pick(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Draw one token at random: a token with chance 0.3 comes out 3 times in 10, on average."""
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def generate(model: Model, ids: list[int], max_new_tokens: int, end: int, generator: torch.Generator,
             temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> list[Step]:
    """The decode loop: score the next token, pick one, append it, and go again. It stops at the end token or after
    max_new_tokens, whichever comes first. Each step records the token and its chance."""
    ids = list(ids)
    steps = []
    for _ in range(max_new_tokens):
        probs = chances(model.next_logits(ids), temperature, top_k, top_p)
        token = pick(probs, generator)
        steps.append(Step(token, float(probs[token])))
        if token == end:
            break
        ids.append(token)
    return steps
