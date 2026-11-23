"""The model file: a GPT, built one part per step. Attention comes first: the part that lets each token read the
tokens before it."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionSteps:
    """Each step of attention, kept so it can be checked by hand. Row i is token i; column j is token j."""
    scores: torch.Tensor  # token i's query times token j's key: how well j matches what i looks for
    scaled: torch.Tensor  # the scores divided by the square root of the head size
    masked: torch.Tensor  # minus infinity wherever j comes after i
    weights: torch.Tensor  # softmax, row by row: each row adds up to 1
    out: torch.Tensor  # each token's mix of the values, by its row of weights


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> AttentionSteps:
    """One head of causal attention. q, k and v hold one row per token. Any axes before those two, such as a batch
    or a head, are carried along."""
    scores = q @ k.transpose(-2, -1)
    scaled = scores / math.sqrt(q.shape[-1])  # without it, bigger heads give bigger scores and a peakier softmax
    tokens = q.shape[-2]
    later = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).triu(diagonal=1)  # True where j > i
    masked = scaled.masked_fill(later, float("-inf"))  # softmax turns minus infinity into a weight of 0
    weights = torch.softmax(masked, dim=-1)
    out = weights @ v
    return AttentionSteps(scores, scaled, masked, weights, out)
