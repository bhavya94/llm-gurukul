"""The model file: a GPT, built one part per step. Attention comes first: the part that lets each token read the
tokens before it."""

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn

log = logging.getLogger(__name__)


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


class Embedding(nn.Module):
    """Each token's vector: its row of the token table, plus the row of the position table for the place it sits in.
    Attention alone treats the tokens before a token as a set; the position rows tell it their order."""

    def __init__(self, vocab_size: int, context: int, width: int):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, width)  # one learned row per vocabulary token
        self.positions = nn.Embedding(context, width)  # one learned row per place in the window

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        places = torch.arange(ids.shape[-1], device=ids.device)  # 0, 1, 2, ...
        return self.tokens(ids) + self.positions(places)


class SelfAttention(nn.Module):
    """Many heads side by side. One Linear makes the queries, keys and values of every head at once. Each head runs
    attention on its own slice of width / heads numbers, and a last Linear mixes the heads' outputs."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        size = width // self.heads
        qkv = self.qkv(x)
        q, k, v = qkv.split(width, dim=-1)
        log.debug("q, k and v from one Linear: %s, cut into three of %s", tuple(qkv.shape), tuple(q.shape))
        q = q.view(batch, tokens, self.heads, size).transpose(1, 2)  # (batch, heads, tokens, size)
        k = k.view(batch, tokens, self.heads, size).transpose(1, 2)
        v = v.view(batch, tokens, self.heads, size).transpose(1, 2)
        log.debug("split into %d heads of %d numbers: each %s", self.heads, size, tuple(q.shape))
        steps = attention(q, k, v)
        log.debug("weights, one table per head: %s", tuple(steps.weights.shape))
        mixed = steps.out.transpose(1, 2).reshape(batch, tokens, width)  # the heads side by side again
        log.debug("the heads side by side again: %s", tuple(mixed.shape))
        return self.out(mixed)
