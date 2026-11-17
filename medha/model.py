"""The model file: a GPT in the classic GPT-2 shape. Attention lets each token read the tokens before it, a block
adds a small network for each token, and the GPT stacks the blocks. Each part is one class, so a part can be
swapped on its own."""

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """A GPT's settings, all in one place."""
    vocab_size: int  # tokens in the vocabulary
    context: int  # the most tokens the model reads at once: its window
    width: int  # numbers in each token's vector
    heads: int  # attention heads in each block; each reads width / heads numbers
    layers: int  # blocks in the stack


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


class MLP(nn.Module):
    """Each token on its own: widen its vector 4 times, apply GELU, and narrow it back."""

    def __init__(self, width: int):
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """One layer of the stack. Attention lets the tokens read each other, then the MLP works on each token alone.
    Each part reads a normalized copy of x and adds its change to x, so x itself, the residual stream, passes
    through every block."""

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = SelfAttention(config.width, config.heads)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = MLP(config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """The whole model: the Embedding, the stack of blocks, a last LayerNorm, and the head, which gives each position
    one logit per vocabulary token. The head reuses the token table, as in GPT-2."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.context, config.width)
        self.blocks = nn.ModuleList()
        for _ in range(config.layers):
            self.blocks.append(Block(config))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.embedding.tokens.weight  # one table to read tokens in and to score them out
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # small, so every token starts near an even chance
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, tokens). Gives logits: (batch, tokens, vocab_size), a guess at the next token at each place."""
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def next_logits(self, ids: list[int]) -> torch.Tensor:
        """What the decode loop needs (medha.sampler.Model). The model reads at most context ids, so once the window
        is full, the oldest ids drop out. It runs where its weights are, and gives the logits back on the CPU, where
        the sampler draws."""
        window = ids[-self.config.context:]
        with torch.no_grad():
            logits = self(torch.tensor([window], device=self.head.weight.device))
        return logits[0, -1].cpu()
