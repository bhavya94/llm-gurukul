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
    or a head, are carried along. With a KV cache, q holds only the newest tokens, so it can have fewer rows than k:
    the queries are then the last tokens."""
    scores = q @ k.transpose(-2, -1)
    scaled = scores / math.sqrt(q.shape[-1])  # without it, bigger heads give bigger scores and a peakier softmax
    queries, keys = q.shape[-2], k.shape[-2]
    # Query i sits at place keys - queries + i, so it may read every key up to that place. True marks a later key.
    later = torch.ones(queries, keys, dtype=torch.bool, device=q.device).triu(diagonal=keys - queries + 1)
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

    def forward(self, ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        """start: the place of the first id; with a KV cache, the number of tokens already read."""
        places = torch.arange(start, start + ids.shape[-1], device=ids.device)  # start, start + 1, ...
        return self.tokens(ids) + self.positions(places)


class SelfAttention(nn.Module):
    """Many heads side by side. One Linear makes the queries, keys and values of every head at once. Each head runs
    attention on its own slice of width / heads numbers, and a last Linear mixes the heads' outputs."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, cache: "KVCache | None" = None, layer: int = 0) -> torch.Tensor:
        batch, tokens, width = x.shape
        size = width // self.heads
        qkv = self.qkv(x)
        q, k, v = qkv.split(width, dim=-1)
        log.debug("q, k and v from one Linear: %s, cut into three of %s", tuple(qkv.shape), tuple(q.shape))
        q = q.view(batch, tokens, self.heads, size).transpose(1, 2)  # (batch, heads, tokens, size)
        k = k.view(batch, tokens, self.heads, size).transpose(1, 2)
        v = v.view(batch, tokens, self.heads, size).transpose(1, 2)
        log.debug("split into %d heads of %d numbers: each %s", self.heads, size, tuple(q.shape))
        if cache is not None:
            k, v = cache.add(layer, k, v)  # the keys and values of every token so far, the new ones last
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

    def forward(self, x: torch.Tensor, cache: "KVCache | None" = None, layer: int = 0) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cache, layer)
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

    def forward(self, ids: torch.Tensor, cache: "KVCache | None" = None) -> torch.Tensor:
        """ids: (batch, tokens). Gives logits: (batch, tokens, vocab_size), a guess at the next token at each place.
        With a cache, ids are only the new tokens, and they sit after the cache's tokens."""
        start = 0 if cache is None else cache.length
        x = self.embedding(ids, start)
        for layer, block in enumerate(self.blocks):
            x = block(x, cache, layer)
        return self.head(self.norm(x))

    def next_logits(self, ids: list[int], cache: "KVCache | None" = None) -> torch.Tensor:
        """What the decode loop needs (medha.sampler.Model). The model reads at most context ids, so once the window
        is full, the oldest ids drop out. It runs where its weights are, and gives the logits back on the CPU, where
        the sampler draws.

        Without a cache, it reads the whole window again at every step. With a cache, it reads only the ids the cache
        has not seen: the whole prompt at the first step (prefill), then one id per step (decode). Once the window is
        full, each step moves every token one place back, so every saved key and value is out of date. The cache then
        starts again from the last window: the same work as without it, and the same logits."""
        window = ids[-self.config.context:]
        if cache is not None and len(ids) > self.config.context:
            cache.clear()
        new = window if cache is None else window[cache.length:]
        with torch.no_grad():
            logits = self(torch.tensor([new], device=self.head.weight.device), cache)
        return logits[0, -1].cpu()


class KVCache:
    """Each block's keys and values for every token read so far, so the next step reads only the new tokens. A cache
    belongs to one text: the ids given to next_logits must start with the ones it has read."""

    def __init__(self, layers: int):
        self.keys: list[torch.Tensor | None] = [None] * layers
        self.values: list[torch.Tensor | None] = [None] * layers

    @property
    def length(self) -> int:
        """How many tokens the cache holds."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[-2]

    def add(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one block's new keys and values, and give back all of that block's keys and values."""
        if self.keys[layer] is not None:
            k = torch.cat([self.keys[layer], k], dim=-2)
            v = torch.cat([self.values[layer], v], dim=-2)
        self.keys[layer] = k
        self.values[layer] = v
        return k, v

    def clear(self) -> None:
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)

    def size_in_bytes(self) -> int:
        total = 0
        for saved in self.keys + self.values:
            if saved is not None:
                total += saved.numel() * saved.element_size()
        return total
