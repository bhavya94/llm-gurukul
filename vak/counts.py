"""The smallest language model: count which token follows which in the training stories, then turn the counts into
the chance of each next token. It sees only the last token, so it has no memory of what came before."""

import math
from collections import Counter
from pathlib import Path

import torch

SMOOTHING = 1.0  # add-one: count every pair once more than it was seen, so no next token has a chance of 0


class PairCounts:
    """Owns the table counts[a, b]: how often token b followed token a in the training stories."""

    def __init__(self, counts: torch.Tensor):
        self.counts = counts
        smoothed = counts + SMOOTHING
        self.chances = smoothed / smoothed.sum(dim=1, keepdim=True)  # row a: the chance of each token after a

    @classmethod
    def from_stories(cls, stories: list[list[int]], vocab_size: int) -> "PairCounts":
        pairs = Counter()
        for ids in stories:
            pairs.update(zip(ids, ids[1:]))  # every pair of neighbouring tokens in this story
        counts = torch.zeros(vocab_size, vocab_size)
        for (a, b), n in pairs.items():
            counts[a, b] = n
        return cls(counts)

    @classmethod
    def load(cls, path: Path) -> "PairCounts":
        return cls(torch.load(path))

    def save(self, path: Path) -> None:
        torch.save(self.counts, path)

    def next_logits(self, ids: list[int]) -> torch.Tensor:
        """One score per vocabulary token, for the token after ids. Only the last id matters. The scores are log
        chances, and softmax turns them back into the chances."""
        return torch.log(self.chances[ids[-1]])

    def losses(self, ids: list[int]) -> list[float]:
        """For each token after the first: minus the log of the chance the model gave it. A sure, right guess
        costs almost 0; a chance of 1 in 2,000 costs 7.6."""
        losses = []
        for a, b in zip(ids, ids[1:]):
            losses.append(-math.log(self.chances[a, b]))
        return losses

    def unseen(self, ids: list[int]) -> int:
        """How many pairs of neighbouring tokens in ids never appeared in training."""
        count = 0
        for a, b in zip(ids, ids[1:]):
            if self.counts[a, b] == 0:
                count += 1
        return count
