import math
import random

import pytest

from vak.counts import PairCounts
from vak.loss import held_out_loss

# A vocabulary of six tokens; a token's id is its place in the list.
WORDS = ["<|endoftext|>", "the", "cat", "sat", "on", "mat"]
STORY = [0, 1, 2, 3, 4, 1, 5, 0]  # <|endoftext|> the cat sat on the mat <|endoftext|>


def test_counts_become_chances_with_add_one_smoothing():
    model = PairCounts.from_stories([STORY], vocab_size=len(WORDS))
    assert model.counts[1].tolist() == [0, 0, 1, 0, 0, 1]  # after "the": cat once, mat once
    # (count + 1) / (2 + 6): cat and mat 2/8, every other token 1/8
    assert model.chances[1].tolist() == pytest.approx([1 / 8, 1 / 8, 2 / 8, 1 / 8, 1 / 8, 2 / 8])
    assert model.next_logits([0, 3, 4, 1]).exp().tolist() == pytest.approx(model.chances[1].tolist())


def test_the_loss_of_each_token_is_minus_the_log_of_its_chance():
    model = PairCounts.from_stories([STORY], vocab_size=len(WORDS))
    held_out = [0, 1, 2, 3, 4, 1, 2, 0]  # the cat sat on the cat: "cat" then the end was never seen
    expected = [-math.log(2 / 7), -math.log(2 / 8), -math.log(2 / 7), -math.log(2 / 7), -math.log(2 / 7),
                -math.log(2 / 8), -math.log(1 / 7)]
    assert model.losses(held_out) == pytest.approx(expected)
    assert model.unseen(held_out) == 1


def test_the_held_out_loss_is_per_token_with_an_interval_around_it():
    draw = random.Random(0)
    stories = []
    for _ in range(40):
        length = draw.randint(5, 300)
        stories.append([draw.gauss(3.7, 1.5) for _ in range(length)])
    tokens = sum(len(losses) for losses in stories)
    loss = held_out_loss(stories, seed=0)
    assert loss.mean == pytest.approx(sum(sum(losses) for losses in stories) / tokens)  # not a mean of story means
    assert loss.tokens == tokens and loss.stories == 40
    assert loss.low < loss.mean < loss.high
    assert loss.high - loss.low < 0.5


def test_the_interval_draws_whole_stories_not_single_tokens():
    # 40 stories of 100 tokens. Inside a story every token costs the same: 3.0 in half the stories, 5.0 in the rest.
    # Whole stories: the mean moves by about 1 / sqrt(40) = 0.16 between draws, so the interval spans about 0.6.
    # Single tokens would count as 4,000 separate draws, and the interval would span only about 0.06.
    stories = []
    for number in range(40):
        cost = 3.0 if number % 2 == 0 else 5.0
        stories.append([cost] * 100)
    loss = held_out_loss(stories, seed=0)
    assert loss.mean == pytest.approx(4.0)
    assert loss.high - loss.low > 0.3
