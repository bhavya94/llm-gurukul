import math
import random

import pytest
import torch
from torch.nn import functional as F

from medha.model import GPT, Config
from medha.tokenizer import train_bpe
from vak.data import encode_stories
from vak.loss import paired_difference
from vak.train import batch, join, story_losses, train

STORIES = ["Tom had a red ball. He threw it high, and his dog ran after it.",
           "The cat sat on the mat. It was warm, and the cat was happy."]


def test_stories_join_with_one_end_token_and_each_target_is_the_next_id():
    stream = join([[0, 5, 6, 0], [0, 7, 8, 9, 0]])  # two stories as ids, each framed by the end token, 0
    assert stream.tolist() == [0, 5, 6, 0, 7, 8, 9, 0]
    windows = []  # every window of 3 ids, with its targets: the same window moved one id on
    for start in range(len(stream) - 3):
        windows.append((stream[start:start + 3].tolist(), stream[start + 1:start + 4].tolist()))
    inputs, targets = batch(stream, size=50, context=3, generator=torch.Generator().manual_seed(0))
    drawn = []
    for row_in, row_out in zip(inputs.tolist(), targets.tolist()):
        drawn.append((row_in, row_out))
    for pair in drawn:
        assert pair in windows
    for pair in windows:
        assert pair in drawn  # the last window too


def test_training_lowers_the_loss():
    tokenizer = train_bpe(STORIES * 10, vocab_size=300)
    stream = join(encode_stories(tokenizer, STORIES * 50))
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=300, context=16, width=32, heads=2, layers=2))
    story = stream.tolist()[:40]
    before = sum(story_losses(model, story)) / 39
    train(model, stream, steps=150, tokenizer=tokenizer, seed=0)
    after = sum(story_losses(model, story)) / 39
    assert before == pytest.approx(math.log(300), abs=0.3)
    assert after < 1.0


def test_a_long_story_is_scored_once_per_token_with_at_least_half_a_window_before_it():
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=50, context=8, width=8, heads=2, layers=1))
    ids = list(range(1, 21))  # 20 ids: 19 tokens to score, with a window of 8
    losses = story_losses(model, ids)
    assert len(losses) == 19
    for t in range(1, 20):
        # token t is scored by the first window that reaches it: windows start at 0, 4, 8, 12 and end 8 ids later
        start = 0 if t <= 8 else 4 * math.ceil((t - 8) / 4)
        with torch.no_grad():
            logits = model(torch.tensor([ids[start:t]]))[0, -1]
        assert losses[t - 1] == pytest.approx(float(F.cross_entropy(logits, torch.tensor(ids[t]))), abs=1e-5)


def test_the_paired_difference_cancels_how_hard_each_story_is():
    draw = random.Random(0)
    theirs = []
    ours = []
    for number in range(40):
        hard = 3.0 if number % 2 == 0 else 5.0  # half the stories are easy, half are hard, for both models
        theirs.append([hard] * 100)
        ours.append([hard - 0.5 + draw.gauss(0, 0.05) for _ in range(100)])
    difference = paired_difference(ours, theirs, seed=0)
    assert difference.mean == pytest.approx(-0.5, abs=0.01)
    assert difference.high - difference.low < 0.05  # each model alone spans about 0.6 (test_counts.py)
    with pytest.raises(ValueError):
        paired_difference([[1.0, 2.0]], [[1.0]], seed=0)  # the two models must score the same tokens
