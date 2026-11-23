import math

import pytest
import torch

from medha.model import attention

# Three tokens, "The river bank". Each vector has two numbers, read as "water" and "money".
QUERIES = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]])
KEYS = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, 0.5]])
VALUES = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])


def test_attention_weights_and_output_match_the_hand_computation():
    steps = attention(QUERIES, KEYS, VALUES)
    # The bank row by hand: its query (1.5, 1.5) times each key gives 0, 3 and 1.5. Divide by the square root of 2,
    # then softmax.
    exps = [math.exp(0 / math.sqrt(2)), math.exp(3 / math.sqrt(2)), math.exp(1.5 / math.sqrt(2))]
    bank = [exps[0] / sum(exps), exps[1] / sum(exps), exps[2] / sum(exps)]
    assert steps.weights[2].tolist() == pytest.approx(bank)
    assert steps.out[2].tolist() == pytest.approx([bank[1] * 1.0 + bank[2] * 0.5, bank[2] * 0.5])
    assert steps.weights[0].tolist() == [1.0, 0.0, 0.0]  # the first token can read only itself


def test_no_token_reads_a_later_token():
    changed_keys, changed_values = KEYS.clone(), VALUES.clone()
    changed_keys[2], changed_values[2] = torch.tensor([9.0, 9.0]), torch.tensor([9.0, 9.0])  # a new last token
    before = attention(QUERIES, KEYS, VALUES).out
    after = attention(QUERIES, changed_keys, changed_values).out
    assert torch.equal(after[:2], before[:2])
    assert not torch.equal(after[2], before[2])
