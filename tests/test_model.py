import math

import pytest
import torch

from medha.model import Embedding, SelfAttention, attention

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


def test_each_head_runs_attention_on_its_own_slice():
    torch.manual_seed(0)
    layer = SelfAttention(width=8, heads=2)
    x = torch.randn(1, 5, 8)  # one sentence of 5 tokens, 8 numbers each
    q, k, v = layer.qkv(x)[0].split(8, dim=-1)
    outputs = []
    for head in range(2):
        part = slice(4 * head, 4 * head + 4)  # head 0 reads numbers 0 to 3 of each vector, head 1 reads 4 to 7
        outputs.append(attention(q[:, part], k[:, part], v[:, part]).out)
    by_hand = layer.out(torch.cat(outputs, dim=-1))
    assert torch.allclose(layer(x)[0], by_hand, atol=1e-6)


def test_with_position_the_order_of_earlier_tokens_changes_the_output():
    torch.manual_seed(0)
    embedding = Embedding(vocab_size=1_000, context=8, width=8)
    layer = SelfAttention(width=8, heads=2)
    dog_saw_cat = torch.tensor([[373, 652, 413, 263, 705, 14]])  # "The dog saw the cat."
    cat_saw_dog = torch.tensor([[373, 705, 413, 263, 652, 14]])  # "The cat saw the dog.": the last token stays "."
    without = [layer(embedding.tokens(dog_saw_cat))[0, -1], layer(embedding.tokens(cat_saw_dog))[0, -1]]
    with_position = [layer(embedding(dog_saw_cat))[0, -1], layer(embedding(cat_saw_dog))[0, -1]]
    assert torch.allclose(without[0], without[1], atol=1e-6)  # attention alone sees a set
    assert (with_position[0] - with_position[1]).abs().max() > 0.01
