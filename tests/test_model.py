import math

import pytest
import torch
from torch import nn

from medha.model import GPT, Block, Config, Embedding, KVCache, SelfAttention, attention
from medha.sampler import generate

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

# A small GPT: the six words of "the cat sat on the mat" and the end token, as in the notebooks.
SMALL = Config(vocab_size=7, context=8, width=8, heads=2, layers=2)


def test_a_block_adds_its_changes_to_the_residual_stream():
    torch.manual_seed(0)
    block = Block(SMALL)
    for last in (block.attention.out, block.mlp.down):  # the last Linear of each part now gives 0
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    x = torch.randn(1, 5, 8)
    assert torch.equal(block(x), x)  # both parts add 0, so x passes through unchanged


def test_the_weight_count_matches_the_hand_count():
    model = GPT(SMALL)
    tables = 7 * 8 + 8 * 8  # the token table, which the head reuses, and the position table
    attention = (8 * 24 + 24) + (8 * 8 + 8)
    mlp = (8 * 32 + 32) + (32 * 8 + 8)
    norms = 2 * (8 + 8)
    total = 0
    for weights in model.parameters():
        total += weights.numel()
    assert total == tables + 2 * (attention + mlp + norms) + (8 + 8)


def test_through_the_whole_stack_no_position_reads_a_later_token():
    torch.manual_seed(0)
    model = GPT(SMALL)
    story = torch.tensor([[0, 1, 2, 3, 4, 1, 5, 0]])  # <|endoftext|> the cat sat on the mat <|endoftext|>
    changed = story.clone()
    changed[0, 5] = 6  # a new token at place 5
    assert torch.equal(model(changed)[0, :5], model(story)[0, :5])
    assert not torch.allclose(model(changed)[0, 5], model(story)[0, 5])


def test_untrained_every_token_has_about_the_same_chance():
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=2_000, context=16, width=32, heads=2, layers=2))
    ids = torch.randint(0, 2_000, (4, 16))
    with torch.no_grad():
        logits = model(ids)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 2_000), ids[:, 1:].reshape(-1))
    assert float(loss) == pytest.approx(math.log(2_000), abs=0.05)


def test_a_full_window_drops_the_oldest_tokens():
    torch.manual_seed(0)
    model = GPT(SMALL)
    ids = [0, 1, 2, 3, 4, 1, 5, 0, 1, 2, 3]  # 11 ids, more than the window of 8
    assert torch.equal(model.next_logits(ids), model.next_logits(ids[-8:]))


def test_one_new_query_reads_every_key_before_it():
    torch.manual_seed(0)
    q, k, v = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    last = attention(q[-1:], k, v)  # a decode step: only the newest token asks, and it may read all five
    assert torch.allclose(last.out[0], attention(q, k, v).out[-1], atol=1e-6)


def test_with_the_cache_the_model_writes_the_same_tokens():
    torch.manual_seed(0)
    model = GPT(SMALL)  # a window of 8, so 20 new tokens fill it, then slide it
    cache = KVCache(SMALL.layers)
    runs = []
    for use in (None, cache):
        steps = generate(model, [0, 1, 2], max_new_tokens=20, end=7, generator=torch.Generator().manual_seed(0),
                         cache=use)  # 7 is no token here, so the loop runs all 20 steps
        runs.append([step.token for step in steps])
    assert runs[0] == runs[1]
    assert len(runs[0]) == 20
    assert cache.length == 8  # the cache was used: it holds the last full window
    ids = [0, 1, 2] + runs[0]
    fresh = KVCache(SMALL.layers)
    for end in range(3, len(ids)):  # the prefill, each decode step, and each step past the full window
        assert torch.allclose(model.next_logits(ids[:end], fresh), model.next_logits(ids[:end]), atol=1e-6)
        assert fresh.length == min(end, 8)  # it holds every token read, so the next step reads one new token
