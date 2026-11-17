import pytest
import torch

from medha.sampler import chances, generate, pick
from vak.counts import PairCounts

# Eight logits, best first. The test below works out their chances at temperature 0.7.
LOGITS = torch.tensor([15.965, 15.397, 14.235, 12.679, 12.518, 11.950, 11.648, 11.565])


def test_a_temperature_near_zero_always_gives_the_top_token():
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        assert pick(chances(LOGITS, temperature=0.05), generator) == 0
    assert chances(LOGITS, temperature=0).tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_top_k_keeps_exactly_the_k_likeliest_tokens():
    kept = chances(LOGITS, top_k=3)
    assert (kept > 0).nonzero().flatten().tolist() == [0, 1, 2]
    assert float(kept.sum()) == pytest.approx(1.0)


def test_top_p_keeps_the_smallest_set_whose_chance_adds_up_to_p():
    # At temperature 0.7 the first two hold 0.644 + 0.286 = 0.930, past 0.9; the first alone holds less.
    assert chances(LOGITS, temperature=0.7).tolist()[:3] == pytest.approx([0.644, 0.286, 0.054], abs=1e-3)
    kept = chances(LOGITS, temperature=0.7, top_p=0.9)
    assert kept.tolist() == pytest.approx([0.692, 0.308, 0, 0, 0, 0, 0, 0], abs=1e-3)
    assert (chances(LOGITS, temperature=0.7, top_p=0.6) > 0).sum() == 1  # 0.644 alone reaches 0.6


def test_the_decode_loop_repeats_with_its_seed_and_stops_at_the_end_token():
    words = ["<|endoftext|>", "the", "cat", "sat", "on", "mat"]
    model = PairCounts.from_stories([[0, 1, 2, 3, 4, 1, 5, 0]], vocab_size=len(words))  # the cat sat on the mat
    runs = []
    for seed in (7, 7):
        steps = generate(model, [0, 1], max_new_tokens=50, end=0, generator=torch.Generator().manual_seed(seed))
        runs.append([step.token for step in steps])
    assert runs[0] == runs[1]
    assert runs[0][-1] == 0 and 0 not in runs[0][:-1]  # it stopped at the first end token it drew
    greedy = generate(model, [0, 1], max_new_tokens=5, end=0, generator=torch.Generator(), temperature=0)
    assert [step.token for step in greedy] == [2, 3, 4, 1, 2]  # the first top token wins ties: cat, not mat
    assert [step.chance for step in greedy] == [1, 1, 1, 1, 1]
