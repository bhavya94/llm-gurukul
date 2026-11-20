from vak.data import split, unique_stories

STORIES = [
    "Once upon a time, there was a little girl named Lily. She loved to play in the park.",
    "Tom had a red ball. He threw it high, and his dog ran after it.",
    "The sun was hot. Mia and her mom went to the beach to swim.",
    "Once upon a time, there was a little girl named Lily. She loved to play in the park.",  # in the file twice
    "Ben saw a big, scary dog. He was afraid, but the dog only wanted to play.",
    "   ",
    "A small bird could not fly. A kind boy gave it water, and soon it flew away.",
    "Sue found a shiny key in the garden. \"What does it open?\" she asked.",
    "The cat was sleepy. It curled up on the warm bed and closed its eyes.",
    "Max wanted a cookie. His dad said, \"First, eat your peas.\"",
]


def test_each_story_is_kept_once_in_file_order_and_blanks_are_dropped():
    unique = unique_stories(STORIES)
    assert len(unique) == 8
    assert unique[0] == STORIES[0] and unique[3] == STORIES[4]


def test_the_seed_fixes_the_split_and_no_story_is_in_both_parts():
    stories = unique_stories(STORIES)
    first = split(stories, n_train=6, n_held_out=2, seed=0)
    assert len(first.train) == 6 and len(first.held_out) == 2
    assert not set(first.train) & set(first.held_out)
    assert split(stories, n_train=6, n_held_out=2, seed=0) == first
    assert split(stories, n_train=6, n_held_out=2, seed=1) != first
