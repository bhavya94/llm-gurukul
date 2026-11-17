from medha.tokenizer import character_alphabet, character_ids, merges, train_bpe


def test_characters_by_hand_give_one_id_per_character():
    alphabet = character_alphabet(["a cat"])
    assert alphabet == [" ", "a", "c", "t"]
    assert character_ids("a cat", alphabet) == [1, 0, 2, 1, 3]


def test_bpe_merges_the_most_frequent_pair_first_and_builds_a_new_word_from_pieces():
    # The toy from the BPE paper: "es" and "st" are the most frequent pairs (in newest and widest, 6 + 3 times).
    words = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
    tokenizer = train_bpe(words, vocab_size=256 + 1 + 4)
    assert merges(tokenizer) == ["e s", "es t", "l o", "lo w"]
    assert tokenizer.encode("lowest").tokens == ["low", "est"]  # never seen, but built from two learned pieces


def test_bpe_gives_back_the_exact_text_in_fewer_ids_than_characters():
    stories = ["Once upon a time, Lily's café had 3 red balloons.",
               "Tom smiled. \"Can I have the red one?\" he asked."]
    tokenizer = train_bpe(stories * 20, vocab_size=320)
    for story in stories:
        ids = tokenizer.encode(story).ids
        assert tokenizer.decode(ids) == story
        assert len(ids) < len(story)
