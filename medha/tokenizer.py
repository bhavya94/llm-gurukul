"""Tokens: the ids a model reads. First by hand, one id per character. Then BPE: the tokenizers library learns
pieces of words from the training text, and each piece gets one id."""

import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END = "<|endoftext|>"  # the one special token: it marks where a story ends


def character_alphabet(texts: list[str]) -> list[str]:
    """Every different character in the texts, sorted. This is the whole vocabulary of a character tokenizer."""
    characters = set()
    for text in texts:
        characters.update(text)
    return sorted(characters)


def character_ids(text: str, alphabet: list[str]) -> list[int]:
    """By hand: a character's id is its place in the alphabet."""
    ids = []
    for character in text:
        ids.append(alphabet.index(character))
    return ids


def train_bpe(texts: list[str], vocab_size: int) -> Tokenizer:
    """Byte-level BPE, as in GPT-2. The vocabulary starts with the 256 byte values and the end token. Then BPE finds
    the most frequent pair of neighbouring symbols in the texts and merges it into one new symbol. It repeats until
    the vocabulary has vocab_size entries, so it learns vocab_size - 257 merges."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # cut at spaces; Ġ marks a space
    tokenizer.decoder = decoders.ByteLevel()  # turns the byte symbols back into text
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[END], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def merges(tokenizer: Tokenizer) -> list[str]:
    """The merges BPE learned, in order. "e s" means that e followed by s became the new symbol es."""
    model = json.loads(tokenizer.to_str())["model"]
    learned = []
    for left, right in model["merges"]:
        learned.append(f"{left} {right}")
    return learned
