"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from medha.tokenizer import character_alphabet, character_ids, merges, train_bpe
from vak.counts import PairCounts
from vak.data import SEED, encode_stories, load_split
from vak.loss import held_out_loss

log = logging.getLogger(__name__)

# Each log line starts with the ms since the command started. Logs go to stderr; results print to stdout.
LOG_FORMAT = "%(relativeCreated)7.0fms  %(message)s"
RUNS = Path(__file__).resolve().parent.parent / "runs"  # git-ignored: the files the commands make
TOKENIZER_FILE = RUNS / "tokenizer.json"
COUNTS_FILE = RUNS / "counts.pt"
SENTENCE = "Once upon a time, there was a little girl named Lily."
VOCAB_SIZE = 2_000


def _aligned(pieces: list[str], ids: list[int]) -> str:
    """Each piece above its id, in rows about 100 characters wide."""
    lines = []
    top, bottom = "", ""
    for piece, token in zip(pieces, ids):
        width = max(len(piece), len(str(token))) + 2
        if len(top) + width > 100:
            lines += [top.rstrip(), bottom.rstrip(), ""]
            top, bottom = "", ""
        top += piece.ljust(width)
        bottom += str(token).ljust(width)
    lines += [top.rstrip(), bottom.rstrip()]
    return "\n".join(lines)


def _load_tokenizer() -> Tokenizer:
    if not TOKENIZER_FILE.exists():
        raise SystemExit("vak: no runs/tokenizer.json yet; run `uv run vak tokens` first")
    return Tokenizer.from_file(str(TOKENIZER_FILE))


def _data(args: argparse.Namespace) -> None:
    parts = load_split()
    in_both = set(parts.train) & set(parts.held_out)
    print(f"\n{len(parts.train):,} stories to train on, {len(parts.held_out):,} held out, {len(in_both)} in both")
    for number, story in enumerate(parts.train[:3], start=1):
        print(f"\nTraining story {number}, {len(story):,} characters:\n{story}")


def _tokens(args: argparse.Namespace) -> None:
    parts = load_split()
    alphabet = character_alphabet(parts.train)
    log.info("the training stories use %d different characters", len(alphabet))
    by_hand = character_ids(SENTENCE, alphabet)
    log.info("training BPE on %s stories, up to %s ids", f"{len(parts.train):,}", f"{VOCAB_SIZE:,}")
    tokenizer = train_bpe(parts.train, VOCAB_SIZE)
    RUNS.mkdir(exist_ok=True)
    tokenizer.save(str(TOKENIZER_FILE))
    log.info("saved the tokenizer to runs/%s", TOKENIZER_FILE.name)
    learned = merges(tokenizer)
    encoding = tokenizer.encode(SENTENCE)

    print(f'\nThe sentence: "{SENTENCE}"')
    print(f"\nAs characters: {len(by_hand)} ids, from an alphabet of {len(alphabet)} characters (␣ is a space)")
    print(_aligned(list(SENTENCE.replace(" ", "␣")), by_hand))
    print(f"\nAs BPE pieces: {len(encoding.ids)} ids, {len(by_hand) / len(encoding.ids):.1f} times fewer, from a "
          f"vocabulary of {tokenizer.get_vocab_size():,} (Ġ is a space at the start of a piece)")
    print(_aligned(encoding.tokens, encoding.ids))
    print(f"\nThe vocabulary: 256 bytes + 1 end token + {len(learned):,} merges = {tokenizer.get_vocab_size():,} ids")
    print(f"The first five merges: {', '.join(learned[:5])}")
    print(f'The BPE ids decoded back: "{tokenizer.decode(encoding.ids)}"')


def _counts(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    parts = load_split()
    train = encode_stories(tokenizer, parts.train)
    model = PairCounts.from_stories(train, vocab_size)
    log.info("counted %s pairs of neighbouring tokens in %s stories: %s different pairs, of %s possible",
             f"{int(model.counts.sum()):,}", f"{len(train):,}", f"{int((model.counts > 0).sum()):,}",
             f"{vocab_size * vocab_size:,}")
    model.save(COUNTS_FILE)
    log.info("saved the counts to runs/%s", COUNTS_FILE.name)

    the = tokenizer.token_to_id("Ġthe")  # " the", with its space
    seen = int(model.counts[the].sum())
    print(f'\nAfter " the" ({seen:,} times in training), the 10 likeliest next tokens:')
    print(f"  token        count   chance = (count + 1) / ({seen:,} + {vocab_size:,})")
    top_ten = 0.0
    for token in torch.topk(model.counts[the], 10).indices.tolist():
        piece = '"' + tokenizer.decode([token]) + '"'
        count = int(model.counts[the, token])
        chance = float(model.chances[the, token])
        top_ten += chance
        print(f"  {piece:10} {count:>7,}   {chance:.4f}")
    print(f"  the other {vocab_size - 10:,} tokens share {1 - top_ten:.4f}")

    held_out = encode_stories(tokenizer, parts.held_out)
    losses = []
    unseen = 0
    for ids in held_out:
        losses.append(model.losses(ids))
        unseen += model.unseen(ids)
    loss = held_out_loss(losses, SEED)
    log.info("%s of the %s held-out pairs never appear in training; add-one gives each a small chance, not 0",
             f"{unseen:,}", f"{loss.tokens:,}")
    print(f"\nHeld-out loss: {loss.mean:.3f} per token, 95% interval {loss.low:.3f} to {loss.high:.3f}, "
          f"over {loss.tokens:,} tokens in {loss.stories:,} stories")
    print(f"For scale, an even guess over all {vocab_size:,} tokens scores ln {vocab_size:,} = "
          f"{math.log(vocab_size):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vak", description="vak (speech): A small GPT learns to write text, "
                                                             "one token at a time.")
    commands = parser.add_subparsers(title="commands", dest="command", metavar="<command>")
    data = commands.add_parser("data", help="download the stories, split them, and show three")
    data.set_defaults(run=_data)
    tokens = commands.add_parser("tokens", help="one sentence as characters, then as BPE pieces; saves the tokenizer")
    tokens.set_defaults(run=_tokens)
    counts = commands.add_parser("counts", help="count which token follows which; its chances and held-out loss")
    counts.set_defaults(run=_counts)
    args = parser.parse_args()

    # One thread for the tokenizers library: on this Mac its thread pool once deadlocked PyTorch on the GPU (MPS).
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.stdout.reconfigure(line_buffering=True)  # results and logs stay in order, even through a pipe
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger("medha").setLevel(logging.INFO)
    logging.getLogger("vak").setLevel(logging.INFO)
    if args.command is None:
        parser.print_help()
        return
    args.run(args)
