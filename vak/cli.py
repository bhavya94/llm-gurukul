"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging
import os
from pathlib import Path

from medha.tokenizer import character_alphabet, character_ids, merges, train_bpe
from vak.data import load_split

log = logging.getLogger(__name__)

# Each log line starts with the ms since the command started. Logs go to stderr; results print to stdout.
LOG_FORMAT = "%(relativeCreated)7.0fms  %(message)s"
RUNS = Path(__file__).resolve().parent.parent / "runs"  # git-ignored: the files the commands make
TOKENIZER_FILE = RUNS / "tokenizer.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="vak", description="vak (speech): A small GPT learns to write text, "
                                                             "one token at a time.")
    commands = parser.add_subparsers(title="commands", dest="command", metavar="<command>")
    data = commands.add_parser("data", help="download the stories, split them, and show three")
    data.set_defaults(run=_data)
    tokens = commands.add_parser("tokens", help="one sentence as characters, then as BPE pieces; saves the tokenizer")
    tokens.set_defaults(run=_tokens)
    args = parser.parse_args()

    # One thread for the tokenizers library: on this Mac its thread pool once deadlocked PyTorch on the GPU (MPS).
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger("medha").setLevel(logging.INFO)
    logging.getLogger("vak").setLevel(logging.INFO)
    if args.command is None:
        parser.print_help()
        return
    args.run(args)
