"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from medha.model import attention
from medha.sampler import chances, generate
from medha.tokenizer import END, character_alphabet, character_ids, merges, train_bpe
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
SETTINGS = [  # (label, temperature, top_k, top_p): the columns of the table vak write prints
    ("T 0", 0.0, None, None),
    ("T 0.5", 0.5, None, None),
    ("T 1", 1.0, None, None),
    ("T 2", 2.0, None, None),
    ("top-k 3", 1.0, 3, None),
    ("top-p 0.9", 1.0, None, 0.9),
]
# The toy for vak attention: three tokens. Each vector has two numbers, read as "water" and "money".
TOY_TOKENS = ["The", "river", "bank"]
TOY_QUERIES = [[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]]
TOY_KEYS = [[0.0, 0.0], [2.0, 0.0], [0.5, 0.5]]
TOY_VALUES = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
LOAN_KEY, LOAN_VALUE = [0.0, 2.0], [0.0, 1.0]  # "loan", to put where "river" was


def _whole_number(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, not {value}")
    return value


def _temperature(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {value}")
    return value


def _share(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be above 0 and at most 1, not {value}")
    return value


def _piece(tokenizer: Tokenizer, token: int) -> str:
    """A token's text in quotes, so a leading space shows, with a new line as \\n; the end token by its name."""
    if token == tokenizer.token_to_id(END):
        return END
    return '"' + tokenizer.decode([token]).replace("\n", "\\n") + '"'


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


def _chance(value: float) -> str:
    """A chance to three decimals. One above 0 that would round to 0.000 shows as <0.001, so it is not read as 0."""
    if 0 < value < 0.0005:
        return "<0.001"
    return f"{value:.3f}"


def _pair(vector: list[float] | torch.Tensor, decimals: int = 3) -> str:
    numbers = []
    for number in vector:
        numbers.append(f"{float(number):.{decimals}f}")
    return "(" + ", ".join(numbers) + ")"


def _matrix(title: str, matrix: torch.Tensor) -> None:
    """A 3 x 3 table: one row per token that reads, one column per token it reads."""
    header = " " * 10
    for token in TOY_TOKENS:
        header += token.rjust(8)
    print(f"\n{title}\n{header}")
    for token, row in zip(TOY_TOKENS, matrix.tolist()):
        line = "  " + token.ljust(8)
        for number in row:
            line += f"{number:8.3f}"
        print(line)


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
        piece = _piece(tokenizer, token)
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


def _attention(args: argparse.Namespace) -> None:
    q, k, v = torch.tensor(TOY_QUERIES), torch.tensor(TOY_KEYS), torch.tensor(TOY_VALUES)
    print('\nThree tokens. Each query, key and value has two numbers, read as "water" and "money".')
    print("  token   query       key         value")
    for token, query, key, value in zip(TOY_TOKENS, TOY_QUERIES, TOY_KEYS, TOY_VALUES):
        print(f"  {token:8}{_pair(query, 1):12}{_pair(key, 1):12}{_pair(value, 1)}")
    steps = attention(q, k, v)
    _matrix("1. Scores: each token's query times each token's key", steps.scores)
    _matrix(f"2. Scaled: divided by {math.sqrt(2):.3f}, the square root of the head size, 2", steps.scaled)
    _matrix("3. Masked: minus infinity wherever a token would read a later token", steps.masked)
    _matrix("4. Weights: softmax, row by row; each row adds up to 1", steps.weights)
    print("\n5. Out: each token's mix of the values, by its row of weights")
    for token, out in zip(TOY_TOKENS, steps.out):
        print(f"  {token:8}{_pair(out)}")

    k[1], v[1] = torch.tensor(LOAN_KEY), torch.tensor(LOAN_VALUE)
    loan = attention(q, k, v)
    print(f'\nPut "loan" where "river" was, with key {_pair(LOAN_KEY, 1)} and value {_pair(LOAN_VALUE, 1)}.')
    print(f"The bank row keeps its weights, {_pair(loan.weights[2])}, and its out becomes {_pair(loan.out[2])}.")


def _write(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    if not COUNTS_FILE.exists():
        raise SystemExit("vak: no runs/counts.pt yet; run `uv run vak counts` first")
    model = PairCounts.load(COUNTS_FILE)
    end = tokenizer.token_to_id(END)
    prompt = tokenizer.encode(args.prompt).ids
    ids = [end] + prompt  # every training story starts after an end token, so writing does too
    log.info('the prompt: the end token, then "%s" as %d tokens: %s', args.prompt, len(prompt), prompt)

    logits = model.next_logits(ids)
    columns = []
    for label, temperature, top_k, top_p in SETTINGS:
        columns.append(chances(logits, temperature, top_k, top_p))
    print(f"\nThe chance of each next token after {_piece(tokenizer, ids[-1])}, under each setting:")
    header = "  " + "token".ljust(14)
    for label, temperature, top_k, top_p in SETTINGS:
        header += label.rjust(10)
    print(header)
    top = torch.topk(logits, 8).indices
    for token in top.tolist():
        row = "  " + _piece(tokenizer, token).ljust(14)
        for probs in columns:
            row += _chance(float(probs[token])).rjust(10)
        print(row)
    others = torch.ones(len(logits), dtype=torch.bool)
    others[top] = False  # every token below the table
    row = "  " + "all others".ljust(14)
    for probs in columns:
        row += _chance(float(probs[others].sum())).rjust(10)
    print(row)

    top_k = "off" if args.top_k is None else args.top_k
    top_p = "off" if args.top_p is None else args.top_p
    log.info("decode loop: temperature %g, top-k %s, top-p %s, at most %d new tokens, seed %d",
             args.temperature, top_k, top_p, args.tokens, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    steps = generate(model, ids, args.tokens, end, generator, args.temperature, args.top_k, args.top_p)
    written = []
    for number, step in enumerate(steps, start=1):
        log.info("step %2d: %-14s chance %s", number, _piece(tokenizer, step.token), _chance(step.chance))
        if step.token != end:
            written.append(step.token)
    if steps[-1].token == end:
        log.info("stop: the model picked the end token")
    else:
        log.info("stop: %d new tokens, the limit", args.tokens)
    print(f"\n{tokenizer.decode(prompt + written)}")


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
    attend = commands.add_parser("attention", help="one attention head on three tokens, one step at a time")
    attend.set_defaults(run=_attention)
    write = commands.add_parser("write", help="the chances under each sampler setting, then the decode loop writes")
    write.add_argument("--prompt", default="Once upon a time", help="the text to continue (default: %(default)s)")
    write.add_argument("--tokens", type=_whole_number, default=40, help="the most new tokens (default: %(default)s)")
    write.add_argument("--temperature", type=_temperature, default=1.0,
                       help="0 is greedy; below 1 sharpens, above 1 flattens (default: %(default)s)")
    write.add_argument("--top-k", type=_whole_number, help="keep only the k likeliest tokens")
    write.add_argument("--top-p", type=_share, help="keep the fewest tokens whose chances add up to p")
    write.add_argument("--seed", type=int, default=0, help="fixes the random draws (default: %(default)s)")
    write.set_defaults(run=_write)
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
