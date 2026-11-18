"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from medha.model import GPT, Config, Embedding, SelfAttention, attention
from medha.sampler import chances, generate
from medha.tokenizer import END, character_alphabet, character_ids, merges, train_bpe
from vak.counts import PairCounts
from vak.data import SEED, encode_stories, load_split
from vak.loss import held_out_loss, paired_difference
from vak.train import BATCH, join, sample, story_losses, train

log = logging.getLogger(__name__)

# Each log line starts with the ms since the command started. Logs go to stderr; results print to stdout.
LOG_FORMAT = "%(relativeCreated)7.0fms  %(message)s"
RUNS = Path(__file__).resolve().parent.parent / "runs"  # git-ignored: the files the commands make
TOKENIZER_FILE = RUNS / "tokenizer.json"
COUNTS_FILE = RUNS / "counts.pt"
STORIES_FILE = RUNS / "stories.pt"  # the training stream and the held-out stories, as ids
GPT_FILE = RUNS / "gpt.pt"
SENTENCE = "Once upon a time, there was a little girl named Lily."
VOCAB_SIZE = 2_000
GPT_CONFIG = Config(vocab_size=VOCAB_SIZE, context=128, width=128, heads=4, layers=4)
TRAIN_STEPS = 5_000
# Measured on this Mac (M1, 8 GB): training runs about 2.4 times faster on the GPU (MPS) than on the CPU, but
# writing one token at a time runs faster on the CPU, because each small step costs the GPU a fixed delay.
TRAIN_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WRITE_DEVICE = "cpu"
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
ORDERS = ["The dog saw the cat.", "The cat saw the dog."]  # the same tokens; dog and cat swap places


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


def _load_gpt() -> GPT:
    if not GPT_FILE.exists():
        raise SystemExit("vak: no runs/gpt.pt yet; run `uv run vak train` first")
    saved = torch.load(GPT_FILE)
    model = GPT(Config(**saved["config"]))
    model.load_state_dict(saved["weights"])
    return model.to(WRITE_DEVICE)


def _encoded(tokenizer: Tokenizer) -> tuple[torch.Tensor, list[list[int]]]:
    """The training stories as one stream of ids, and the held-out stories as ids, each framed by end tokens as the
    baseline scored them. Reading and encoding the stories takes about 15 s and 1.3 GB, so it runs once; later runs
    load runs/stories.pt, and the stories' text never stays in memory while the model trains."""
    if not STORIES_FILE.exists():
        parts = load_split()
        stream = join(encode_stories(tokenizer, parts.train))
        torch.save({"train": stream, "held_out": encode_stories(tokenizer, parts.held_out)}, STORIES_FILE)
        log.info("saved the ids to runs/%s", STORIES_FILE.name)
    saved = torch.load(STORIES_FILE)
    return saved["train"], saved["held_out"]


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


def _heads(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    torch.manual_seed(0)  # an untrained layer: random weights, fixed by the seed
    embedding = Embedding(tokenizer.get_vocab_size(), context=16, width=8)
    layer = SelfAttention(width=8, heads=2)
    orders = []
    for sentence in ORDERS:
        orders.append(torch.tensor([tokenizer.encode(sentence).ids]))
    log.info('"%s" as %d tokens: %s', ORDERS[0], orders[0].shape[1], orders[0][0].tolist())
    logging.getLogger("medha.model").setLevel(logging.DEBUG)  # the layer logs each shape
    with torch.no_grad():
        x = embedding(orders[0])
        log.info("token vectors plus position vectors: %s", tuple(x.shape))
        out = layer(x)
        log.info("after the output Linear: %s", tuple(out.shape))
    logging.getLogger("medha.model").setLevel(logging.INFO)

    print('\nThe output of the last token, ".", in each order:')
    gaps = []
    for label, vectors in [("without position", embedding.tokens), ("with position", embedding)]:
        last = []
        for sentence, ids in zip(ORDERS, orders):
            with torch.no_grad():
                last.append(layer(vectors(ids))[0, -1])
            print(f'  {label:17} "{sentence}"  {_pair(last[-1])}')
        gaps.append(float((last[0] - last[1]).abs().max()))
    print(f"\nThe largest difference: {gaps[0]:.6f} without position, {gaps[1]:.6f} with it.")


def _log_shape(part: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
    """A forward hook: PyTorch calls it each time the part it is attached to has run."""
    log.info("after %-10s %s", type(part).__name__, tuple(output.shape))


def _count(part: torch.nn.Module) -> int:
    total = 0
    for weights in part.parameters():
        total += weights.numel()
    return total


def _model(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG)
    end = tokenizer.token_to_id(END)
    ids = [end] + tokenizer.encode("Once upon a time").ids
    log.info('the end token, then "Once upon a time": %d ids, %s', len(ids), ids)
    parts = [model.embedding]
    parts.extend(model.blocks)
    parts.extend([model.norm, model.head])
    for part in parts:
        part.register_forward_hook(_log_shape)
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0]

    vocab, context, width, layers = GPT_CONFIG.vocab_size, GPT_CONFIG.context, GPT_CONFIG.width, GPT_CONFIG.layers
    block = model.blocks[0]
    rows = [
        ("token table, also the head", _count(model.embedding.tokens), f"{vocab:,} x {width}"),
        ("position table", _count(model.embedding.positions), f"{context} x {width}"),
        ("  attention in one block", _count(block.attention),
         f"{width} x {3 * width} + {3 * width} + {width} x {width} + {width}"),
        ("  MLP in one block", _count(block.mlp),
         f"{width} x {4 * width} + {4 * width} + {4 * width} x {width} + {width}"),
        ("  2 LayerNorms in one block", _count(block.norm1) + _count(block.norm2), f"2 x ({width} + {width})"),
        (f"{layers} blocks", _count(model.blocks), f"{layers} x {_count(block):,}"),
        ("last LayerNorm", _count(model.norm), f"{width} + {width}"),
        ("total", _count(model), ""),
    ]
    print(f"\nThe weights: {layers} blocks of {GPT_CONFIG.heads} heads, width {width}, a window of {context} tokens")
    for name, count, by_hand in rows:
        print(f"  {name:28}{count:>10,}   {by_hand}".rstrip())
    chances = torch.softmax(logits, dim=-1)
    losses = F.cross_entropy(logits[:-1], torch.tensor(ids[1:]), reduction="none")
    print(f'\nUntrained, the likeliest token after " time" has a chance of {float(chances[-1].max()):.5f}; '
          f"an even chance is 1 / {vocab:,} = {1 / vocab:.5f}.")
    print(f"The loss on the prompt's {len(losses)} next tokens is {float(losses.mean()):.3f}; an even guess scores "
          f"ln {vocab:,} = {math.log(vocab):.3f}.")


def _train(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    if not COUNTS_FILE.exists():
        raise SystemExit("vak: no runs/counts.pt yet; run `uv run vak counts` first")
    stream, held_out = _encoded(tokenizer)
    torch.manual_seed(SEED)
    model = GPT(GPT_CONFIG).to(TRAIN_DEVICE)
    tokens = TRAIN_STEPS * BATCH * GPT_CONFIG.context
    log.info("training %s weights on %s: %s steps of %d windows of %d tokens, %s tokens, %.1f times the %s in the "
             "stream", f"{_count(model):,}", TRAIN_DEVICE, f"{TRAIN_STEPS:,}", BATCH, GPT_CONFIG.context,
             f"{tokens:,}", tokens / len(stream), f"{len(stream):,}")
    started = time.time()
    train(model, stream, TRAIN_STEPS, tokenizer, SEED)
    seconds = time.time() - started
    log.info("trained in %.0f s: %s tokens/s", seconds, f"{tokens / seconds:,.0f}")
    model.to(WRITE_DEVICE)
    torch.save({"config": asdict(GPT_CONFIG), "weights": model.state_dict()}, GPT_FILE)
    log.info("saved the GPT to runs/%s; scoring the held-out stories on the %s", GPT_FILE.name, WRITE_DEVICE)

    counts = PairCounts.load(COUNTS_FILE)
    gpt_losses = []
    count_losses = []
    for ids in held_out:
        gpt_losses.append(story_losses(model, ids))
        count_losses.append(counts.losses(ids))
    gpt = held_out_loss(gpt_losses, SEED)
    baseline = held_out_loss(count_losses, SEED)
    gap = paired_difference(gpt_losses, count_losses, SEED)
    print(f"\nHeld-out loss per token, over {gpt.tokens:,} tokens in {gpt.stories:,} stories, with 95% intervals:")
    print(f"  counting pairs        {baseline.mean:.3f}   {baseline.low:.3f} to {baseline.high:.3f}")
    print(f"  the GPT               {gpt.mean:.3f}   {gpt.low:.3f} to {gpt.high:.3f}")
    print(f"  the GPT minus pairs  {gap.mean:+.3f}   {gap.low:+.3f} to {gap.high:+.3f}, drawing the same stories")
    print("\nThree stories from the GPT, at temperature 1:")
    for seed in range(3):
        print(f"\n{sample(model, tokenizer, 100, seed)}")


def _write(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer()
    if args.model == "gpt":
        model = _load_gpt()
    elif COUNTS_FILE.exists():
        model = PairCounts.load(COUNTS_FILE)
    else:
        raise SystemExit("vak: no runs/counts.pt yet; run `uv run vak counts` first")
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
    heads = commands.add_parser("heads", help="shapes through many heads; with position, order changes the output")
    heads.set_defaults(run=_heads)
    model = commands.add_parser("model", help="shapes through the whole GPT, and its weights part by part")
    model.set_defaults(run=_model)
    training = commands.add_parser("train", help="train the GPT, then score it against the pair counts")
    training.set_defaults(run=_train)
    write = commands.add_parser("write", help="the chances under each sampler setting, then the decode loop writes")
    write.add_argument("--model", choices=["counts", "gpt"], default="counts",
                       help="the pair counts from step 4, or the GPT from step 9 (default: %(default)s)")
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
