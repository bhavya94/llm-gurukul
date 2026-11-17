"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging

from vak.data import load_split

# Each log line starts with the ms since the command started. Logs go to stderr; results print to stdout.
LOG_FORMAT = "%(relativeCreated)7.0fms  %(message)s"


def _data(args: argparse.Namespace) -> None:
    parts = load_split()
    in_both = set(parts.train) & set(parts.held_out)
    print(f"\n{len(parts.train):,} stories to train on, {len(parts.held_out):,} held out, {len(in_both)} in both")
    for number, story in enumerate(parts.train[:3], start=1):
        print(f"\nTraining story {number}, {len(story):,} characters:\n{story}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vak", description="vak (speech): A small GPT learns to write text, "
                                                             "one token at a time.")
    commands = parser.add_subparsers(title="commands", dest="command", metavar="<command>")
    data = commands.add_parser("data", help="download the stories, split them, and show three")
    data.set_defaults(run=_data)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger("medha").setLevel(logging.INFO)
    logging.getLogger("vak").setLevel(logging.INFO)
    if args.command is None:
        parser.print_help()
        return
    args.run(args)
