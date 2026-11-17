"""`vak`: the one command for this ability. Each concept the stack adds becomes a subcommand here."""

import argparse
import logging

# Each log line starts with the ms since the command started. Logs go to stderr; results print to stdout.
LOG_FORMAT = "%(relativeCreated)7.0fms  %(message)s"


def main() -> None:
    parser = argparse.ArgumentParser(prog="vak", description="vak (speech): A small GPT learns to write text, "
                                                             "one token at a time.")
    parser.add_subparsers(title="commands", dest="command", metavar="<command>")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger("medha").setLevel(logging.INFO)
    logging.getLogger("vak").setLevel(logging.INFO)
    if args.command is None:
        parser.print_help()
        return
    args.run(args)
