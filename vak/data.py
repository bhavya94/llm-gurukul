"""The stories: one file of TinyStories, with repeats dropped, split by a seed into stories to train on and stories
held out for testing. No model sees a held-out story until it is tested on it."""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # git-ignored: the download is never committed
DATASET = "roneneldan/TinyStories"  # short stories in simple English; licence CDLA-Sharing-1.0
FILE = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"  # the first of its four training files, 249 MB
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned: the dataset's commit of 2024-08-12
TRAIN_STORIES = 50_000
HELD_OUT_STORIES = 2_000
SEED = 0


@dataclass(frozen=True)
class Split:
    train: list[str]  # the stories every model learns from
    held_out: list[str]  # the stories every model is tested on


def download() -> Path:
    """The file, downloaded once into data/. Later runs read it from there."""
    path = DATA_DIR / FILE
    if not path.exists():
        log.info("downloading %s from %s: 249 MB, first run only", FILE, DATASET)
        hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION, local_dir=DATA_DIR)
    return path


def unique_stories(stories: list[str]) -> list[str]:
    """Each story once, in file order, with blank ones dropped. A story that is in the file twice could otherwise
    land in both parts of the split, and the test would then score text the model trained on."""
    seen = set()
    unique = []
    for story in stories:
        text = story.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def split(stories: list[str], n_train: int, n_held_out: int, seed: int) -> Split:
    """Pick n_train + n_held_out different stories at random. The seed fixes which ones. The first n_train picks
    are for training, the rest are held out."""
    picked = random.Random(seed).sample(range(len(stories)), n_train + n_held_out)
    train = [stories[i] for i in picked[:n_train]]
    held_out = [stories[i] for i in picked[n_train:]]
    return Split(train, held_out)


def load_split() -> Split:
    """The split every command uses: download once, read, drop repeats, then split with SEED."""
    path = download()
    stories = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    log.info("read %s stories from %s", f"{len(stories):,}", path.name)
    unique = unique_stories(stories)
    log.info("dropped %s blank or repeated stories: %s unique stories left",
             f"{len(stories) - len(unique):,}", f"{len(unique):,}")
    parts = split(unique, TRAIN_STORIES, HELD_OUT_STORIES, SEED)
    log.info("seed %d picks %s stories to train on and %s to hold out", SEED,
             f"{len(parts.train):,}", f"{len(parts.held_out):,}")
    return parts
