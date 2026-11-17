# vak: speech

A small GPT learns to write text, one token at a time.

A language model reads text as tokens, small pieces of text, and predicts the next token. Here we build one from
the ground up and train it on one Mac. Every later ability starts from this model.

The goal: `uv run vak write --prompt "Once upon a time"` writes short story sentences. On stories it has never
seen, the trained GPT must also score a lower loss than a model that only counts which token follows which.

## Steps

One pull request per step. From step 2 on, each adds one command, one notebook and one test.

| Step | What it adds | Notebook |
|---|---|---|
| 1 | This structure: the `vak` command | none |
| 2 | The stories: a slice of TinyStories, with some stories held out for testing | [02-data.ipynb](02-data.ipynb) |
| 3 | Tokens: characters first, then BPE pieces | |
| 4 | Count pairs: the smallest language model, and its loss on held-out stories | |
| 5 | Sample: temperature, top-k, top-p, and the loop that writes text | |
| 6 | One attention head on three tokens | |
| 7 | Many heads, and position | |
| 8 | The block and the stack: the model file, `medha/model.py` | |
| 9 | Train: the loss falls and beats the counting model | |
| 10 | The KV cache: the same tokens, with less work | |

## Run it

From the top of the repo:

```console
uv sync
uv run vak --help
```

## The notebooks

Each step from step 2 on has its own notebook, linked in the table above. A notebook takes its step's idea at its
smallest size, small enough to check by hand. Each section calls the same functions as the command, then says what
the output shows. A test runs every notebook and checks that each "What you see" note still matches what the cells
print.

To open them, add this project's Python to Jupyter as a kernel, once:

```console
uv run python -m ipykernel install --user --name llm-gurukul
```

Then open a notebook in Jupyter. Each notebook asks for the `llm-gurukul` kernel, so Jupyter starts this project's
Python, with its packages.

`02-data.ipynb` reads the downloaded stories, so run `uv run vak data` once before you open it. Until the file is
there, the test skips that notebook. The other notebooks work on small cases typed into their cells and need no
download.

## The stories

`vak data` downloads one file of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): short
stories in simple English, written by GPT-3.5 and GPT-4. Its licence is CDLA-Sharing-1.0. The file is the first of
the four training files: 249 MB and 529,930 stories. The first run downloads it into `data/`, which git ignores.
The download is pinned to the dataset's commit of 2024-08-12 (`REVISION` in `vak/data.py`), so a later upload
cannot change the stories or any number below.

```console
$ uv run vak data
    836ms  read 529,930 stories from train-00000-of-00004-2d5a1467fff1081b.parquet
   1263ms  dropped 20,305 blank or repeated stories: 509,625 unique stories left
   1283ms  seed 0 picks 50,000 stories to train on and 2,000 to hold out

50,000 stories to train on, 2,000 held out, 0 in both

Training story 1, 638 characters:
Once upon a time, there was a bird. The bird was very happy because it could fly high in the sky. One day, the bird saw a fancy yard with lots of pretty flowers. The bird wanted to see the yard up close, so it flew down to the ground.
...
```

Two more stories follow.

- **Held out** means that no model learns from these 2,000 stories. They are the test. A model's loss on them shows
  how well it predicts text it has never seen.
- **The seed** fixes which stories land in each part. Every command and every model sees the same split. The seed,
  0, and the sizes are named values at the top of `vak/data.py`.
- **Repeats:** the file holds about 20,000 stories twice. We keep one copy of each. Without that step, 12 held-out
  stories would also be training stories, and the test would score text the model learned from.
- The first run takes about 16 s, mostly the download. Later runs take about 1.3 s. Reading the whole file uses
  about 1.3 GB of memory for a moment.
