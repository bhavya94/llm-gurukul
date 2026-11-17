# vak: speech

A small GPT learns to write text, one token at a time.

A language model reads text as tokens, small pieces of text, and predicts the next token. Here we build one from
the ground up and train it on one Mac. Every later ability starts from this model.

The goal: `uv run vak write --prompt "Once upon a time"` writes short story sentences. On stories it has never
seen, the trained GPT must also score a lower loss than a model that only counts which token follows which.

## Steps

One pull request per step. Each adds one command or one notebook section, and one test.

| Step | What it adds |
|---|---|
| 1 | This structure: the `vak` command |
| 2 | The stories: a slice of TinyStories, with some stories held out for testing |
| 3 | Tokens: characters first, then BPE pieces |
| 4 | Count pairs: the smallest language model, and its loss on held-out stories |
| 5 | Sample: temperature, top-k, top-p, and the loop that writes text |
| 6 | One attention head on three tokens |
| 7 | Many heads, and position |
| 8 | The block and the stack: the model file, `medha/model.py` |
| 9 | Train: the loss falls and beats the counting model |
| 10 | The KV cache: the same tokens, with less work |

## Run it

From the top of the repo:

```console
uv sync
uv run vak --help
```
