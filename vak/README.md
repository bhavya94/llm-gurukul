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
| 3 | Tokens: characters first, then BPE pieces | [03-tokens.ipynb](03-tokens.ipynb) |
| 4 | Count pairs: the smallest language model, and its loss on held-out stories | [04-counts.ipynb](04-counts.ipynb) |
| 5 | Sample: temperature, top-k, top-p, and the loop that writes text | [05-sample.ipynb](05-sample.ipynb) |
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

## Tokens

A model reads ids, not text. A **token** is the piece of text behind one id. `vak tokens` turns one sentence into
ids twice: first one id per character, by hand, then with BPE. It saves the BPE tokenizer to `runs/tokenizer.json`,
and the later steps load it from there.

```console
$ uv run vak tokens
   1022ms  read 529,930 stories from train-00000-of-00004-2d5a1467fff1081b.parquet
   1448ms  dropped 20,305 blank or repeated stories: 509,625 unique stories left
   1463ms  seed 0 picks 50,000 stories to train on and 2,000 to hold out
   1685ms  the training stories use 107 different characters
   1685ms  training BPE on 50,000 stories, up to 2,000 ids
   8077ms  saved the tokenizer to runs/tokenizer.json

The sentence: "Once upon a time, there was a little girl named Lily."

As characters: 53 ids, from an alphabet of 107 characters (␣ is a space)
O   n   c   e   ␣  u   p   o   n   ␣  a   ␣  t   i   m   e   ,   ␣  t   h   e   r   e   ␣  w   a
43  72  61  63  1  79  74  73  72  1  59  1  78  67  71  63  11  1  78  66  63  76  63  1  81  59

s   ␣  a   ␣  l   i   t   t   l   e   ␣  g   i   r   l   ␣  n   a   m   e   d   ␣  L   i   l   y
77  1  59  1  70  67  78  78  70  63  1  65  67  76  70  1  72  59  71  63  62  1  40  67  70  83

.
13

As BPE pieces: 13 ids, 4.1 times fewer, from a vocabulary of 2,000 (Ġ is a space at the start of a piece)
Once  Ġupon  Ġa   Ġtime  ,   Ġthere  Ġwas  Ġa   Ġlittle  Ġgirl  Ġnamed  ĠLily  .
432   445    259  396    12  398     282   259  397      448    498     362    14

The vocabulary: 256 bytes + 1 end token + 1,743 merges = 2,000 ids
The first five merges: h e, Ġ t, Ġ a, Ġ s, Ġ w
The BPE ids decoded back: "Once upon a time, there was a little girl named Lily."
```

- **Characters:** the alphabet is every different character in the training stories, 107 of them. A character's id
  is its place in the sorted alphabet. The sentence takes 53 ids.
- **BPE** (byte-pair encoding) starts from the 256 byte values and the end token. Then it merges the most frequent
  pair of neighbouring symbols into a new symbol, again and again. 1,743 merges fill a vocabulary of 2,000. The
  [`tokenizers`](https://github.com/huggingface/tokenizers) library does the work, in byte-level BPE, as GPT-2 does.
- **Ġ** marks a space at the start of a piece: ` upon`, with its space, is one token, id 445. The first merges join
  h and e, then a space with t, a, s and w. Those start common words such as the, a, she and was.
- The same sentence takes 13 BPE ids instead of 53, 4.1 times fewer. A model reads a fixed number of ids at once, so
  with BPE more of the story fits.
- The 13 ids decode back to the exact sentence.
- The run takes about 8 s, and training BPE takes about 6.4 s of it. The library runs on one thread here
  (`TOKENIZERS_PARALLELISM=false`); on several threads it takes 1.7 s and learns the same merges.

## Counting pairs: the baseline

The smallest language model counts which token follows which in the training stories. The chance of token b after
token a is the count of the pair (a, b) over the count of a. It reads only the last token. `vak counts` counts every
pair in the 50,000 training stories, saves the table to `runs/counts.pt`, and scores the 2,000 held-out stories.

```console
$ uv run vak counts
   2285ms  read 529,930 stories from train-00000-of-00004-2d5a1467fff1081b.parquet
   2804ms  dropped 20,305 blank or repeated stories: 509,625 unique stories left
   2821ms  seed 0 picks 50,000 stories to train on and 2,000 to hold out
  15956ms  counted 12,560,549 pairs of neighbouring tokens in 50,000 stories: 268,373 different pairs, of 4,000,000 possible
  15974ms  saved the counts to runs/counts.pt

After " the" (392,466 times in training), the 10 likeliest next tokens:
  token        count   chance = (count + 1) / (392,466 + 2,000)
  " park"     14,198   0.0360
  " little"    6,229   0.0158
  " p"         5,905   0.0150
  " c"         5,492   0.0139
  " sky"       5,341   0.0135
  " sun"       5,212   0.0132
  " bird"      5,043   0.0128
  " ground"    4,487   0.0114
  " tree"      4,428   0.0112
  " garden"    4,368   0.0111
  the other 1,990 tokens share 0.8461
  19296ms  4,631 of the 502,358 held-out pairs never appear in training; add-one gives each a small chance, not 0

Held-out loss: 3.706 per token, 95% interval 3.695 to 3.718, over 502,358 tokens in 2,000 stories
For scale, an even guess over all 2,000 tokens scores ln 2,000 = 7.601
```

- **The chances after " the":** " park" follows " the" 14,198 times in 392,466. **Add-one smoothing** adds 1 to
  every count first, so the chance of " park" is (14,198 + 1) / (392,466 + 2,000) = 0.0360. The ten likeliest tokens
  share only 15% of the chance: after " the", this model is far from sure.
- " p" and " c" start longer words that have no piece of their own: " power" is " p" then "ower", and " cave" is
  " c" then "ave".
- **The loss** of one token is minus the natural log of the chance the model gave the true next token. The
  **held-out loss** is its mean over all 502,358 tokens of the held-out stories, end tokens included. Lower is better.
- **The baseline** scores 3.706 per token, with a 95% interval from 3.695 to 3.718. The interval comes from a
  bootstrap over whole stories (`scipy.stats.bootstrap`), because the tokens of one story depend on each other.
  An even guess over 2,000 tokens scores ln 2,000 = 7.601, so counting pairs halves the loss.
- **Smoothing:** 4,631 held-out pairs, about 1%, never appear in training. Without smoothing, each would get a
  chance of 0 and an infinite loss. Add-one is the textbook rule, and we keep it. It is not the best value here:
  adding 0.1 instead of 1 gives 3.571. We did not switch, because a value chosen by the held-out score would tune the
  baseline on its own test. The GPT must beat 3.706, and it should beat 3.571 too.
- The table holds 2,000 × 2,000 counts, but only 268,373 of those pairs ever occur.
- This model sees one token and nothing before it. The next step shows the text that comes out of that.
- The run takes about 20 s, and encoding and counting the stories take about 13 s of it. It uses about 1.3 GB of
  memory at its peak.

## Sampling and the decode loop

A model gives one score, a **logit**, to every token, and softmax turns the scores into chances. The **sampler**
reshapes the chances, then draws one token. It applies three settings, in this order:

1. **Temperature** divides every logit. Below 1 it sharpens the chances toward the top token, and above 1 it
   flattens them. At 0 the top token always wins: that is **greedy** decoding.
2. **Top-k** keeps only the k likeliest tokens.
3. **Top-p** keeps the fewest tokens whose chances add up to p.

The kept chances are then rescaled to add up to 1. The **decode loop** writes text: score the next token, pick one,
append it, and go again. It stops when it picks the end token, or when it reaches its limit of new tokens.
`medha/sampler.py` holds both, so the GPT will use the same code.

`vak write` loads the pair counts from step 4. It starts after the end token, as every training story does, then
reads the prompt. It prints the chances of the next token under six settings. Then the loop writes, and the log
shows each pick with its chance.

```console
$ uv run vak write --tokens 20
   1856ms  the prompt: the end token, then "Once upon a time" as 4 tokens: [432, 445, 259, 396]

The chance of each next token after " time", under each setting:
  token                T 0     T 0.5       T 1       T 2   top-k 3 top-p 0.9
  ","                1.000     0.911     0.556     0.054     0.727     0.617
  " there"           0.000     0.065     0.148     0.028     0.194     0.164
  "."                0.000     0.011     0.061     0.018     0.080     0.068
  " to"              0.000     0.011     0.060     0.018     0.000     0.067
  " for"             0.000     0.001     0.021     0.011     0.000     0.024
  " and"             0.000    <0.001     0.013     0.008     0.000     0.014
  " she"             0.000    <0.001     0.008     0.007     0.000     0.009
  " he"              0.000    <0.001     0.007     0.006     0.000     0.008
  all others         0.000     0.001     0.125     0.850     0.000     0.029
   1875ms  decode loop: temperature 1, top-k off, top-p off, at most 20 new tokens, seed 0
   1878ms  step  1: ","            chance 0.556
   1878ms  step  2: " and"         chance 0.059
   1879ms  step  3: " asked"       chance 0.008
   1879ms  step  4: " Lily"        chance 0.017
   1879ms  step  5: " sad"         chance <0.001
   1879ms  step  6: " and"         chance 0.247
   1879ms  step  7: " pick"        chance 0.001
   1879ms  step  8: " flowers"     chance 0.022
   1879ms  step  9: " and"         chance 0.161
   1879ms  step 10: " cup"         chance <0.001
   1879ms  step 11: " pen"         chance <0.001
   1879ms  step 12: ";"            chance <0.001
   1879ms  step 13: " she"         chance 0.027
   1879ms  step 14: " was"         chance 0.117
   1879ms  step 15: " time"        chance 0.011
   1879ms  step 16: ","            chance 0.556
   1879ms  step 17: " she"         chance 0.040
   1879ms  step 18: " asked"       chance 0.021
   1879ms  step 19: "."            chance 0.168
   1879ms  step 20: "\n"           chance 0.152
   1879ms  stop: 20 new tokens, the limit

Once upon a time, and asked Lily sad and pick flowers and cup pen; she was time, she asked.
```

- **The table:** after " time", the comma has 0.556 at T 1. T 0.5 raises it to 0.911. T 2 drops it to 0.054, and the
  1,992 tokens below the table gain: together they hold 0.850. Top-k 3 keeps three tokens, and top-p 0.9 keeps 13.
- **The text is poor, as expected.** Each pick depends only on the token before it, so no sentence forms.
- **Many picks had almost no chance:** " sad", " cup", " pen" and ";". The log shows `<0.001` for a chance above 0
  that rounds to 0.000. These tiny chances come from the counts that add-one added: every pair gets a count of at
  least 1. For the median token, those added counts hold about half of the next-token chance, a long tail of
  unlikely tokens. At T 1 with no cut, the loop often draws from that tail.
- Top-p 0.9 does not fix this here: for the median token, it keeps 1,597 of the 2,000 tokens.

Top-k cuts the tail. `tail -n +14` skips the table:

```console
$ uv run vak write --top-k 5 2>/dev/null | tail -n +14
Once upon a time, and said, "You can make the little girl named Tim was so happy.







"I love. She said.
"Look, there was a time
```

Each pick is now one of the five likeliest tokens. Neighbouring words fit, as in "the little girl named Tim was so
happy", but still no sentence forms, and the new lines pile up.

Greedy is no cure:

```console
$ uv run vak write --temperature 0 --tokens 10 2>&1 | grep -E "step|stop"
   1076ms  step  1: ","            chance 1.000
   1076ms  step  2: " ""           chance 1.000
   1076ms  step  3: "I"            chance 1.000
   1076ms  step  4: "'m"           chance 1.000
   1076ms  step  5: " sorry"       chance 1.000
   1076ms  step  6: "."            chance 1.000
   1076ms  step  7: "\n"           chance 1.000
   1076ms  step  8: "\n"           chance 1.000
   1076ms  step  9: "\n"           chance 1.000
   1076ms  step 10: "\n"           chance 1.000
   1076ms  stop: 10 new tokens, the limit
```

After a new line, another new line is the likeliest token, so greedy never leaves that loop. This is one reason
why text generation samples instead of always taking the top token.

Each run takes about 1.3 s. This is the baseline's text. The GPT in step 9 reads many tokens back, not one.
