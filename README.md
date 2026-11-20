# llm-gurukul

In this repo, I am going to attempt training my own "student" LLM through different stages of training, from writing text to resisting attacks. I'll call this model "Medha", which means "intellect" in Sanskrit.

Gurukul = a school in the Vedic tradition.

Training an LLM typically involves taking it through different stages. A randomly initialized model first learns "the internet". In other words, it uses a vocabulary of tokens (pieces of text, picked using BPE etc.), learns an embedding for each token (this is what captures semantic meaning) and trains a decoder model (a stack of attention and MLP layers) that allows it to predict the probability of each token in its vocabulary being the next token. It then generates text one token at a time, hence it's also called a "generative" AI model. Eventually this model (or a student) learns how to communicate with humans via Instruction Tuning (a form of Supervised Fine-Tuning, or SFT) and then specializes in different domains (e.g. coding) via a combination of SFT and Reinforcement Learning (often with human feedback + a reward model involved). At each stage of its training, it's evaluated using different benchmarks (or "exams").

This process has parallels to a student going to a school, learning different skills and then appearing for standardized tests to assess how well they've learned those skills. Hence, I chose to call this repo "LLM Gurukul".

One honest limit: I am doing this exercise on an 8 GB MacBook Pro, so I cannot actually pretrain a model with hundreds of millions of parameters on a huge corpus of text, which requires trillions of tokens and many GPUs. So I will start by pretraining a tiny model from scratch (to learn how pretraining works). I'll call this tiny model "Anu", which means "atom". Then I will load an open model's pretrained weights into the same model code. That model becomes Medha, and I will primarily work with Medha from there.

Medha will learn the following skills or abilities in sequence, and each one will live in its own folder in this repo:

| Name | Means | For the reader |
|---|---|---|
| `vak` | speech | Anu is born as a small GPT and learns to write text, one token at a time (later Medha takes over, starting from an open model's pretrained weights). |
| `shruti` | listening | Medha now learns to follow an instruction and to stop when it has answered. |
| `viveka` | discernment | Medha makes quick typed decisions (inspired by Jev, TypeSafe's decision model) and says when it is not sure. |
| `karma` | action | Medha is now used with a harness: a program around the model that lets it use tools, step by step, to finish a task. |
| `pariksha` | examination | Tests measure what the Medha agent gets right and show why it fails. |
| `shodhana` | refinement | The agent improves through better prompts and tools, with the model unchanged. |
| `abhyasa` | practice | Medha itself improves by training on its own attempts. |
| `raksha` | protection | Attacks test the agent, and guards make it harder to misuse. |

And the two students that go through them:

| Name | Means | For the reader |
|---|---|---|
| `anu` | atom | The tiny student I start with: trained from scratch on my MacBook and small enough to check by hand. It stays around after Medha takes over. |
| `medha` | intellect | The main student: the same model code with an open model's pretrained weights. It takes over from Anu and gains all eight abilities. |

## Where it stands

`vak` is being built, one concept per pull request. The other abilities follow, in the order of the table.

## Run it

Needs [uv](https://docs.astral.sh/uv/). It installs Python 3.12 if you don't have it.

```console
uv sync
uv run pytest
uv run vak --help
```

## Layout

The folders match the names in the table.

- `medha/`: the parts of the model that every ability shares. It never imports an ability; a test checks that.
- `vak/`: the first ability. Its command, its notes and its notebooks. Start at [vak/README.md](vak/README.md).
- `tests/`: the tests for both.
