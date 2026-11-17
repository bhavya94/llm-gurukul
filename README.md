# llm-gurukul

A small language model goes to school: one model, trained by hand, learns eight abilities, from writing text to
resisting attacks.

A gurukul is a school where the student lives and learns. Here the student is `medha`, one small language model.
Each folder teaches it one ability, and each ability builds on the ones before it.

| Name | Means | For the reader |
|---|---|---|
| `vak` | speech | A small GPT learns to write text, one token at a time. |
| `shruti` | listening | The model learns to follow an instruction and to stop when it has answered. |
| `viveka` | discernment | The model makes quick typed decisions, and says when it is not sure. |
| `karma` | action | A program around the model lets it use tools, step by step, to finish a task. |
| `pariksha` | examination | Tests measure what the agent gets right and show why it fails. |
| `shodhana` | refinement | The agent improves through better prompts and tools, with the model unchanged. |
| `abhyasa` | practice | The model itself improves by training on its own attempts. |
| `raksha` | protection | Attacks test the agent, and guards make it harder to misuse. |
| `medha` | intellect | The model that gains all eight abilities. |
| `anu` | atom | A tiny copy of the model, trained from scratch and small enough to check by hand. |

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
