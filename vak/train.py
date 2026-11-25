"""Training: the GPT learns to predict each next token of the training stories. One step reads a batch of windows
from the stories, scores the next token at every place, and moves every weight a little to lower that loss."""

import logging
import time

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from medha.model import GPT
from medha.sampler import generate
from medha.tokenizer import END

log = logging.getLogger(__name__)

BATCH = 64  # windows per step: 64 x 128 = 8,192 tokens
LEARNING_RATE = 1e-3  # for AdamW
LOG_AT = (0, 100, 1_000)  # the steps that log the loss and a sample, besides the last
PROMPT = "Once upon a time"


def join(stories: list[list[int]]) -> torch.Tensor:
    """The training stories as one long row of ids, with a single end token between two stories. Each story comes
    with an end token before and after it, so each one gives up its last end token, except the last story."""
    ids = []
    for story in stories:
        ids.extend(story[:-1])
    ids.append(stories[-1][-1])
    return torch.tensor(ids, dtype=torch.int16)  # 2 bytes per id: the vocabulary is far below 32,768


def batch(stream: torch.Tensor, size: int, context: int,
          generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """size windows of context ids, from random places in the stream. The targets are the same windows moved one
    token on, so at every place the model must predict the next id."""
    starts = torch.randint(len(stream) - context, (size,), generator=generator)
    inputs = []
    targets = []
    for start in starts.tolist():
        inputs.append(stream[start:start + context])
        targets.append(stream[start + 1:start + context + 1])
    return torch.stack(inputs).long(), torch.stack(targets).long()


def sample(model: GPT, tokenizer: Tokenizer, tokens: int, seed: int) -> str:
    """The prompt and what the model writes after it, at temperature 1."""
    end = tokenizer.token_to_id(END)
    prompt = tokenizer.encode(PROMPT).ids
    steps = generate(model, [end] + prompt, tokens, end, torch.Generator().manual_seed(seed))
    written = []
    for step in steps:
        if step.token != end:
            written.append(step.token)
    return tokenizer.decode(prompt + written)


def train(model: GPT, stream: torch.Tensor, steps: int, tokenizer: Tokenizer, seed: int) -> None:
    """AdamW on random windows of the stream, for steps steps. At each step in LOG_AT, and at the last, it logs the
    mean loss since the last log and a short sample."""
    device = model.head.weight.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    context = model.config.context
    started = time.time()
    total = torch.zeros((), device=device)  # summed on the device, so the GPU never waits for a print
    count = 0
    for step in range(steps + 1):
        inputs, targets = batch(stream, BATCH, context, generator)
        logits = model(inputs.to(device))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1))
        total += loss.detach()
        count += 1
        if step in LOG_AT or step == steps:
            log.info("step %5d: loss %.3f, the mean of the last %d batches; %.0f s", step, float(total / count),
                     count, time.time() - started)
            log.info("  %s", sample(model, tokenizer, 30, seed).replace("\n", " "))
            total = torch.zeros((), device=device)
            count = 0
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()  # the gradient: which way each weight should move to lower the loss
        optimizer.step()


def story_losses(model: GPT, ids: list[int]) -> list[float]:
    """For each token after the first: minus the log of the chance the model gave it, as PairCounts.losses does, so
    both models score exactly the same tokens. A story longer than the window is read in windows that overlap by
    half. The first window scores all its tokens; each later window scores only its new half, so every token is
    scored once, with at least half a window of tokens before it."""
    context = model.config.context
    device = model.head.weight.device
    losses = []
    start = 0
    with torch.no_grad():
        while len(losses) < len(ids) - 1:
            window = ids[start:start + context + 1]
            logits = model(torch.tensor([window[:-1]], device=device))[0]
            targets = torch.tensor(window[1:], device=device)
            minus_log = F.cross_entropy(logits, targets, reduction="none").tolist()
            already = len(losses) - start  # tokens of this window that an earlier window scored
            losses.extend(minus_log[already:])
            start += context // 2
    return losses
