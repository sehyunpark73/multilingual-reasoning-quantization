from collections.abc import Callable, Iterator
import math
import random
from statistics import mean

from torch import Tensor
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

LOG_INTERVAL_STEPS = 16


def train(
    weights: list[Tensor],
    samples: list,
    compute_losses: Callable[[list], Iterator[Tensor]],
    learning_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
) -> None:
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=learning_rate, weight_decay=0)
    num_batches = math.ceil(len(samples) / batch_size)
    num_steps = num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio * num_steps), num_steps)

    batch_losses = []
    for epoch in range(num_epochs):
        random.shuffle(samples)
        for batch_index, batch_start in enumerate(range(0, len(samples), batch_size)):
            batch_loss = 0.0
            for sample_loss in compute_losses(samples[batch_start : batch_start + batch_size]):
                sample_loss.backward()
                batch_loss += sample_loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            batch_losses.append(batch_loss)
            step = epoch * num_batches + batch_index + 1
            if step == 1 or step % LOG_INTERVAL_STEPS == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(batch_losses):.4f} learning_rate {scheduler.get_last_lr()[0]:.2e}")
                batch_losses = []
