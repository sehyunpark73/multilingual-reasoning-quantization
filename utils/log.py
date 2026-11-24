from collections.abc import Generator
from contextlib import contextmanager
import time

import torch


def _format_duration(
    duration: float,
) -> str:
    minute, second = divmod(round(duration), 60)
    return f"{minute}m{second:02d}s"


def _format_memory(
    num_bytes: int,
) -> str:
    return f"{num_bytes / 2**30:.1f}GB"


@contextmanager
def log_time(
    name: str,
) -> Generator[None]:
    start_time = time.perf_counter()
    yield
    print(f"{name} time {_format_duration(time.perf_counter() - start_time)}")


@contextmanager
def log_time_and_memory(
    name: str,
) -> Generator[None]:
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    yield
    print(f"{name} time {_format_duration(time.perf_counter() - start_time)} memory {_format_memory(torch.cuda.max_memory_allocated())}")
