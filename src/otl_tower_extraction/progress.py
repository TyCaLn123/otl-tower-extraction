from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter
from typing import Iterable, Optional, TypeVar
from tqdm import tqdm
T = TypeVar('T')

class ProgressReporter:
    def __init__(self, enabled: bool = True, stage_messages: bool = True):
        self.enabled = enabled
        self.stage_messages = stage_messages
    def log(self, message: str) -> None:
        if self.stage_messages:
            print(message, flush=True)
    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        self.log(f"\n[Stage] {name} ...")
        try:
            yield
        finally:
            self.log(f"[Done] {name} ({perf_counter() - start:.2f} s)")
    def iter(self, iterable: Iterable[T], desc: str, total: Optional[int] = None):
        return tqdm(iterable, desc=desc, total=total, leave=False) if self.enabled else iterable
