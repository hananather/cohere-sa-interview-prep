from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_seconds: int = 30
    failures: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at > self.reset_seconds:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()

    def call(self, func: Callable[[], T]) -> T:
        if not self.allow():
            raise RuntimeError(f"Circuit breaker open for {self.name}")
        try:
            result = func()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


cohere_breaker = CircuitBreaker("cohere")
qdrant_breaker = CircuitBreaker("qdrant")
sandbox_breaker = CircuitBreaker("sandbox")
