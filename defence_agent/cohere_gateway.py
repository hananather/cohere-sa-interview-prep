"""Minimal real Cohere gateway for Embed v4, Rerank v4, and Chat."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from defence_agent.config import get_settings
from defence_agent.resilience import cohere_breaker


T = TypeVar("T")


class CohereRateLimiter:
    """Process-local pacing for billable Cohere API calls."""

    def __init__(self, requests_per_minute: float) -> None:
        self.min_interval_seconds = 60.0 / max(float(requests_per_minute), 0.01)
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = self.min_interval_seconds - (now - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()


class CohereGateway:
    """Small adapter around live Cohere calls the ADK agent needs."""

    def __init__(self) -> None:
        self.settings = get_settings()

        import cohere

        self._client: Any = cohere.ClientV2(api_key=self.settings.cohere_api_key)
        self._rate_limiter = CohereRateLimiter(self.settings.cohere_requests_per_minute)

    def embedding_size(self) -> int:
        return self.settings.cohere_embed_output_dimension

    def embed_texts(self, texts: list[str], input_type: str) -> list[list[float]]:
        return cohere_breaker.call(lambda: self._embed_real(texts=texts, inputs=None, input_type=input_type))

    def embed_inputs(self, inputs: list[dict[str, Any]], input_type: str) -> list[list[float]]:
        return cohere_breaker.call(lambda: self._embed_real(texts=None, inputs=inputs, input_type=input_type))

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], input_type="search_query")[0]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return cohere_breaker.call(lambda: self._rerank_real(query, documents))

    def chat_stream(self, **kwargs: Any) -> list[Any]:
        return cohere_breaker.call(lambda: self._chat_stream_real(**kwargs))

    def chat(self, **kwargs: Any) -> Any:
        return cohere_breaker.call(lambda: self._chat_real(**kwargs))

    def _embed_real(
        self,
        *,
        texts: list[str] | None,
        inputs: list[dict[str, Any]] | None,
        input_type: str,
    ) -> list[list[float]]:
        def call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.settings.cohere_embed_model,
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": self.settings.cohere_embed_output_dimension,
                "request_options": self._request_options(),
            }
            if inputs is not None:
                kwargs["inputs"] = inputs
            else:
                kwargs["texts"] = texts or []
            return self._client.embed(**kwargs)

        response = self._call_with_retries(call)
        embeddings = getattr(response.embeddings, "float", None) or response.embeddings["float"]
        return [list(vector) for vector in embeddings]

    def _rerank_real(self, query: str, documents: list[str]) -> list[float]:
        def call() -> Any:
            return self._client.rerank(
                model=self.settings.cohere_rerank_model,
                query=query,
                documents=documents,
                top_n=len(documents),
                request_options=self._request_options(),
            )

        response = self._call_with_retries(call)
        scores = [0.0] * len(documents)
        for item in response.results:
            index = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)
            if index is not None and score is not None:
                scores[index] = float(score)
        return scores

    def _chat_stream_real(self, **kwargs: Any) -> list[Any]:
        def call() -> list[Any]:
            kwargs["request_options"] = self._request_options()
            return list(self._client.chat_stream(**kwargs))

        return self._call_with_retries(call)

    def _chat_real(self, **kwargs: Any) -> Any:
        def call() -> Any:
            kwargs["request_options"] = self._request_options()
            return self._client.chat(**kwargs)

        return self._call_with_retries(call)

    def _call_with_retries(self, call: Callable[[], T]) -> T:
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable_cohere_error),
            wait=wait_exponential(multiplier=1, min=1, max=self.settings.cohere_retry_max_wait_seconds),
            stop=stop_after_attempt(self.settings.cohere_max_retries),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                self._rate_limiter.wait()
                return call()
        raise RuntimeError("Cohere retry loop exited without returning")

    def _request_options(self) -> dict[str, float]:
        return {"timeout_in_seconds": self.settings.cohere_timeout_seconds}


def _is_retryable_cohere_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    retryable_fragments = [
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "remoteprotocolerror",
        "incomplete chunked read",
        "peer closed connection",
    ]
    return any(fragment in text for fragment in retryable_fragments)


cohere_gateway = CohereGateway()
