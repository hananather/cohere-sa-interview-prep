from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlmodel import Session, select

from defence_agent.config import get_settings
from defence_agent.db import engine
from defence_agent.models import TraceRecord, TraceSpan


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class TraceManager:
    def __init__(self) -> None:
        self.settings = get_settings()

    def new_trace_id(self) -> str:
        return f"tr_{uuid.uuid4().hex}"

    def start_trace(self, trace_id: str, user_id: str, request: dict[str, Any]) -> None:
        record = TraceRecord(trace_id=trace_id, user_id=user_id, request_json=json.dumps(request, default=_json_default))
        with Session(engine) as session:
            session.add(record)
            session.commit()
        self._append_jsonl({"type": "trace_started", "trace_id": trace_id, "user_id": user_id, "request": request})

    def finish_trace(
        self,
        trace_id: str,
        status: str,
        response: dict[str, Any] | None = None,
        route: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        with Session(engine) as session:
            record = session.get(TraceRecord, trace_id)
            if record:
                record.status = status
                record.route = route or record.route
                record.response_json = json.dumps(response or {}, default=_json_default)
                record.duration_ms = duration_ms
                session.add(record)
                session.commit()
        self._append_jsonl(
            {
                "type": "trace_finished",
                "trace_id": trace_id,
                "status": status,
                "route": route,
                "duration_ms": duration_ms,
            }
        )

    def set_route(self, trace_id: str, route: str) -> None:
        with Session(engine) as session:
            record = session.get(TraceRecord, trace_id)
            if record:
                record.route = route
                session.add(record)
                session.commit()

    @contextmanager
    def span(self, trace_id: str, name: str, attributes: dict[str, Any] | None = None) -> Iterator[TraceSpan]:
        started = time.perf_counter()
        span = TraceSpan(
            trace_id=trace_id,
            name=name,
            started_at=datetime.now(timezone.utc),
            attributes_json=json.dumps(attributes or {}, default=_json_default),
        )
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = str(exc)
            raise
        finally:
            span.ended_at = datetime.now(timezone.utc)
            span.duration_ms = (time.perf_counter() - started) * 1000
            event = {
                "type": "span",
                "trace_id": trace_id,
                "name": name,
                "status": span.status,
                "duration_ms": span.duration_ms,
                "attributes": json.loads(span.attributes_json),
                "error": span.error,
            }
            with Session(engine) as session:
                session.add(span)
                session.commit()
            self._append_jsonl(event)

    def add_span(
        self,
        trace_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        span = TraceSpan(
            trace_id=trace_id,
            name=name,
            started_at=now,
            ended_at=now,
            duration_ms=0,
            status=status,
            attributes_json=json.dumps(attributes or {}, default=_json_default),
            error=error,
        )
        with Session(engine) as session:
            session.add(span)
            session.commit()
        self._append_jsonl(
            {
                "type": "span",
                "trace_id": trace_id,
                "name": name,
                "status": status,
                "duration_ms": 0,
                "attributes": attributes or {},
                "error": error,
            }
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with Session(engine) as session:
            record = session.get(TraceRecord, trace_id)
            if not record:
                return None
            spans = session.exec(select(TraceSpan).where(TraceSpan.trace_id == trace_id).order_by(TraceSpan.started_at)).all()
        return {
            "trace_id": record.trace_id,
            "user_id": record.user_id,
            "route": record.route,
            "status": record.status,
            "request": json.loads(record.request_json),
            "response": json.loads(record.response_json or "{}"),
            "created_at": record.created_at.isoformat(),
            "duration_ms": record.duration_ms,
            "spans": [
                {
                    "name": span.name,
                    "status": span.status,
                    "duration_ms": span.duration_ms,
                    "attributes": json.loads(span.attributes_json or "{}"),
                    "error": span.error,
                }
                for span in spans
            ],
        }

    def _append_jsonl(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with self.settings.trace_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")


trace_manager = TraceManager()
