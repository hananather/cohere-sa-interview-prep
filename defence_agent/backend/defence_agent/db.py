from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from defence_agent.config import get_settings


def _sqlite_connect_args(database_url: str) -> dict[str, bool]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


settings = get_settings()
engine = create_engine(settings.database_url, connect_args=_sqlite_connect_args(settings.database_url))


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _ensure_column(session, "document", "doc_family", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "document", "status", "TEXT NOT NULL DEFAULT 'approved'")
        _ensure_column(session, "document", "owner", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "document", "review_due", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "document", "language", "TEXT NOT NULL DEFAULT 'en'")
        _ensure_column(session, "document", "source_type", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "document", "checksum", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "chunk", "doc_family", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "chunk", "status", "TEXT NOT NULL DEFAULT 'approved'")
        _ensure_column(session, "chunk", "owner", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "chunk", "review_due", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "chunk", "language", "TEXT NOT NULL DEFAULT 'en'")
        _ensure_column(session, "chunk", "source_type", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(session, "chunk", "row_id", "TEXT")
        session.exec(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                USING fts5(chunk_id UNINDEXED, title, section, text, summary, keywords)
                """
            )
        )
        session.commit()


def _ensure_column(session: Session, table_name: str, column_name: str, definition: str) -> None:
    columns = {row[1] for row in session.exec(text(f"PRAGMA table_info({table_name})")).all()}
    if column_name not in columns:
        session.exec(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def db_path() -> Path | None:
    if settings.database_url.startswith("sqlite:///"):
        return Path(settings.database_url.replace("sqlite:///", "", 1))
    return None
