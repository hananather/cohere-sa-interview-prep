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
        session.exec(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                USING fts5(chunk_id UNINDEXED, title, section, text, summary, keywords)
                """
            )
        )
        session.commit()


@contextmanager
def session_scope() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def db_path() -> Path | None:
    if settings.database_url.startswith("sqlite:///"):
        return Path(settings.database_url.replace("sqlite:///", "", 1))
    return None
