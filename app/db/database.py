"""
Database engine + session management. Uses Alembic for migrations instead
of create_all. Swaps between SQLite (prototype) and Postgres (production)
via DATABASE_URL env var.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core import config
from app.db.models import Base

_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

if config.DATABASE_URL.startswith("sqlite"):
    # WAL lets readers (status polls) proceed while the background enrollment
    # worker holds a write transaction — without this, /status hangs and the
    # progress bar freezes during the worker's DB writes. busy_timeout makes
    # writers wait briefly for a lock instead of failing immediately.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


def _reconcile_sqlite_columns():
    """Prototype safety net: SQLite create_all never ALTERs existing tables,
    so a model column added after the DB was first created is missing and any
    query/insert touching it crashes with 'no such column'. Add such columns
    in place. (Postgres uses real Alembic migrations instead.)"""
    from sqlalchemy import text
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table.name})"))}
            if not existing:
                continue  # table doesn't exist yet; create_all handles it
            for col in table.columns:
                if col.name not in existing:
                    coltype = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {coltype}'))


def init_db():
    """Run Alembic migrations programmatically on startup. Falls back to
    create_all only for SQLite test databases."""
    if config.DATABASE_URL.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
        _reconcile_sqlite_columns()
    else:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
