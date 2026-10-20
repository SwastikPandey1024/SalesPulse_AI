"""
SalesPulse AI - Persistent Storage Engine
Provides:
- Thread-safe SQLAlchemy engine and session factory with SQLite WAL mode.
- Configurable DATABASE_URL and connection timeouts.
- Versioned, transactional schema initialization and migrations.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("salespulse.storage")

DEFAULT_DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "salespulse_audit.db")
)
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_FILE}"
SCHEMA_VERSION = 2

Base = declarative_base()


def get_database_url() -> str:
    """Retrieve database URL from environment or return default SQLite path."""
    return os.environ.get("SALESPULSE_DATABASE_URL", os.environ.get("DATABASE_URL", DEFAULT_DB_URL))


def configure_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable SQLite WAL mode, busy timeouts, and foreign key constraints."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
    finally:
        cursor.close()


def create_storage_engine(db_url: Optional[str] = None, busy_timeout_ms: int = 5000) -> Engine:
    """Create a thread-safe SQLAlchemy engine with SQLite hardening if applicable."""
    url = db_url or get_database_url()
    is_sqlite = url.startswith("sqlite")

    connect_args = {}
    if is_sqlite:
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = busy_timeout_ms / 1000.0

    engine = create_engine(
        url,
        connect_args=connect_args,
        future=True,
    )

    if is_sqlite:
        event.listen(engine, "connect", configure_sqlite_pragmas)

    return engine


_DEFAULT_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create singleton storage engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = create_storage_engine()
    return _DEFAULT_ENGINE


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Get or create session factory for engine."""
    global _SESSION_FACTORY
    eng = engine or get_engine()
    if _SESSION_FACTORY is None or engine is not None:
        factory = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
        if engine is None:
            _SESSION_FACTORY = factory
        return factory
    return _SESSION_FACTORY


@contextmanager
def get_db_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager yielding a transactional database session."""
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error; rolled back: {e}")
        raise
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None, drop_all: bool = False) -> int:
    """Initialize versioned database schema transactionally.
    
    Returns:
        int: Active schema version.
    """
    from salespulse.storage.models import Base  # Local import to avoid circular dependency
    
    eng = engine or get_engine()
    is_sqlite = eng.url.drivername.startswith("sqlite")

    with eng.begin() as conn:
        if drop_all:
            Base.metadata.drop_all(conn)
            if is_sqlite:
                conn.execute(text("PRAGMA user_version = 0;"))
            logger.warning("Database schema dropped.")

        current_version = 0
        if is_sqlite:
            res = conn.execute(text("PRAGMA user_version;")).fetchone()
            current_version = res[0] if res else 0

        if current_version == 0:
            Base.metadata.create_all(conn)
            if is_sqlite:
                conn.execute(text(f"PRAGMA user_version = {SCHEMA_VERSION};"))
            current_version = SCHEMA_VERSION
            logger.info(f"Initialized database schema version {SCHEMA_VERSION}")
        elif current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than the application's "
                f"supported schema version {SCHEMA_VERSION}. Please upgrade the application."
            )
        elif current_version < SCHEMA_VERSION:
            if current_version == 1:
                # Migration v1 -> v2: Add anomaly_records table
                Base.metadata.create_all(conn)
                if is_sqlite:
                    conn.execute(text(f"PRAGMA user_version = {SCHEMA_VERSION};"))
                current_version = SCHEMA_VERSION
                logger.info(f"Migrated database schema from version 1 to {SCHEMA_VERSION}")
            else:
                Base.metadata.create_all(conn)
                if is_sqlite:
                    conn.execute(text(f"PRAGMA user_version = {SCHEMA_VERSION};"))
                current_version = SCHEMA_VERSION
                logger.info(f"Upgraded database schema to {SCHEMA_VERSION}")

    return current_version
