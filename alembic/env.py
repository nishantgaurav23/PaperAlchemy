"""Alembic environment configuration for async PostgreSQL migrations.

Reads the database URL from src.config (environment variables / .env),
imports all ORM models via src.models so autogenerate detects them,
and runs migrations using SQLAlchemy's async engine.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

# Import all models so Base.metadata is fully populated for autogenerate.
import src.models  # noqa: F401
from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from src.config import get_settings
from src.db.base import Base

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# Set the SQLAlchemy URL from application config (not hardcoded in alembic.ini).
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.postgres.url)

# Configure Python logging from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData target for autogenerate support.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without a DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — delegates to async runner."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
