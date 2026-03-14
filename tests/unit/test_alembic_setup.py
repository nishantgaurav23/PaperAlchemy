"""Tests for S9b.1: Alembic migration setup.

Validates Alembic config, env.py, initial migration, Makefile targets,
and model registration — all structural/config tests, no DB needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from alembic.config import Config as AlembicConfig

# Project root (tests/unit/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestAlembicIni:
    """FR-1: alembic.ini exists with correct configuration."""

    def test_alembic_ini_exists(self) -> None:
        ini_path = PROJECT_ROOT / "alembic.ini"
        assert ini_path.exists(), "alembic.ini must exist at project root"

    def test_alembic_ini_script_location(self) -> None:
        ini_path = PROJECT_ROOT / "alembic.ini"
        content = ini_path.read_text()
        assert "script_location = alembic" in content, "script_location must point to 'alembic' directory"

    def test_alembic_config_loads(self) -> None:
        ini_path = PROJECT_ROOT / "alembic.ini"
        config = AlembicConfig(str(ini_path))
        assert config.get_main_option("script_location") == "alembic"


class TestAlembicDirectory:
    """FR-1: Alembic directory structure."""

    def test_alembic_dir_exists(self) -> None:
        alembic_dir = PROJECT_ROOT / "alembic"
        assert alembic_dir.is_dir(), "alembic/ directory must exist"

    def test_env_py_exists(self) -> None:
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        assert env_py.exists(), "alembic/env.py must exist"

    def test_versions_dir_exists(self) -> None:
        versions_dir = PROJECT_ROOT / "alembic" / "versions"
        assert versions_dir.is_dir(), "alembic/versions/ directory must exist"

    def test_script_mako_exists(self) -> None:
        mako = PROJECT_ROOT / "alembic" / "script.py.mako"
        assert mako.exists(), "alembic/script.py.mako template must exist"


class TestEnvPy:
    """FR-2: env.py imports Base.metadata and uses async engine."""

    def test_env_imports_base_metadata(self) -> None:
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        content = env_py.read_text()
        assert "Base" in content, "env.py must import Base"
        assert "target_metadata" in content, "env.py must set target_metadata"

    def test_env_imports_models(self) -> None:
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        content = env_py.read_text()
        # Must import models so autogenerate can detect them
        assert "src.models" in content or "src.db.base" in content, "env.py must import models or base for metadata discovery"

    def test_env_uses_async_engine(self) -> None:
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        content = env_py.read_text()
        assert "async" in content, "env.py must support async migrations"

    def test_env_reads_config_url(self) -> None:
        """env.py must get DB URL from src.config, not hardcoded."""
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        content = env_py.read_text()
        assert "get_settings" in content or "config.get_main_option" in content, (
            "env.py must read DB URL from settings or alembic config"
        )

    def test_env_no_hardcoded_password(self) -> None:
        env_py = PROJECT_ROOT / "alembic" / "env.py"
        content = env_py.read_text()
        assert "paperalchemy_secret" not in content, "env.py must not contain hardcoded passwords"


class TestInitialMigration:
    """FR-3: Initial migration for papers table."""

    def test_initial_migration_exists(self) -> None:
        versions_dir = PROJECT_ROOT / "alembic" / "versions"
        migration_files = list(versions_dir.glob("*.py"))
        # Filter out __pycache__ and __init__
        migration_files = [f for f in migration_files if not f.name.startswith("__")]
        assert len(migration_files) >= 1, "At least one migration file must exist in alembic/versions/"

    def test_initial_migration_creates_papers_table(self) -> None:
        versions_dir = PROJECT_ROOT / "alembic" / "versions"
        migration_files = [f for f in versions_dir.glob("*.py") if not f.name.startswith("__")]
        assert len(migration_files) >= 1, "Need at least one migration file"

        # Read the first migration and check it creates 'papers' table
        first_migration = sorted(migration_files)[0]
        content = first_migration.read_text()
        assert "papers" in content, "Initial migration must reference 'papers' table"
        assert "def upgrade" in content, "Migration must have upgrade() function"
        assert "def downgrade" in content, "Migration must have downgrade() function"

    def test_initial_migration_has_key_columns(self) -> None:
        versions_dir = PROJECT_ROOT / "alembic" / "versions"
        migration_files = [f for f in versions_dir.glob("*.py") if not f.name.startswith("__")]
        assert len(migration_files) >= 1

        first_migration = sorted(migration_files)[0]
        content = first_migration.read_text()
        # Check for key column names
        for col in ["arxiv_id", "title", "abstract", "authors", "published_date"]:
            assert col in content, f"Migration must include '{col}' column"


class TestMakefileTargets:
    """FR-4: Makefile has db-migrate, db-upgrade, db-downgrade targets."""

    def test_makefile_has_db_migrate(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert re.search(r"^db-migrate:", content, re.MULTILINE), "Makefile must have db-migrate target"

    def test_makefile_has_db_upgrade(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert re.search(r"^db-upgrade:", content, re.MULTILINE), "Makefile must have db-upgrade target"

    def test_makefile_has_db_downgrade(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert re.search(r"^db-downgrade:", content, re.MULTILINE), "Makefile must have db-downgrade target"

    def test_db_migrate_uses_autogenerate(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert "--autogenerate" in content, "db-migrate must use --autogenerate flag"

    def test_db_upgrade_targets_head(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert "upgrade head" in content, "db-upgrade must target 'head'"

    def test_db_downgrade_reverts_one(self) -> None:
        makefile = PROJECT_ROOT / "Makefile"
        content = makefile.read_text()
        assert "downgrade -1" in content, "db-downgrade must revert one step (-1)"


class TestModelRegistration:
    """FR-5: Models are importable via src.models for autogenerate."""

    def test_models_init_imports_paper(self) -> None:
        from src.models import Paper

        assert Paper.__tablename__ == "papers"

    def test_paper_in_models_all(self) -> None:
        import src.models

        assert "Paper" in src.models.__all__

    def test_base_metadata_has_papers_table(self) -> None:
        """Base.metadata must know about the papers table for autogenerate."""
        # Force model import
        import src.models  # noqa: F401
        from src.db.base import Base

        assert "papers" in Base.metadata.tables, "Base.metadata must contain 'papers' table"
