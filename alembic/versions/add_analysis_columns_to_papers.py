"""Add summary, highlights, methodology columns to papers table.

Revision ID: a3c8f1b2d4e6
Revises: 7fa2bf4a9f17
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "a3c8f1b2d4e6"
down_revision = "7fa2bf4a9f17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("summary", sa.JSON(), nullable=True))
    op.add_column("papers", sa.Column("highlights", sa.JSON(), nullable=True))
    op.add_column("papers", sa.Column("methodology", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "methodology")
    op.drop_column("papers", "highlights")
    op.drop_column("papers", "summary")
