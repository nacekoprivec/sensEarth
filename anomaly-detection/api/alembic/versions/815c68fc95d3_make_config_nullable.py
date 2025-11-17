"""make config nullable

Revision ID: 815c68fc95d3
Revises: 67d3c21fc6a2
Create Date: 2025-09-15 12:00:38.863072

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '815c68fc95d3'
down_revision: Union[str, Sequence[str], None] = '67d3c21fc6a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
