"""empty message

Revision ID: 5042b6d27005
Revises: 
Create Date: 2023-11-13 19:55:28.039625

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5042b6d27005'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    pass  # No changes to the database schema


def downgrade():
    pass  # No changes to the database schema
