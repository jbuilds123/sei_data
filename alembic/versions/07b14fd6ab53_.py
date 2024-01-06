"""empty message

Revision ID: 07b14fd6ab53
Revises: 5042b6d27005
Create Date: 2023-11-13 20:04:53.522385

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '07b14fd6ab53'
down_revision: Union[str, None] = '5042b6d27005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add the lock_data column to the newpairs table
    op.add_column('newpairs', sa.Column('lock_data', sa.JSON, nullable=True))


def downgrade():
    # Remove the lock_data column from the newpairs table
    op.drop_column('newpairs', 'lock_data')
